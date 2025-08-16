import torch
import numpy as np
import random
import os
import json
import argparse
from torch.nn import functional as F
import torchvision.transforms as transforms
import logging
import yaml
import sys
import open_clip
from dataset import VisaDataset, MVTecDataset
from model import LinearLayer
from loss import FocalLoss, BinaryDiceLoss
from prompt_ensemble import encode_text_with_prompt_ensemble
from tqdm import *
from modules.scoring import VPKscoring,LPKscoring,genregion_feature,VL_seg


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def train(args):
    # configs
    epochs = args.epoch
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    image_size = args.image_size
    device=args.device
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    txt_path = os.path.join(save_path, 'log.txt')  # log

    # model configs
    features_list = args.features_list
    with open(args.clip_config_path, 'r') as f:
        model_configs = json.load(f)

    mask_info = {'mask_th': args.mask_th, 'mask_weight': args.mask_weight, 'mask_topk': args.mask_topk,
                 'mask_layers': args.mask_layers,'mask_attention': True}
    region_alpha=args.region_alpha
    # clip model
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, image_size, pretrained=args.pretrained)
    model.to(device)
    tokenizer = open_clip.get_tokenizer(args.model)

    # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('train')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # record parameters
    for arg in vars(args):
        logger.info(f'{arg}: {getattr(args, arg)}')

    # transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])
    
    # datasets
    if args.dataset_train == 'mvtec':
        train_data = MVTecDataset(root=args.train_data_path, transform=preprocess, target_transform=transform,
                                  aug_rate=args.aug_rate)
    else:
        train_data = VisaDataset(root=args.train_data_path, transform=preprocess, target_transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # linear layer
    trainable_layer = LinearLayer(model_configs['vision_cfg']['width'], model_configs['embed_dim'],
                                  len(args.features_list), args.model).to(device)

    params = list(trainable_layer.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(0.5, 0.999))

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    # text prompt
    with torch.cuda.amp.autocast(), torch.no_grad():
        obj_list = train_data.get_cls_names()
        text_prompts = encode_text_with_prompt_ensemble(model, obj_list, tokenizer, device)

    for epoch in range(epochs):
        loss_list = []
        idx = 0
        for items in tqdm(train_dataloader):
            idx += 1
            image = items['img'].to(device)
            cls_name = items['cls_name']
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    image_features, patch_tokens = model.encode_image(image, features_list,weighted=mask_info['mask_attention'],mask_info=mask_info)
                    text_features = []
                    for cls in cls_name:
                        text_features.append(text_prompts[cls])
                    text_features = torch.stack(text_features, dim=0)
                    
                    VPK_mask = VPKscoring(torch.stack(patch_tokens, dim=1)[:, :, 1:, :], topk=mask_info['mask_topk'],threshold=mask_info['mask_th'])  # B*L*HW

                patch_tokens = trainable_layer(patch_tokens)

                L = len(patch_tokens)
                # B, HW, C = patch_tokens[0].shape
                # H = int(np.sqrt(HW))

                #normalize
                for layer in range(0, L):
                    patch_tokens[layer] = F.normalize(patch_tokens[layer], dim=-1)

                LPK_mask = LPKscoring(patch_tokens, text_features,threshold=mask_info['mask_th'])
                region_feature = genregion_feature(patch_tokens, VPK_mask, LPK_mask, device)
                anomaly_maps =  VL_seg(patch_tokens, text_features, region_feature, image_size=image_size, region_alpha=region_alpha)

            # losses
            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5], gt[gt <= 0.5] = 1, 0
            loss = 0
            for num in range(len(anomaly_maps)):
                loss += loss_focal(anomaly_maps[num], gt)
                loss += loss_dice(anomaly_maps[num][:, 1, :, :], gt)
                if torch.isnan(loss).any():
                    print("loss error")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))

        # save model
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(save_path, 'epoch_' + str(epoch + 1) + '.pth')
            torch.save({'trainable_linearlayer': trainable_layer.state_dict()}, ckp_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("VAND Challenge", add_help=True)
    # path
    parser.add_argument("--train_data_path", type=str, default="./data/mvtec", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./exps/tmp', help='path to save results')
    parser.add_argument("--clip_config_path", type=str, default='./open_clip/model_configs/ViT-L-14-336.json', help="clip_model configs")
    # model
    parser.add_argument("--dataset_train", type=str, default='mvtec', help="train dataset name")
    parser.add_argument("--model", type=str, default="ViT-L-14-336", help="model used")
    parser.add_argument("--pretrained", type=str, default="openai", help="pretrained weight used")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6,12,18,24], help="features used")
    # hyper-parameter
    parser.add_argument("--epoch", type=int, default=3, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--aug_rate", type=float, default=0.2, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--device", type=str, default="cuda:0", help="device2run")
    parser.add_argument("--mask_th", type=float, default=0.5, help="mask threshold")
    parser.add_argument("--mask_topk", type=int, default=100, help="topk patch4mask ")
    parser.add_argument("--mask_weight", type=float, nargs="+",default=[1.2,0.8], help="mask weight")
    parser.add_argument("--mask_layers", type=int, nargs="+", default=[6,12,18,24], help="mask layers")
    parser.add_argument("--region_alpha", type=float, default=0.2, help="region feature weight")
    parser.add_argument("--config_path", type=str, default=None, help="Path to config.yaml")
    
    args, unknown = parser.parse_known_args()
    cli_args = {arg.lstrip('-').replace('-', '_') for arg in sys.argv[1:] if arg.startswith('--')}
    
    if args.config_path:
        cfg = load_config(args.config_path)
        for k, v in cfg.items():
            if hasattr(args, k) and k not in cli_args:
                setattr(args, k, v)

    setup_seed(111)
    train(args)

