import os
import json

import cv2
import torch
import random
import logging
import argparse
import numpy as np
from skimage import measure
from tabulate import tabulate
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve
import yaml
import sys
import open_clip
from few_shot import memory
from model import LinearLayer
from dataset import VisaDataset, MVTecDataset,PCBBankDataset
from prompt_ensemble import encode_text_with_prompt_ensemble
from tqdm import *
from few_shot import memory
from modules.scoring import VPKscoring,LPKscoring,genregion_feature,VL_seg
from modules.ICTR import ICTR

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)

def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def test(args):
    img_size = args.image_size
    features_list = args.features_list
    few_shot_features = args.few_shot_features
    dataset_dir = args.test_data_path
    save_path = args.save_path
    dataset_name = args.dataset_test
    device=args.device

    filename = dataset_name +  '_' + str(args.k_shot)+"shot"

    save_path=os.path.join(save_path,filename)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    txt_name='log_seed_'+str(args.seed) +'_'+str(args.k_shot)+"shot"+".txt"
    txt_path = os.path.join(save_path, txt_name)


    mask_info = {'mask_th': args.mask_th, 'mask_weight': args.mask_weight, 'mask_topk': args.mask_topk,
                 'mask_layers': args.mask_layers,'mask_attention': True}
    region_alpha = args.region_alpha

    # clip
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, img_size, pretrained=args.pretrained)
    model.to(device)
    tokenizer = open_clip.get_tokenizer(args.model)

    # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('test')
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

    # seg
    with open(args.clip_config_path, 'r') as f:
        model_configs = json.load(f)
    linearlayer = LinearLayer(model_configs['vision_cfg']['width'], model_configs['embed_dim'],
                              len(features_list), args.model).to(device)
    checkpoint = torch.load(args.checkpoint_path)
    linearlayer.load_state_dict(checkpoint["trainable_linearlayer"])


    # dataset
    transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ])
    if dataset_name == 'mvtec':
        test_data = MVTecDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                                 aug_rate=-1, mode='test')
    elif dataset_name == 'visa':
        test_data = VisaDataset(root=dataset_dir, transform=preprocess, target_transform=transform, mode='test')
    else :
        test_data = PCBBankDataset(root=dataset_dir, transform=preprocess, target_transform=transform, mode='test')
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.get_cls_names()

    mem_features, mem_cls_features, _= memory(args.model, model, obj_list, dataset_dir, save_path, preprocess,
                                                transform,
                                                args.k_shot, few_shot_features, dataset_name, device,weighted=False,mask_info=None)

    # text prompt
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_prompts = encode_text_with_prompt_ensemble(model, obj_list, tokenizer, device)

    results = {}
    results['cls_names'] = []
    results['imgs_masks'] = []
    results['anomaly_maps'] = []
    results['gt_sp'] = []
    results['pr_sp'] = []
    results['pr_sp_fewshot'] = []
    for items in tqdm(test_dataloader):
        image = items['img'].to(device)
        cls_name = items['cls_name']
        results['cls_names'].append(cls_name[0])
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.25], gt_mask[gt_mask <= 0.25] = 1, 0
        results['imgs_masks'].append(gt_mask)  # px
        results['gt_sp'].append(items['anomaly'].item())

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features, patch_tokens = model.encode_image(image, features_list,weighted=mask_info['mask_attention'],mask_info=mask_info)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features = []
            for cls in cls_name:
                text_features.append(text_prompts[cls])
            text_features = torch.stack(text_features, dim=0)

            text_probs = (100.0 * image_features @ text_features[0]).softmax(dim=-1)
            results['pr_sp'].append(text_probs[0][1].cpu().item())

            # pixel
            VPK_mask = VPKscoring(torch.stack(patch_tokens, dim=1)[:, :, 1:, :], topk=mask_info['mask_topk'],threshold=mask_info['mask_th'])  # B*L*HW

            patch_tokens = linearlayer(patch_tokens)

            L = len(patch_tokens)
            B, HW, C = patch_tokens[0].shape
            H = int(np.sqrt(HW))

            for layer in range(0, L):
                patch_tokens[layer] = F.normalize(patch_tokens[layer], dim=-1)

            LPK_mask = LPKscoring(patch_tokens, text_features,threshold=mask_info['mask_th'])
            region_feature = genregion_feature(patch_tokens, VPK_mask, LPK_mask, device)
            anomaly_maps =  VL_seg(patch_tokens, text_features, region_feature, image_size=args.image_size, region_alpha=region_alpha)
            anomaly_maps = [m[:,1:,:,:].squeeze(1).cpu().numpy() for m in anomaly_maps]
            anomaly_map = np.sum(anomaly_maps, axis=0)

            # few shot
            class_tokens = []
            image_features, patch_tokens = model.encode_image(image, few_shot_features, weighted=False, mask_info=None)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            class_tokens.append(image_features.squeeze().cpu().numpy())

            anomaly_maps_few_shot = ICTR(patch_tokens, mem_features, mem_cls_features, cls_name, args)

            anomaly_map_few_shot = np.sum(anomaly_maps_few_shot, axis=0)
            anomaly_map = anomaly_map + anomaly_map_few_shot

            results['pr_sp_fewshot'].append(np.max(anomaly_map))
            results['anomaly_maps'].append(anomaly_map)

            

    # metrics
    table_ls = []
    auroc_sp_ls = []
    auroc_px_ls = []
    f1_sp_ls = []
    f1_px_ls = []
    aupro_ls = []
    ap_sp_ls = []
    ap_px_ls = []
    for obj in obj_list:
        table = []
        gt_px = []
        pr_px = []
        gt_sp = []
        pr_sp = []
        pr_sp_tmp = []
        table.append(obj)
        for idxes in range(len(results['cls_names'])):
            if results['cls_names'][idxes] == obj:
                gt_px.append(results['imgs_masks'][idxes].squeeze(1).numpy())
                pr_px.append(results['anomaly_maps'][idxes])
                
                pr_sp_tmp.append(results['pr_sp_fewshot'][idxes])
                gt_sp.append(results['gt_sp'][idxes])
                pr_sp.append(results['pr_sp'][idxes])
        gt_px = np.array(gt_px)
        gt_sp = np.array(gt_sp)
        pr_px = np.array(pr_px)
        pr_sp = np.array(pr_sp)
        
        pr_sp_tmp = np.array(pr_sp_tmp)
        pr_sp_tmp = (pr_sp_tmp - pr_sp_tmp.min()) / (pr_sp_tmp.max() - pr_sp_tmp.min())
        pr_sp = 0.5 * (pr_sp + pr_sp_tmp)

        auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel())
        auroc_sp = roc_auc_score(gt_sp, pr_sp)


        ap_sp = average_precision_score(gt_sp, pr_sp)
        ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())
        # f1_sp
        precisions, recalls, thresholds = precision_recall_curve(gt_sp, pr_sp)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        f1_sp = np.max(f1_scores[np.isfinite(f1_scores)])
        # f1_px
        precisions, recalls, thresholds = precision_recall_curve(gt_px.ravel(), pr_px.ravel())
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        f1_px = np.max(f1_scores[np.isfinite(f1_scores)])

        # aupro
        if len(gt_px.shape) == 4:
            gt_px = gt_px.squeeze(1)
        if len(pr_px.shape) == 4:
            pr_px = pr_px.squeeze(1)
        aupro = cal_pro_score(gt_px, pr_px)



        table.append(str(np.round(auroc_px * 100, decimals=1)))
        table.append(str(np.round(f1_px * 100, decimals=1)))
        table.append(str(np.round(ap_px * 100, decimals=1)))
        table.append(str(np.round(aupro * 100, decimals=1)))
        table.append(str(np.round(auroc_sp * 100, decimals=1)))
        table.append(str(np.round(f1_sp * 100, decimals=1)))
        table.append(str(np.round(ap_sp * 100, decimals=1)))

        table_ls.append(table)
        auroc_px_ls.append(auroc_px)
        auroc_sp_ls.append(auroc_sp)
        f1_sp_ls.append(f1_sp)
        f1_px_ls.append(f1_px)
        aupro_ls.append(aupro)
        ap_sp_ls.append(ap_sp)
        ap_px_ls.append(ap_px)

    # logger
    table_ls.append(['mean', str(np.round(np.mean(auroc_px_ls) * 100, decimals=3)),
                     str(np.round(np.mean(f1_px_ls) * 100, decimals=3)), str(np.round(np.mean(ap_px_ls) * 100, decimals=3)),
                     str(np.round(np.mean(aupro_ls) * 100, decimals=3)), str(np.round(np.mean(auroc_sp_ls) * 100, decimals=3)),
                     str(np.round(np.mean(f1_sp_ls) * 100, decimals=3)), str(np.round(np.mean(ap_sp_ls) * 100, decimals=3))])
    results = tabulate(table_ls, headers=['objects', 'auroc_px', 'f1_px', 'ap_px', 'aupro', 'auroc_sp',
                                          'f1_sp', 'ap_sp'], tablefmt="pipe")
    logger.info("\n%s", results)


if __name__ == '__main__':

    parser = argparse.ArgumentParser("VAND Challenge", add_help=True)
    # paths
    parser.add_argument("--test_data_path", type=str, default="./data/mvtec", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./vis/proposal', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='./exps/pretrained/visa_pretrained.pth', help='path to save results')
    parser.add_argument("--clip_config_path", type=str, default='./open_clip/model_configs/ViT-L-14-336.json', help="model configs")
    # model
    parser.add_argument("--dataset_test", type=str, default='mvtec', help="test dataset")
    parser.add_argument("--model", type=str, default="ViT-L-14-336", help="model used")
    parser.add_argument("--pretrained", type=str, default="openai", help="pretrained weight used")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6,12,18,24], help="features used")
    parser.add_argument("--few_shot_features", type=int, nargs="+", default=[6,12,18,24], help="features used for few shot")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    # few shot
    parser.add_argument("--k_shot", type=int, default=4, help="e.g., 10-shot, 5-shot, 1-shot")
    parser.add_argument("--seed", type=int, default=10, help="random seed")
    parser.add_argument("--lambd", type=float, default=0.5, help="lambd used")
    parser.add_argument("--device", type=str, default="cuda:0", help="device2run")
    parser.add_argument("--mask_th", type=float, default=0.5, help="mask threshold")
    parser.add_argument("--mask_topk", type=int, default=100, help="topk patch4mask ")
    parser.add_argument("--mask_weight", type=float, nargs="+",default=[1.2,0.8], help="mask weight")
    parser.add_argument("--mask_layers", type=int, nargs="+", default=[6,12,18,24], help="mask layers")
    parser.add_argument("--region_alpha", type=float, default=0.05, help="region feature weight")
    parser.add_argument("--config_path", type=str, default=None, help="Path to config.yaml")

    args, unknown = parser.parse_known_args()
    cli_args = {arg.lstrip('-').replace('-', '_') for arg in sys.argv[1:] if arg.startswith('--')}
    
    if args.config_path:
        cfg = load_config(args.config_path)
        for k, v in cfg.items():
            if hasattr(args, k) and k not in cli_args:
                setattr(args, k, v)
    setup_seed(args.seed)
    test(args)
