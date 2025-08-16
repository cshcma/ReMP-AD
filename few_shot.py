import torch
from dataset import VisaDataset, MVTecDataset, PCBBankDataset, MPDDDataset


def memory(model_name, model, obj_list, dataset_dir, save_path, preprocess, transform, k_shot,
           few_shot_features, dataset_name, device,weighted=False,mask_info=None):
    mem_features = {}
    cls_tokens={}
    image_feature_dict={}
    for obj in obj_list:
        if dataset_name == 'mvtec':
            data = MVTecDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                                aug_rate=-1, mode='train', k_shot=k_shot, save_dir=save_path, obj_name=obj)
        elif dataset_name == 'visa':
            data = VisaDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                               mode='train', k_shot=k_shot, save_dir=save_path, obj_name=obj)
        elif dataset_name == 'mpdd':
            data = MPDDDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                               mode='train', k_shot=k_shot, save_dir=save_path, obj_name=obj)
        elif dataset_name == 'pcbbank':
            data = PCBBankDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                               mode='train', k_shot=k_shot, save_dir=save_path, obj_name=obj)
        dataloader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
        features = []
        cls_features=[]
        image_feature=[]

        for items in dataloader:
            image = items['img'].to(device)
            with torch.no_grad():
                image_features, patch_tokens = model.encode_image(image, few_shot_features, weighted=weighted, mask_info=mask_info)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                if 'ViT' in model_name:
                    patch_tokens_ = [p[0, 1:, :] for p in patch_tokens]
                    cls_token=[p[0, 0:1, :] for p in patch_tokens]
                else:
                    patch_tokens_ = [p[0].view(p.shape[1], -1).permute(1, 0).contiguous() for p in patch_tokens]
                features.append(patch_tokens_)
                cls_features.append(cls_token)
                image_feature.append(image_features)
        mem_features[obj] = [torch.cat(
            [features[j][i] for j in range(len(features))], dim=0) for i in range(len(features[0]))]
        cls_tokens[obj]=[torch.cat(
            [cls_features[j][i] for j in range(len(features))], dim=0) for i in range(len(features[0]))]
        image_feature_dict[obj] = image_feature
    return mem_features ,cls_tokens ,image_feature_dict
