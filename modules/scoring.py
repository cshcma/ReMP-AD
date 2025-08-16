import torch
import numpy as np
import torch.nn.functional as F
from tqdm import *
import cv2

def sqrt(input):
    assert not torch.any(torch.isnan(input))
    signs = torch.sign(input)
    sq=torch.sqrt(torch.abs(input)+1e-8)
    assert not torch.any(torch.isnan(sq))
    res=signs*sq
    assert not torch.any(torch.isnan(res))
    return res

def VPKscoring(p,topk=1,threshold=0.5):
    p_norm=p/p.norm(dim=-1, keepdim=True)
    cos=p_norm @ p_norm.permute(0,1,3,2)
    topK_values = cos.topk(k=topk, dim=-1).values
    VPK_score = 1-torch.mean(topK_values, dim=-1)
    VPK_score[VPK_score > threshold], VPK_score[VPK_score <= threshold] = 1, 0
    VPK_mask = VPK_score.to(torch.int)
    return VPK_mask

def LPKscoring(patch_tokens,text_features,threshold=0.5):
    L = len(patch_tokens)
    B, HW, C = patch_tokens[0].shape
    H = int(np.sqrt(HW))

    vl_maps = []
    for layer in range(len(patch_tokens)):
        patch_tokens_norm = patch_tokens[layer].norm(dim=-1, keepdim=True)
        _patch_tokens = patch_tokens[layer] / patch_tokens_norm

        vl_map = (100.0 * _patch_tokens @ text_features) 
        vl_map = torch.softmax(vl_map, dim=2).clone()
        _vl_map = (1 - vl_map[:, :, 0] + vl_map[:, :, 1]) / 2
        vl_maps.append(_vl_map)

    vl_maps = torch.sum(torch.stack(vl_maps, dim=1), dim=1) 
    min_val = vl_maps.min(dim=1, keepdim=True)[0]
    max_val = vl_maps.max(dim=1, keepdim=True)[0]

    range_val = max_val - min_val
    range_val = range_val + (range_val == 0).float()

    vl_mask = (vl_maps - min_val) / range_val
    vl_mask[vl_mask >threshold], vl_mask[vl_mask <= threshold] = 1, 0
    vl_mask = vl_mask.reshape([B, H, H]).to(torch.int)

    return vl_mask


def genregion_feature(patch_tokens, VPK_mask, LPK_mask, device):
    L = len(patch_tokens)
    B, HW, C = patch_tokens[0].shape
    H = int(np.sqrt(HW))    
    
    f = []
    p = []
    fmean_p = []
    for j in range(0, L):
        fl = []
        pl = []
        fmean_pl = []
        for i in range(0, B):
            vvmask = VPK_mask[i, j, :].squeeze().reshape(H, H)
            vlmask = LPK_mask[i].squeeze().reshape(H, H)
            intersection = vvmask & vlmask

            proposal, f_mean = genproposal_feature(
                intersection.cpu().numpy(), patch_tokens[j][i, :, :], device)
            if len(f_mean) > 0:
                fmean_map = torch.einsum("ijk,im->ijkm", torch.stack(proposal, dim=0),
                                            torch.stack(f_mean, dim=0))
                fmean_map = torch.sum(fmean_map, dim=0)
                fmean_pl.append(fmean_map)
            else:
                fmean_pl.append(torch.zeros([H, H, C], dtype=patch_tokens[0].dtype).to(device))
            fl.append(f_mean)
            pl.append(proposal)
        f.append(fl)
        p.append(pl)
        fmean_p.append(fmean_pl)

    proposal_feature = []
    for i in range(0, L):
        proposal_feature.append(torch.stack(fmean_p[i], dim=0))
    return proposal_feature

def genproposal_feature(mask,feature,device,ref=None,area_threshold=2):
    mask=mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    HW,C=feature.shape
    H = int(np.sqrt(HW))
    feature=feature.reshape(H,H,C)
    proposal=[]
    fs_mean=[]

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= area_threshold:
            mask_label = labels == i
            region = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)
            region[mask_label] = 1
            region=torch.from_numpy(region).to(device)

            f=torch.einsum("ij,ijk->ijk",region,feature)
            f_mean=torch.sum(f.reshape(HW,C),dim=0)/stats[i, cv2.CC_STAT_AREA] 
            f_mean=f_mean/f_mean.norm(dim=-1, keepdim=True)

            proposal.append(region)
            fs_mean.append(f_mean)

            

    return proposal,fs_mean

def VL_seg(patch_tokens,text_features,region_feature,image_size=518,region_alpha=0.2):               
    L = len(patch_tokens)
    B, HW, C = patch_tokens[0].shape
    H = int(np.sqrt(HW))

    anomaly_maps = []
    for layer in range(L):

        region_tokens = region_feature[layer].reshape([B, HW, C])
        _patch_tokens = region_alpha * region_tokens + (1 - region_alpha) * patch_tokens[layer]
        _patch_tokens = F.normalize(_patch_tokens)
        anomaly_map = cal_ano_map(_patch_tokens, text_features, image_size)

        anomaly_maps.append(anomaly_map)
    return anomaly_maps


def cal_ano_map(patch_tokens,text_features,image_size,r_list=[1,3,5]):
    B, HW, C = patch_tokens.shape
    H = int(np.sqrt(HW))

    anomaly_map = (100.0 * patch_tokens @ text_features) 
    anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                size=image_size, mode='bilinear', align_corners=True)
    anomaly_map = torch.softmax(anomaly_map, dim=1).clone()
    return anomaly_map

def cal_ano_map_(patch_tokens,text_features,image_size,r_list=[1,3,5]):
    B, HW, C = patch_tokens.shape
    H = int(np.sqrt(HW))
    anomaly_maps_r = []
    for r in r_list:
        patch_tokens_local = local_aware(patch_tokens, r)
        anomaly_map = (100.0 * patch_tokens_local @ text_features) 

        anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                    size=image_size, mode='bilinear', align_corners=True)
        anomaly_map = torch.softmax(anomaly_map, dim=1).clone()
        anomaly_maps_r.append(anomaly_map)
    anomaly_map_mean = torch.mean(torch.stack(anomaly_maps_r), 0)
    return anomaly_map_mean

def local_aware(anomaly_map,patchsize):
    if patchsize==1:
        return anomaly_map
    B, L, C = anomaly_map.shape
    H = int(np.sqrt(L))
    patcharea = patchsize*patchsize
    stride = 1
    padding = int((patchsize - 1) / 2)  # 1
    unfolder = torch.nn.Unfold(
        kernel_size=patchsize, stride=stride, padding=padding, dilation=1
    )
    anomaly_map = anomaly_map.permute(0, 2, 1).view(B, C, H, H)
    unfolded_features = unfolder(anomaly_map.float())  
    unfolded_features = unfolded_features.permute(0, 2, 1)
    m = torch.nn.AvgPool1d(patcharea,patcharea)
    return m(unfolded_features)



