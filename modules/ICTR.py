import torch
import numpy as np
import torch.nn.functional as F
def ICTR(patch_tokens, mem_features, mem_cls_features, cls_name, args):
    lambd, img_size, device= args.lambd, args.image_size, args.device
    anomaly_maps_few_shot = []
    for idx, pi in enumerate(patch_tokens):
        if 'ViT' in args.model:
            p = pi[0, 1:, :]
            cls = pi[0, 0, :]
        else:
            p = pi[0].view(pi.shape[1], -1).permute(1, 0).contiguous()
        mem = mem_features[cls_name[0]][idx]
        mem /= mem.norm(dim=-1, keepdim=True)
        p /= p.norm(dim=-1, keepdim=True)
        cos = mem_features[cls_name[0]][idx] @ p.permute(1, 0)
        height = int(np.sqrt(cos.shape[1]))
        hw = height * height
        anomaly_map_few_shot_max = torch.min((1 - cos), dim=0).values.reshape(1, 1, height,
                                                                                          height).cpu().numpy()
        cls_tokens_reweight = []
        cls_tokens_reweight.append(cls.squeeze())
        for c in mem_cls_features[cls_name[0]][idx]:
            cls_tokens_reweight.append(c.squeeze())
        cls_tokens_reweight = torch.stack(cls_tokens_reweight)
        cls_tokens_reweight /= cls_tokens_reweight.norm(dim=-1, keepdim=True)
        similarity_matrix = (cls_tokens_reweight @ cls_tokens_reweight.T).to(torch.float32)
        weight_cls = torch.min(similarity_matrix[1:, 1:], dim=0).values - similarity_matrix[1:, 0]
        weight_cls = (1 - weight_cls) ** 2
        weights_reweight = torch.einsum('kl,k->kl',
                                        [cos.reshape(int(args.k_shot), -1), weight_cls]).reshape(
            int(args.k_shot) * hw, hw)

        weights_reweight_shrink = hard_shrink_relu(weights_reweight, lambd)
        result = torch.mm(weights_reweight_shrink.transpose(0, 1).to(device), mem.to(device))

        anomaly_map_few_shot = 1 - F.cosine_similarity(result, p, dim=-1).cpu().numpy()
        assert (True in np.isnan(anomaly_map_few_shot)) == False
        anomaly_map_few_shot = anomaly_map_few_shot.reshape(1, 1, height, height)
        anomaly_map_few_shot = anomaly_map_few_shot + anomaly_map_few_shot_max
        
        anomaly_map_few_shot = F.interpolate(torch.tensor(anomaly_map_few_shot),
                                                size=img_size, mode='bilinear', align_corners=True)
        anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
    
    return anomaly_maps_few_shot    


def hard_shrink_relu(input, lambd=0,epsilon=1e-7):
    output = (F.relu(input-lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output