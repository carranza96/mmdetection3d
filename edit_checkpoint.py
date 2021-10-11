import torch
from copy import deepcopy

model = torch.load("checkpoints/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20200620_230725-0817d270.pth")
sd = model['state_dict']
new_sd = dict()

for k,v in sd.items():
    # if 'pts_voxel_encoder' in k:
    #     k = k.replace('pts_voxel_encoder', 'voxel_encoder')
    # elif 'pts_backbone' in k:
    #     k = k.replace('pts_backbone', 'backbone')
    # elif 'pts_neck' in k:
    #     k = k.replace('pts_neck', 'neck')
    # elif 'pts_bbox_head' in k:
    #     k = k.replace('pts_bbox_head', 'bbox_head')

    if 'pts' in k:
        k = k.replace('pts_', '')


    new_sd[k] = v

new_model = deepcopy(model)
new_model['state_dict'] = new_sd
torch.save(new_model, "checkpoints/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_mod_dv.pth")