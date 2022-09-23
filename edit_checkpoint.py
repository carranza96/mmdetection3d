import torch
from copy import deepcopy

# model = torch.load("results/nuscenes/centerpoint02/30epochs/4samplesx2gpus/nd_pillar_transfv2_row/latest_old.pth")
# model = torch.load("results/nuscenes/centerpoint02/30epochs/4samplesx2gpus/nd_pillar_convlstm/latest_old.pth")
model = torch.load("results/nuscenes/centerpoint015/30epochs/nd_pillar_transfv2_row_fix/latest_old.pth")

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

    if 'attn' in k:
        k = k.replace('attn', 'pts_temporal_encoder')

    # if 'convlstm' in k:
    #     k = k.replace('convlstm', 'pts_temporal_encoder')


    new_sd[k] = v

new_model = deepcopy(model)
new_model['state_dict'] = new_sd
# torch.save(new_model, "results/nuscenes/centerpoint02/30epochs/4samplesx2gpus/nd_pillar_transfv2_row/latest.pth")
# torch.save(new_model, "results/nuscenes/centerpoint02/30epochs/4samplesx2gpus/nd_pillar_convlstm/latest.pth")
torch.save(new_model, "results/nuscenes/centerpoint015/30epochs/nd_pillar_transfv2_row_fix/latest.pth")