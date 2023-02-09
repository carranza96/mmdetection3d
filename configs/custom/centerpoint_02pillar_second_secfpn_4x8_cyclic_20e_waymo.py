_base_ = [
    '../_base_/datasets/waymoD5-3d-3class.py',
    '../_base_/models/custom/centerpoint_02pillar_second_secfpn_waymo_sol.py',
    '../_base_/schedules/schedule_2x.py', 
    '../_base_/default_runtime.py'
]

data = dict(train=dict(dataset=dict(load_interval=1)), samples_per_gpu=1)
evaluation = dict(interval=24)

point_cloud_range = [-74.88, -74.88, -2, 74.88, 74.88, 4.0]
model = dict(
    pts_voxel_layer=dict(point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(point_cloud_range=point_cloud_range),
    pts_bbox_head=dict(bbox_coder=dict(pc_range=point_cloud_range[:2])),
    # model training and testing settings
    train_cfg=dict(pts=dict(point_cloud_range=point_cloud_range)),
    test_cfg=dict(pts=dict(pc_range=point_cloud_range[:2])))