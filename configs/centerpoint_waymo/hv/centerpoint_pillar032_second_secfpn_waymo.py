_base_ = [
    '../_base_/schedules/cyclic-20e.py', '../_base_/default_runtime.py'
]

voxel_size = [0.32, 0.32, 6]
point_cloud_range = [-74.88, -74.88, -2, 74.88, 74.88, 4.0] #Official CenterPointRepo
class_names = ['Car', 'Pedestrian', 'Cyclist']

model = dict(
    type='CenterPoint',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=20, 
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_voxels=(32000, 60000))), 
    pts_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=5,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        legacy=False),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=(468, 468)),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[1, 2, 2],
        out_channels=[64, 128, 256],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[1, 2, 4],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=sum([128, 128, 128]),
        tasks = [dict(num_class=3, class_names=['Car', 'Pedestrian', 'Cyclist'])],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-80, -80, -10.0, 80, 80, 10.0],
            max_num=500,
            score_threshold=0.1,
            pc_range=[-74.88, -74.88],
            out_size_factor=1,
            voxel_size=voxel_size[:2],
            code_size=7),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'), 
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=2),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[468, 468, 1],
            voxel_size=voxel_size,
            out_size_factor=1,
            dense_reg=1,
            gaussian_overlap=0.1,
            point_cloud_range=point_cloud_range,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=[-80, -80, -10.0, 80, 80, 10.0],
            max_per_img=500, # It is not used
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            pc_range=[-74.88, -74.88],
            out_size_factor=1, 
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=4096,
            post_max_size=500,
            nms_thr=0.7)) 
    )


#  '../_base_/datasets/waymoD5-3d-3class.py'
dataset_type = 'WaymoDataset'
# data_root = 'data/waymo/kitti_format/'
data_root = '/mnt/hd/mmdetection3d/data/waymo/kitti_format/'
metainfo = dict(classes=class_names)
input_modality = dict(use_lidar=True, use_camera=False)
file_client_args = dict(backend='disk')


db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'waymo_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5)),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4],
        norm_intensity=True)
)


train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=6, use_dim=5, norm_intensity=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    # TODO: Neither CenterPoint Official Repo nor Centerformer seem to use RandomFlip3D in training
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]), 
    # TODO: Centerformer includes translation_std=[0.5, 0.5, 0]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=5,
        norm_intensity=True,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]


# TODO: Adjust batch size
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='waymo_infos_train.pkl',
        data_prefix=dict(
            pts='training/velodyne', sweeps='training/velodyne'),
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        metainfo=metainfo,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR',
        # load one frame every five frames
        load_interval=5,
        file_client_args=file_client_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne', sweeps='training/velodyne'),
        ann_file='waymo_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        file_client_args=file_client_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='WaymoMetric',
    ann_file= data_root + 'waymo_infos_val.pkl',
    waymo_bin_file='./data/waymo/waymo_format/gt.bin',
    data_root='./data/waymo/waymo_format',
    file_client_args=file_client_args,
    convert_kitti_format=False,
    idx2metainfo='./data/waymo/waymo_format/idx2metainfo.pkl')
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')


# '../_base_/schedules/cyclic-20e.py'
train_cfg = dict(val_interval=20)


default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1))