_base_ = [
    '../../_base_/schedules/cyclic-20e.py', '../../_base_/default_runtime.py'
]
# TODO: Check optimizer params (compare with CenterFormer)

# model settings
# Voxel size for voxel encoder
# Usually voxel size is changed consistently with the point cloud range
# If point cloud range is modified, do remember to change all related
# keys in the config.
voxel_size = [0.1, 0.1, 0.15]
point_cloud_range = [-75.2, -75.2, -2, 75.2, 75.2, 4] #Official CenterPointRepo
class_names = ['Car', 'Pedestrian', 'Cyclist']


model = dict(
    type='CenterPoint',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        # TODO: Dynamic/non-deterministic voxelization (in Centerformer is Dynamic)
        voxel_layer=dict(
            max_num_points=5, # As in Official CenterPointRepo (in Waymo PointPillars is 20, in NUS CP-Voxel is 10)
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_voxels=(150000, 150000))), # As in Official CenterPointRepo (in NUS CP-Voxel is (32000,32000))
    pts_voxel_encoder=dict(
        type='HardSimpleVFE', # in CenterFormer is DynamicSimpleVFE
        num_features=5
        ),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        sparse_shape=[41, 1504, 1504],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='naiveSyncBN1d', eps=0.001, momentum=0.01), 
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((1, 1, 1), (1, 1, 1), (1, 1, [0, 1, 1]), (1, 1)), # As in Official CenterPointRepo
        block_type='basicblock'),
    
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        layer_nums=[5, 5], # Official CenterPointRepo (NUS CP-Voxel is same, in Waymo PointPillars is [3,5,5], in CenterFormer [5,5,1?])
        layer_strides=[1, 2], # Official CenterPointRepo (NUS CP-Voxel is same, in Waymo PointPillars is [1,2,2], in CenterFormer is also [1,2])
        out_channels=[128, 256], # Official CenterPointRepo (NUS CP-Voxel is same, in Waymo PointPillars is [64, 128, 256], in CenterFormer is [256,256])
        norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01), 
        conv_cfg=dict(type='Conv2d', bias=False)
    ),
    # Official CenterPointRepo (NUS CP-Voxel is same)
    # TODO: Change BEV map size
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256], # (in CenterFormer is [256,256])
        out_channels=[256, 256], # (in CenterFormer is [128,128])
        upsample_strides=[2, 4], # (in CenterFormer is [2,4])
        norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01), 
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=sum([256, 256]),  # (in CenterFormer is 256)
        tasks = [dict(num_class=3, class_names=['Car', 'Pedestrian', 'Cyclist'])], # https://github.com/tianweiy/CenterPoint/issues/130 (works similarly as dividing into three tasks)
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)), # Centerformer includes  'iou': (1, 2)
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-80, -80, -10.0, 80, 80, 10.0],
            max_num=500, # TODO: in Centerformer paper is 1000 in testing, but here in CenterFormer MMDetection is 500
            score_threshold=0.1,
            pc_range=[-75.2, -75.2],
            out_size_factor=4, # TODO: (in NUS CP-Voxel is 8, in Centerformer is 4)
            voxel_size=voxel_size[:2],
            code_size=7),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'), # Centerformer uses FastFocalLoss
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=2),
        # Centerformer includes iou_loss and corner_loss
        norm_bbox=True),
    
    # In Centerformer the heatmap is obtained direcly after BEV feature extraction, while in CenterPoint it is obtained after shared_conv
    # In Centerformer the bounding box regression is different: a regression head to predict the bounding box at each enhanced center feature (after being processed by transformer decoder)
    
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[1504, 1504, 40],
            voxel_size=voxel_size,
            out_size_factor=4, # TODO: (in NUS CP-Voxel is 8, in Centerformer is 4)
            dense_reg=1,
            gaussian_overlap=0.1,
            point_cloud_range=point_cloud_range,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )),
    
    
    # TODO: Try different nms_thr. In CenterFormer is different , uses multi_class_nms with nms_thr [0.8, 0.55, 0.55]
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=[-80, -80, -10.0, 80, 80, 10.0],
            max_per_img=500, # It is not used
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            pc_range=[-75.2, -75.2],
            out_size_factor=4, # TODO: (in NUS CP-Voxel is 8, in Centerformer is 4)
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=4096,
            post_max_size=500,
            nms_thr=0.7)) 
    )


#  '../_base_/datasets/waymoD5-3d-3class.py'
dataset_type = 'WaymoDataset'
data_root = 'data/waymo/kitti_format/'
# data_root = '/mnt/hd/mmdetection3d/data/waymo/kitti_format/'
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
# TODO: Centerformes disables ObjectSample
# custom_hooks = [dict(type='DisableObjectSampleHook', disable_after_epoch=15)] # NEW: Fade augmentation strategy. The model can adjust to the real data distribution at the end of the training

# fp16 = dict(loss_scale=32.)


# Notes CenterFormer
# Since the BEV map resolution needs to be relatively large to
# maintain the fine-grained features for small objects, it is impractical to use all
# BEV features as the attending keypoints. Alternatively, we confine the attending
# keypoints to a small 3×3 window near the center location at each scale, as illustrated in Figure 3. The complexity of this cross-attention is O(9SN), which is
# more efficient than the normal implementation. Because of multi-scale features,
# we are able to capture a wide range of features around proposed centers


# The proposed centers are then
# used as the query feature embedding in a transformer decoder to aggregate
# features from other centers and from multi-scale feature maps. Finally, we use
# a regression head to predict the bounding box at each enhanced center feature.


# In the cross-attention layer, we use the location of the center proposal to find
# the corresponding features in the aligned previous frames. The extracted features
# will be added to the attending keys. Since our normal cross-attention design uses
# features in a small window close to the center location, it has limited learnability
# if the object was out of the window area due to fast movement. Meanwhile, our
# deformable cross-attention is able to model any level of movement, and is more
# suitable for the long time-range case. Because our multi-frame model only needs
# the final BEV feature of the previous frame, it is easy to be deployed to the
# online prediction by saving the BEV feature in a memory bank


# Comparison with two-stage LiDAR detection
# In contrast, our method works
# between one-stage and two-stage. We use a center proposal network to generate
# initial center queries. The self-attention layer allows the network to directly learn
# object-level contextual information. The cross-attention layer can also capture
# long-range information in the multi-scale BEV feature. The classification and
# regression are done only once in our method.


# we use the same training strategy in the center-based object detection network, i.e.
# only training the network when the proposed center is at the same position as
# the ground truth center. To utilize all annotation information in training, we
# manually select the center positions of all ground truth bounding boxes as the
# initial center proposals in training. And the position with the highest heatmap
# response other than those positions are selected as the remaining proposals. This
# allows the network to have a meaningful training objective from the beginning
# of the training, and thus converges faster