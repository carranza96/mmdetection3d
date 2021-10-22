# Centerpoint
MODEL=centerpoint
RESULTS_DIR=results/nuscenes/${MODEL}/voxel
CONFIG_FILE=configs/centerpoint/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus.py
CHECKPOINT=checkpoints/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20201001_135205-5db91e00.pth

MODEL=centerpoint
RESULTS_DIR=results/nuscenes/${MODEL}/pillar
CONFIG_FILE=configs/centerpoint/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus.py
CHECKPOINT=checkpoints/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20201004_170716-a134a233.pth

# Centerpoint Non-deterministic HV
MODEL=centerpoint
RESULTS_DIR=results/nuscenes/${MODEL}/nd_pillar_no_cbgs
CONFIG_FILE=configs/centerpoint/centerpoint_02pillar_nd_second_secfpn_4x8_cyclic_20e_nus.py
CHECKPOINT=results/nuscenes/centerpoint/nd_pillar_no_cbgs/epoch_5.pth

# Centerpoint Non-deterministic HV with ConvLSTM module
MODEL=centerpoint
RESULTS_DIR=results/nuscenes/${MODEL}/nd_pillar_convlstm_no_cbgs
CONFIG_FILE=configs/centerpoint/centerpoint_02pillar_nd_convlstm_second_secfpn_4x8_cyclic_20e_nus.py
CHECKPOINT=results/nuscenes/centerpoint/nd_pillar_convlstm_no_cbgs/latest.pth


# Pointpillars
MODEL=hv_pointpillars
RESULTS_DIR=results/nuscenes/${MODEL}

CONFIG_FILE=configs/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d.py
# CONFIG_FILE=configs/fp16/hv_pointpillars_secfpn_sbn-all_fp16_2x8_2x_nus-3d.py
CHECKPOINT=checkpoints/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20200620_230725-0817d270.pth

CONFIG_FILE=configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py
CHECKPOINT=checkpoints/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20200620_230405-2fa62f3d.pth


# Dynamic voxelization PointPillars
MODEL=dv_pointpillars
RESULTS_DIR=results/nuscenes/${MODEL}
CONFIG_FILE=configs/dynamic_voxelization/dv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d.py
# CONFIG_FILE=configs/dynamic_voxelization/dv_pointpillars_secfpn_sbn-all_fp16_4x8_2x_nus-3d.py
# CHECKPOINT=checkpoints/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_mod_dv.pth
CHECKPOINT=results/nuscenes/${MODEL}/latest.pth

# Dynamic voxelization Centernet (Not working yet)
MODEL=dv_centerpoint
RESULTS_DIR=results/nuscenes/${MODEL}/pillar
CONFIG_FILE=configs/dynamic_voxelization/dv_centerpoint_02pillar_second_secfpn_nus.py
# CONFIG_FILE=configs/dynamic_voxelization/dv_pointpillars_secfpn_sbn-all_fp16_4x8_2x_nus-3d.py
# CHECKPOINT=checkpoints/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_mod_dv.pth
CHECKPOINT=results/nuscenes/${MODEL}/latest.pth



# Test
./tools/dist_test.sh ${CONFIG_FILE} \
${CHECKPOINT} 1 \
--out ${RESULTS_DIR}/result_epoch5.pkl \
--eval mAP 


# Train
./tools/dist_train.sh ${CONFIG_FILE} 1 --work-dir=${RESULTS_DIR}


# Browse NuScenes dataset 3D LiDAR detection
python tools/misc/browse_dataset.py  configs/_base_/datasets/nus-3d.py \
--task=det \
--output-dir=results/browse_dataset/nuscenes \
--online 

# Browse NuScenes monocular 3D
python tools/misc/browse_dataset.py configs/_base_/datasets/nus-mono3d.py \
--task mono-det \
--output-dir=results/browse_dataset/nuscenes \
--online


# Browse NuScenes dataset multi-modality (not working)
python tools/misc/browse_dataset.py  configs/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d.py \
--task=multi_modality-det \
--output-dir=results/browse_dataset/nuscenes \
--online 



# Offline visualization
python ./tools/misc/visualize_results_nuscenes.py configs/_base_/datasets/nus-3d.py \
--result=${RESULTS_DIR}/result.pkl \
--show-dir=${RESULTS_DIR}"/show_results" \
--score-th=0.3


# Benchmark FPS
python tools/analysis_tools/benchmark.py ${CONFIG_FILE} ${CHECKPOINT}
python -m torch.distributed.launch --nproc_per_node=1 --master_port=${PORT:-29500} \
tools/analysis_tools/benchmark_dist.py ${CONFIG_FILE} ${CHECKPOINT}


# FLOPS
python tools/analysis_tools/get_flops.py ${CONFIG_FILE} 