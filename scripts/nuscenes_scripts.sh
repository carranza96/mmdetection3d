# Centerpoint default settings MMDetection
MODEL=centerpoint
RESULTS_DIR=results/nuscenes/${MODEL}/voxel
CONFIG_FILE=configs/centerpoint/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus.py
CHECKPOINT=checkpoints/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20201001_135205-5db91e00.pth

MODEL=centerpoint
RESULTS_DIR=results/nuscenes/${MODEL}/pillar
CONFIG_FILE=configs/centerpoint/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus.py
CHECKPOINT=checkpoints/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20201004_170716-a134a233.pth



################# 
## Pillar 0.2 ##
#################
# CP Non-deterministic HV
MODEL=centerpoint
RESULTS_DIR=results/nuscenes/${MODEL}02/30epochs/4samplesx2gpus/nd_pillar
CONFIG_FILE=configs/centerpoint_temporal/centerpoint_02pillar_nd_second_secfpn_4x8_cyclic_20e_nus.py

# Centerpoint Non-deterministic HV with Temporal Encoder (Transformer)
MODEL=centerpoint
RESULTS_DIR=results/nuscenes/${MODEL}02/30epochs/4samplesx2gpus/nd_pillar_transfv2_row
CONFIG_FILE=configs/centerpoint_temporal/centerpoint_02pillar_nd_transformer_second_secfpn_4x8_cyclic_20e_nus.py

# Centerpoint Non-deterministic HV with Temporal Encoder (ConvLSTM)
MODEL=centerpoint
RESULTS_DIR=results/nuscenes/${MODEL}02/30epochs/4samplesx2gpus/nd_pillar_convlstm
CONFIG_FILE=configs/centerpoint_temporal/centerpoint_02pillar_nd_convlstm_second_secfpn_4x8_cyclic_20e_nus.py


################# 
## Pillar 0.15 ##
#################
# CP Non-deterministic HV
MODEL=centerpoint
RESULTS_DIR=results/nuscenes/${MODEL}015/30epochs/nd_pillar
CONFIG_FILE=configs/centerpoint_temporal/centerpoint_015pillar_nd_second_secfpn_4x8_cyclic_20e_nus.py

# Centerpoint Non-deterministic HV with Temporal Encoder (Transformer)
MODEL=centerpoint
RESULTS_DIR=results/nuscenes/${MODEL}015/30epochs/nd_pillar_transfv2_row_fix
CONFIG_FILE=configs/centerpoint_temporal/centerpoint_015pillar_nd_transformer_second_secfpn_4x8_cyclic_20e_nus.py

# Centerpoint Non-deterministic HV with Temporal Encoder (ConvLSTM)
MODEL=centerpoint
RESULTS_DIR=results/nuscenes/${MODEL}015/30epochs/nd_pillar_convlstm
CONFIG_FILE=configs/centerpoint_temporal/centerpoint_015pillar_nd_convlstm_second_secfpn_4x8_cyclic_20e_nus.py



################# 
## Pillar 0.3 ##
#################
# CP Non-deterministic HV
MODEL=centerpoint
RESULTS_DIR=results/nuscenes/centerpoint03/30epochs/4samplesx2gpus/nd_pillar
CONFIG_FILE=configs/centerpoint_temporal/centerpoint_03pillar_nd_second_secfpn_4x8_cyclic_20e_nus.py

# Centerpoint Non-deterministic HV with Temporal Encoder (Transformer)
MODEL=centerpoint
RESULTS_DIR=results/nuscenes/${MODEL}03/30epochs/4samplesx2gpus/nd_pillar_transfv2_row
CONFIG_FILE=configs/centerpoint_temporal/centerpoint_03pillar_nd_transformer_second_secfpn_4x8_cyclic_20e_nus.py

# Centerpoint Non-deterministic HV with Temporal Encoder (ConvLSTM)
MODEL=centerpoint
RESULTS_DIR=results/nuscenes/${MODEL}03/30epochs/nd_pillar_convlstm
CONFIG_FILE=configs/centerpoint_temporal/centerpoint_03pillar_nd_convlstm_second_secfpn_4x8_cyclic_20e_nus.py




# Voxel 0.1
MODEL=centerpoint
RESULTS_DIR=results/nuscenes/${MODEL}01/30epochs/nd_voxel_transfv2_row
CONFIG_FILE=configs/centerpoint/centerpoint_01voxel_nd_convlstm_second_secfpn_4x8_cyclic_20e_nus.py
CHECKPOINT=${RESULTS_DIR}/latest.pth



# Test
CHECKPOINT=${RESULTS_DIR}/latest.pth
./tools/dist_test.sh ${CONFIG_FILE} \
${CHECKPOINT} 2 \
--out ${RESULTS_DIR}/result.pkl \
--eval mAP 


# Train
./tools/dist_train.sh ${CONFIG_FILE} 2 --work-dir=${RESULTS_DIR}


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
CHECKPOINT=${RESULTS_DIR}/latest.pth
python tools/analysis_tools/benchmark.py ${CONFIG_FILE} ${CHECKPOINT}
python -m torch.distributed.launch --nproc_per_node=1 --master_port=${PORT:-29510} \
tools/analysis_tools/benchmark_dist.py ${CONFIG_FILE} ${CHECKPOINT}


# FLOPS
python tools/analysis_tools/get_flops.py ${CONFIG_FILE} 


# Pillar 0.2
# 20 epochs loss
python tools/analysis_tools/analyze_logs.py plot_curve \
results/nuscenes/centerpoint02/20epochs/4samplesx2gpus/nd_pillar/20211031_232222.log.json \
results/nuscenes/centerpoint02/20epochs/4samplesx2gpus/nd_pillar_convlstm/20211101_101831.log.json \
results/nuscenes/centerpoint02/20epochs/4samplesx2gpus/nd_pillar_transfv2_row/20211208_002939.log.json \
--keys loss --legend CenterPoint ConvLSTM Transformer  --out loss20.pdf 

# 30 epochs loss
python tools/analysis_tools/analyze_logs.py plot_curve \
results/nuscenes/centerpoint02/30epochs/4samplesx2gpus/nd_pillar/20211222_113253.log.json \
results/nuscenes/centerpoint02/30epochs/4samplesx2gpus/nd_pillar_convlstm/20211211_022807.log.json \
results/nuscenes/centerpoint02/30epochs/4samplesx2gpus/nd_pillar_transfv2_row/20211213_113612.log.json \
--keys loss --legend CP CP-ConvLSTM CP-TAA  --out loss30.pdf 
#--mode eval --interval 1

# 30 epochs mAP
python tools/analysis_tools/analyze_logs.py plot_curve \
results/nuscenes/centerpoint02/30epochs/4samplesx2gpus/nd_pillar/20211222_113253.log.json \
results/nuscenes/centerpoint02/30epochs/4samplesx2gpus/nd_pillar_convlstm/20211211_022807.log.json \
results/nuscenes/centerpoint02/30epochs/4samplesx2gpus/nd_pillar_transfv2_row/20211213_113612.log.json \
--keys pts_bbox_NuScenes/mAP --legend CP CP-ConvLSTM CP-TAA  --mode eval --out map30.pdf 
#--mode eval --interval 1



# Pillar 0.15
# 30 epochs loss
python tools/analysis_tools/analyze_logs.py plot_curve \
results/nuscenes/centerpoint015/30epochs/nd_pillar/20220113_150508.log.json \
results/nuscenes/centerpoint015/30epochs/nd_pillar_convlstm/20220117_090334.log.json \
results/nuscenes/centerpoint015/30epochs/nd_pillar_transfv2_row/20220114_154715.log.json \
--keys loss --legend CP CP-ConvLSTM CP-TAA  --out loss_015_30.pdf 
#--mode eval --interval 1

# 30 epochs mAP
python tools/analysis_tools/analyze_logs.py plot_curve \
results/nuscenes/centerpoint015/30epochs/nd_pillar/20220113_150508.log.json \
results/nuscenes/centerpoint015/30epochs/nd_pillar_convlstm/20220117_090334.log.json \
results/nuscenes/centerpoint015/30epochs/nd_pillar_transfv2_row/20220114_154715.log.json \
--keys pts_bbox_NuScenes/mAP --legend CP CP-ConvLSTM CP-TAA  --mode eval --out map_015_30.pdf 
#--mode eval --interval 1