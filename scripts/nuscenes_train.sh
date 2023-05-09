MODEL=centerpoint
RESULTS_DIR=results/nuscenes/${MODEL}02/20epochs/3xb4/dv_pillar_10s
CONFIG_FILE=configs/centerpoint_nuscenes/dv/centerpoint_pillar02_dv_second_secfpn_8xb4_cyclic-20e_nus-3d.py

# MODEL=centerpoint
# RESULTS_DIR=results/nuscenes/${MODEL}02/20epochs/3xb4/dv_pillar_10s_convlstm
# CONFIG_FILE=configs/centerpoint_nuscenes/dv/centerpoint_pillar02_dv_convlstm_second_secfpn_8xb4_cyclic-20e_nus-3d.py

# MODEL=centerpoint
# RESULTS_DIR=results/nuscenes/${MODEL}02/20epochs/3xb4/dv_pillar_10s_transformer
# CONFIG_FILE=configs/centerpoint_nuscenes/dv/centerpoint_pillar02_dv_transformer_second_secfpn_8xb4_cyclic-20e_nus-3d.py

# MODEL=centerpoint
# RESULTS_DIR=results/nuscenes/${MODEL}01/20epochs/3xb4/dv_voxel_10s
# CONFIG_FILE=configs/centerpoint_nuscenes/dv/centerpoint_voxel01_dv_second_secfpn_8xb4-cyclic-20e_nus-3d.py

# MODEL=centerpoint
# RESULTS_DIR=results/nuscenes/${MODEL}01/20epochs/3xb4/dv_voxel_10s_convlstm
# CONFIG_FILE=configs/centerpoint_nuscenes/dv/centerpoint_voxel01_dv_convlstm_second_secfpn_8xb4-cyclic-20e_nus-3d.py

./tools/dist_train.sh ${CONFIG_FILE} 3 --work-dir=${RESULTS_DIR}


CHECKPOINT=${RESULTS_DIR}/epoch_20.pth
SAVE_FILE=result

./tools/dist_test.sh ${CONFIG_FILE} \
${CHECKPOINT} 3 \
--work-dir ${RESULTS_DIR} \
--cfg-options "test_evaluator.jsonfile_prefix=${RESULTS_DIR}/${SAVE_FILE}"