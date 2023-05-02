MODEL=centerpoint
RESULTS_DIR=results/waymo/${MODEL}01/20epochs/3xb2/voxelD5
CONFIG_FILE=configs/centerpoint_waymo/centerpoint_voxel01_second_secfpn_waymo.py


./tools/dist_train.sh ${CONFIG_FILE} 3 --work-dir=${RESULTS_DIR}


CHECKPOINT=${RESULTS_DIR}/epoch_20.pth
SAVE_FILE=result

./tools/dist_test.sh ${CONFIG_FILE} \
${CHECKPOINT} 3 \
--work-dir ${RESULTS_DIR} \
--cfg-options "test_evaluator.pklfile_prefix=${RESULTS_DIR}/${SAVE_FILE}"

mmdet3d/evaluation/functional/waymo_utils/compute_detection_metrics_main ${RESULTS_DIR}/${SAVE_FILE}.bin data/waymo/waymo_format/gt.bin
