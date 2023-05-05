MODEL=centerpoint
RESULTS_DIR=results/nuscenes/${MODEL}02/20epochs/4xb4/nd_pillar_10s
CONFIG_FILE=configs/centerpoint_nuscenes/centerpoint_pillar02_nd_second_secfpn_8xb4_cyclic-20e_nus-3d.py

./tools/dist_train.sh ${CONFIG_FILE} 3 --work-dir=${RESULTS_DIR}


CHECKPOINT=${RESULTS_DIR}/epoch_20.pth
SAVE_FILE=result

./tools/dist_test.sh ${CONFIG_FILE} \
${CHECKPOINT} 3 \
--work-dir ${RESULTS_DIR} \
--cfg-options "test_evaluator.jsonfile_prefix=${RESULTS_DIR}/${SAVE_FILE}"