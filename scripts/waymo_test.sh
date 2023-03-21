CONFIG_FILE=configs/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class.py
CHECKPOINT=results/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class_20200831_204144-d1a706b1.pth
RESULTS_DIR=results/pointpillarsD5NoRepeat


./tools/dist_train.sh ${CONFIG_FILE} 3 --work-dir=${RESULTS_DIR} 
## Debe salir 58.4 mAPL2

CHECKPOINT=${RESULTS_DIR}/latest.pth
./tools/dist_test.sh ${CONFIG_FILE} \
${CHECKPOINT} 3 \
--out ${RESULTS_DIR}/result.pkl \
--eval waymo 
--eval-options 'pklfile_prefix=${RESULTS_DIR}/result