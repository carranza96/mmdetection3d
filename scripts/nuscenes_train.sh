MODEL=centerpoint
RESULTS_DIR=results/nuscenes/${MODEL}02/40epochs_cbgs/4xb4/nd_pillar
CONFIG_FILE=configs/centerpoint_temporal/centerpoint_02pillar_nd_second_secfpn_4x8_cyclic_20e_nus.py

./tools/dist_train.sh ${CONFIG_FILE} 4 --work-dir=${RESULTS_DIR}


MODEL=centerpoint
RESULTS_DIR=results/nuscenes/${MODEL}02/40epochs_cbgs/4xb4/nd_pillar_transf
CONFIG_FILE=configs/centerpoint_temporal/centerpoint_02pillar_nd_transformer_second_secfpn_4x8_cyclic_20e_nus.py

./tools/dist_train.sh ${CONFIG_FILE} 4 --work-dir=${RESULTS_DIR}


MODEL=centerpoint
RESULTS_DIR=results/nuscenes/${MODEL}02/40epochs_cbgs/4xb4/nd_pillar_convlstm
CONFIG_FILE=configs/centerpoint_temporal/centerpoint_02pillar_nd_convlstm_second_secfpn_4x8_cyclic_20e_nus.py

./tools/dist_train.sh ${CONFIG_FILE} 4 --work-dir=${RESULTS_DIR}