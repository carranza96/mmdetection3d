MODEL=centerpoint
RESULTS_DIR=results/nuscenes/${MODEL}/nd_pillar
CONFIG_FILE=configs/centerpoint_temporal/centerpoint_02pillar_nd_second_secfpn_4x8_cyclic_20e_nus.py

./tools/dist_train.sh ${CONFIG_FILE} 2 --work-dir=${RESULTS_DIR}



MODEL=centerpoint
RESULTS_DIR=results/nuscenes/${MODEL}/nd_pillar_convlstm
CONFIG_FILE=configs/centerpoint_temporal/centerpoint_02pillar_nd_temporal_second_secfpn_4x8_cyclic_20e_nus.py

./tools/dist_train.sh ${CONFIG_FILE} 2 --work-dir=${RESULTS_DIR}

