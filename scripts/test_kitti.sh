MODEL=pointpillars
RESULTS_DIR=results/kitti/${MODEL}

# 3 classes
CONFIG_FILE=configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py
CHECKPOINT=checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20200620_230421-aa0f3adb.pth
# Car only
CONFIG_FILE=configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py
CHECKPOINT=checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20200620_230614-77663cd6.pth

################################################################

MODEL=second
RESULTS_DIR=results/kitti/${MODEL}
# 3 classes
CONFIG_FILE=configs/second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py
CHECKPOINT=checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-3class_20200620_230238-9208083a.pth
# Car only
CONFIG_FILE=configs/second/hv_second_secfpn_6x8_80e_kitti-3d-car.py
CHECKPOINT=checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-car_20200620_230238-393f000c.pth

################################################################

MODEL=parta2
RESULTS_DIR=results/kitti/${MODEL}
# 3 classes
CONFIG_FILE=configs/parta2/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class.py
CHECKPOINT=checkpoints/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class_20200620_230724-a2672098.pth
# Car only
CONFIG_FILE=configs/parta2/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-car.py
CHECKPOINT=checkpoints/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-car_20200620_230755-f2a38b9a.pth


################################################################

MODEL=3dssd
RESULTS_DIR=results/kitti/${MODEL}
# Car only
CONFIG_FILE=configs/3dssd/3dssd_4x4_kitti-3d-car.py 
CHECKPOINT=checkpoints/3dssd_kitti-3d-car_20210602_124438-b4276f56.pth


################################################################

MODEL=mvxnet
RESULTS_DIR=results/kitti/${MODEL}
# 3 classes
CONFIG_FILE=configs/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py 
CHECKPOINT=checkpoints/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class_20200621_003904-10140f2d.pth


##########################################

# mAP evaluation
./tools/dist_test.sh ${CONFIG_FILE} \
${CHECKPOINT} 1 \
--eval mAP \
--out=${RESULTS_DIR}/result.pkl


# Format results
./tools/dist_test.sh ${CONFIG_FILE} \
${CHECKPOINT} 1 \
--format-only \
--out=${RESULTS_DIR}/result.pkl \
--eval-options "pklfile_prefix="${RESULTS_DIR}"/results_submission" "submission_prefix="${RESULTS_DIR}"/submission"


# Online visualization while evaluating
./tools/dist_test.sh ${CONFIG_FILE} \
${CHECKPOINT} 1 \
--eval mAP \
--eval-options "show=True" "out_dir="${RESULTS_DIR}"/show_results"


# Offline visualization
python ./tools/misc/visualize_results.py ${CONFIG_FILE} \
--result=${RESULTS_DIR}/result.pkl \
--show-dir=${RESULTS_DIR}"/show_results"


# Browse KITTI dataset multi-modality
python tools/misc/browse_dataset.py  configs/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py \
--task=multi_modality-det \
--output-dir=results/browse_dataset \
--online 


# FPS Benchmark
python tools/analysis_tools/benchmark.py ${CONFIG_FILE} ${CHECKPOINT}


# Plot with mayavi
python kitti_object_vis/kitti_object.py \
--dir=data/kitti \
--show_lidar_with_depth \
--img_fov \
--const_box \
--vis \
--pred \
--preddir=/home/manuelc/mmdetection3d/results/kitti/pointpillars/kitti_val_submission \
--ind=1 \

--show_lidar_topview_with_boxes \
--show_image_with_boxes \
--show_lidar_on_image \
