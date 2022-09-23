## [TEMPORAL AXIAL ATTENTION FOR LIDAR-BASED 3D OBJECT DETECTION IN AUTONOMOUS DRIVING](http://www-video.eecs.berkeley.edu/papers/cgmanuel/ICIP_2022_Manuel__Copy_.pdf)

This project is forked from the MMDetection 3D repository https://github.com/open-mmlab/mmdetection3d.
For installation and dataset preparation please follow the instructions on [this page](README_mmdet3d.md)

We propose a modified CenterPoint architecture, with a novel temporal encoder  that uses temporal
axial attention to exploit the sequential nature of autonomous driving data for 3D object detection. The last ten LiDAR sweeps are split into three groups of frames, and the axial attention transformer block captures both spatial and temporal dependencies among the features extracted from each group.
We used the nuScenes dataset, available at www.nuScenes.org, and we plan to extend this work with other datasets such as Waymo.

![](resources/arch.png)

The main modifications in the MMDetection3D repository are the following:

- **Temporal encoders**: The designed [axial attention transformer](mmdet3d/models/temporal_encoders/axial_attention_transformer.py) and [ConvLSTM](mmdet3d/models/temporal_encoders/convlstm.py) are registered as modules, so they can be easily build from the config files.
- **Detector with temporal encoder**: We have modified the [centerpoint.py](mmdet3d/models/detectors/centerpoint.py) and [mvx_two_stage.py](mmdet3d/models/detectors/mvx_two_stage.py) to support the use of the temporal encoder, named pts_temporal_encoder.
- **Config files**: Prepared several configurations with different pillar sizes using the CP-TAA architecture in the [centerpoint_temporal](configs/centerpoint_temporal) folder.
- **Scripts**: Prepared a [script file](scripts/nuscenes_scripts.sh) with all necessary scripts to run train, test, evaluation, visualization, etc.


## Example usage

## Citation

If you find this project useful in your research, please consider cite:

```latex
@inproceedings{cp-taa,
    title={Temporal axial attention for {LiDAR}-based {3D} object detection in autonomous driving},
    author={Manuel Carranza-García, José C. Riquelme, Avideh Zakhor},
    booktitle = {29th IEEE International Conference on Image Processing (IEEE ICIP)},
    year={2022}
}
```



