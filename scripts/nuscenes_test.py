from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-trainval', dataroot='/data/nuscenes', verbose=True)

my_sample0 = nusc.sample[0]
my_sample1 = nusc.sample[1]
my_sample10 = nusc.sample[10]
my_sample15 = nusc.sample[15]

nusc.list_scenes()

# nusc.render_instance(nusc.instance[0]['token'])
# nusc.render_pointcloud_in_image(my_sample10['token'], pointsensor_channel='LIDAR_TOP')
# nusc.render_sample_data(my_sample10['data']['CAM_FRONT'], with_anns=False)
nusc.render_sample_data(my_sample15['data']['LIDAR_TOP'], nsweeps=10, underlay_map=True)
print()

my_scene_token = nusc.field2token('scene', 'name', 'scene-0061')[0]

tok = my_sample15['data']['LIDAR_TOP']
sample_data = nusc.get('sample_data',tok)
ref_pose_rec = nusc.get('ego_pose',sample_data['ego_pose_token'])
ref_cs_rec = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
ref_time = 1e-6 * sample_data['timestamp']


# Video
my_scene_token = nusc.field2token('scene', 'name', 'scene-0008')[0]
nusc.render_scene_channel(my_scene_token, 'CAM_FRONT')
nusc.render_scene(my_scene_token)

