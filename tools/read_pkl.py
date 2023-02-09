import pickle

objects = []
file_name = "/mnt/hd/mmdetection3d/data/waymo/kitti_format/waymo_infos_test.pkl"

with (open(file_name, "rb")) as f:
    while True:
        try:
            objects.append(pickle.load(f))
        except EOFError:
            break

print("objects")