import pickle
import os
from waymo_open_dataset.protos import metrics_pb2

n = 1 # number of images to get from each scene

full_dataset_path = "/mnt/hd/mmdetection3d/data/waymo_full/"
root_path = "/mnt/hd/"
out_dir = "/mnt/hd/mmdetection3d/data/waymo/"

# Creation of reduced ImageSets .txt files

h = [f'{i:03d}' for i in range(n)]

set_names = ["trainval.txt", "train.txt", "val.txt", "test.txt"]

for set_name in set_names:
    if not os.path.exists(out_dir + "kitti_format/ImageSets"):
        os.makedirs(out_dir + "kitti_format/ImageSets")
    with open(full_dataset_path + "kitti_format/ImageSets/" + set_name, "r") as f:
        lines = f.readlines()
    with open(out_dir + "kitti_format/ImageSets/" + set_name, "w") as f:
        for line in lines:
            if line.strip("\n").endswith(tuple(h)):
                f.write(line)

# Copy waymo_format folder as symbolic links
import shutil

def copy(src, dst):
    if not os.path.exists(dst):
        if os.path.islink(src):
            linkto = os.readlink(src)
            os.symlink(linkto, dst)
        else:
            shutil.copy(src,dst)

if not os.path.exists(out_dir + "waymo_format"):
    os.makedirs(out_dir + "waymo_format")

copy(full_dataset_path + "waymo_format/training", out_dir + "waymo_format/training")
copy(full_dataset_path + "waymo_format/validation", out_dir + "waymo_format/validation")
copy(full_dataset_path + "waymo_format/gt.bin", out_dir + "waymo_format/gt.bin")
# os.rmdir(out_dir + "kitti_format/training")
if not os.path.exists(out_dir + "kitti_format/training"):
    os.symlink(full_dataset_path + "kitti_format/training", out_dir + "kitti_format/training")


cmd =  f'python tools/create_data.py waymo --root-path ' + root_path + ' --out-dir ' + out_dir + ' --workers 128 --extra-tag waymo --no-conversion'
# cmd =  f'python tools/create_data.py waymo --root-path ' + root_path + ' --out-dir ' + out_dir + ' --workers 1 --extra-tag waymo'
os.system(cmd)

# read all lines from multiple files in a folder and save as strings in a list
timestamps = []

import glob

list_of_files = glob.glob(out_dir + "kitti_format/training/timestamp/*.txt")           # create the list of file
for file_name in list_of_files:
    if file_name.strip(".txt").endswith(tuple(h)):
        FI = open(file_name, 'r')
        for line in FI:
            timestamps.append(line)
        FI.close()

objects_full = metrics_pb2.Objects()
with open(full_dataset_path + "waymo_format/gt.bin", 'rb') as f:
    objects_full.ParseFromString(f.read())

# Create reduced gt.bin file with only the objects and no label zone objects that has the timestamps in the list
objects_reduced = metrics_pb2.Objects()
for obj in objects_full.objects:
    if str(obj.frame_timestamp_micros) in timestamps:
        objects_reduced.objects.append(obj)

for no_label_zone in objects_full.no_label_zone_objects:
    if str(no_label_zone.frame_timestamp_micros) in timestamps:
        objects_reduced.no_label_zone_objects.append(no_label_zone)

with open(out_dir + "waymo_format/gt.bin", 'wb') as f:
    f.write(objects_reduced.SerializeToString())

