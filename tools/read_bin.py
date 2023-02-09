from waymo_open_dataset.protos import metrics_pb2
from glob import glob
from os.path import join

objects = []
file_name_full = "/mnt/hd/mmdetection3d/data/waymo/waymo_format/gt.bin"
# file_name_test = "/mnt/hd/mmdetection3d/data/waymo/waymo_format/gt_test_result.bin"
file_name_test = "/mnt/hd/mmdetection3d/data/waymo/waymo_format/gt_full.bin"

combined = metrics_pb2.Objects()


objects_full = metrics_pb2.Objects()
with open(file_name_full, 'rb') as f:
    objects_full.ParseFromString(f.read())

objects_2 = metrics_pb2.Objects()
with open(file_name_test, 'rb') as f:
    objects_2.ParseFromString(f.read())

objects_names_test = []
for i in objects_2.objects:
    if (i.context_name, i.frame_timestamp_micros) not in objects_names_test:
        objects_names_test.append((i.context_name, i.frame_timestamp_micros))

no_objects_names_test = []
for i in objects_2.no_label_zone_objects:
    if (i.context_name, i.frame_timestamp_micros) not in no_objects_names_test:
        no_objects_names_test.append((i.context_name, i.frame_timestamp_micros))

combined_objects_names_test = []
for i in objects_2.objects:
    if (i.context_name, i.frame_timestamp_micros) not in combined_objects_names_test:
        combined_objects_names_test.append((i.context_name, i.frame_timestamp_micros))
for i in objects_2.no_label_zone_objects:
    if (i.context_name, i.frame_timestamp_micros) not in combined_objects_names_test:
        combined_objects_names_test.append((i.context_name, i.frame_timestamp_micros))

# context_names = []

# print(objects_names_test)
# print(objects_full)
# print("Full lenght: ", len(context_names_full))
print("Part Object Lenght: ", len(objects_names_test))
print("Part No Object Lenght: ", len(no_objects_names_test))
print("Part Combined Object Lenght: ", len(combined_objects_names_test))