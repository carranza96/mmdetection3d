from waymo_open_dataset.protos import metrics_pb2
from glob import glob
from os.path import join

objects = []
file_name_full = "/mnt/hd/mmdetection3d/data/waymo/waymo_format/gt_full.bin"
file_name_test = "/mnt/hd/mmdetection3d/data/waymo/waymo_format/gt_test_result.bin"

objects_full = metrics_pb2.Objects()
with open(file_name_full, 'rb') as f:
    objects_full.ParseFromString(f.read())

objects_test = metrics_pb2.Objects()
with open(file_name_test, 'rb') as f:
    objects_test.ParseFromString(f.read())

context_names_test = []
for i in objects_test.objects:
    if (i.context_name, i.frame_timestamp_micros) not in context_names_test:
        context_names_test.append((i.context_name, i.frame_timestamp_micros))

# for j,i in enumerate(objects_full.objects):
#     if i.frame_timestamp_micros in context_names_test:
#         print(j)
#         objects_full.objects.remove(i)


a = list(filter(lambda i: (i.context_name, i.frame_timestamp_micros) in context_names_test, objects_full.objects))
b = list(filter(lambda i: (i.context_name, i.frame_timestamp_micros) in context_names_test, objects_full.no_label_zone_objects))

objects_final = metrics_pb2.Objects()

for o in a:
    objects_final.objects.append(o)

for nlzo in b:
    objects_final.no_label_zone_objects.append(nlzo)

context_names_final = []
for i in objects_final.objects:
    if (i.context_name, i.frame_timestamp_micros) not in context_names_final:
        context_names_final.append((i.context_name, i.frame_timestamp_micros))


with open("/mnt/hd/mmdetection3d/data/waymo/waymo_format/gt.bin",'wb') as f:
    f.write(objects_final.SerializeToString())

# print(context_names)
print("New full lenght: ", len(context_names_final))
print("Part lenght: ", len(context_names_test))