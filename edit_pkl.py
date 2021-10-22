import pickle

with open("data/nuscenes/nuscenes_infos_val.pkl", 'rb') as f:
    data = pickle.load(f)

new_data = dict()
new_data['metadata'] = data['metadata']
new_data['infos'] = [data['infos'][1]]

with open("data/nuscenes/nuscenes_infos_val_sample.pkl", "wb") as f2:
    pickle.dump(new_data, f2)

print()