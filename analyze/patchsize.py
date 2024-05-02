import numpy as np

rois = [
    {'mid': np.array([160.5, 289, 149]), 'name': 'normal1'},
    {'mid': np.array([152.5, 230.5, 168.5]), 'name': 'normal2'},
    {'mid': np.array([324.5, 332, 157.5]), 'name': 'cyst1'},
    {'mid': np.array([188.5, 278, 185.5]), 'name': 'cyst2'},
    {'mid': np.array([209.5, 316, 158.5]), 'name': 'hemangioma'},
    {'mid': np.array([110, 273.5, 140.5]), 'name': 'metastasis'},
]

def overlap(b1, b2):
    return all(b1['end'][i] >= b2['start'][i] and b1['start'][i] <= b2['end'][i] for i in range(3))

def find_overlaps(rois, size):
    pairs = []
    for i in range(len(rois)):
        for j in range(i+1, len(rois)):
            box1 = {'start': rois[i]['mid'] - size / 2, 'end': rois[i]['mid'] + size / 2}
            box2 = {'start': rois[j]['mid'] - size / 2, 'end': rois[j]['mid'] + size / 2}
            if overlap(box1, box2):
                pairs.append((rois[i]['name'], rois[j]['name']))
    return pairs

size_x, size_y, size_z = 1, 1, 1  
while True:
    overlaps = find_overlaps(rois, np.array([size_x, size_y, size_z]))
    if overlaps:
        print(f"Tailles: x={size_x}, y={size_y}, z={size_z}, Chevauchements: {overlaps}")
        break
    size_x += 1  
    size_y += 1
    size_z += 1
