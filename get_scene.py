import os
import sys
task_path = "data/datasets/objectnav/hm3d/objectnav_hm3d_v1/train/content1"
data_list = sorted(os.listdir(task_path))
scene_list = []
for scene in data_list:
    scene_name = scene.split(".")[0]
    scene_list.append(scene_name)
data_list = []
scene_path = "/mnt/yupeng/OGN/Object-Goal-Navigation/data/scene_datasets/hm3d/train" 
data_list = sorted(os.listdir(scene_path))
for scene in scene_list:
    for dir in data_list:
        if dir.split("-")[-1] == scene:
            print(dir)