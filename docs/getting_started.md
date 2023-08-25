## Download data
DriveLM is the **first** language-driving dataset facilitating P3(`Perception`, `Prediction` and `Planning`) and human-written logic reasoning, which is now based on nuScenes.Therefore, you should download [nuScenes](https://www.nuscenes.org/) first to get started.Next, you can download different versions of DriveLM. The following are the versions we have released.


- `version-1.0` [Download in Github](https://github.com/OpenDriveLab/DriveLM), [Download in Huggingface](https://huggingface.co/datasets/OpenDrive/DriveLM)
## Prepare dataset
*We genetate custom annotation files which are different from mmdet3d's*
```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0 --canbus ./data
```

Using the above code will generate `nuscenes_infos_temporal_{train,val}.pkl`.

#### Folder structure.
```
DriveLM
├── models/
├── data/
│   ├── DriveLM_dataset/
│   │   ├── train/
│   │   ├── val/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── nuscenes_infos_temporal_train.pkl
|   |   ├── nuscenes_infos_temporal_val.pkl
```


### File structure
```
train.json
├── scene_token:{
│   ├── scene_description:
│   ├──key_frame:{
│   │   ├── timestamp:{
│   │   │   ├── Perception:{
│   │   │   │   ├──q:[]
│   │   │   │   ├──a:[]
│   │   │   │   ├──description:{
│   │   │   │   │   ├──<c1>: <c1> is a moving car to the front of ego-car
│   │   │   │   ├──}
│   │   │   ├──}
│   │   │   ├── Prediction:{
│   │   │   │   ├──q:[]
│   │   │   │   ├──a:[]
│   │   │   ├──}
│   │   │   ├── Planning:{
│   │   │   │   ├──q:[]
│   │   │   │   ├──a:[]
│   │   │   ├──}
│   │   ├──}
│   ├──}
├──}


## Evaluation
