## Download data
First, you should download [nuScenes](https://www.nuscenes.org/) first to get started. Next, you can download different versions of DriveLM. The following are the versions we have released.


- `version-1.0` [Download in Google Drive](https://drive.google.com/file/d/1HTx7N1N00H8LfU4isovnFRfYhVUXSLUC/view?usp=drive_link), [Download in Huggingface](https://huggingface.co/datasets/OpenDrive/DriveLM)
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
```

## Evaluation
