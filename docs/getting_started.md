## Download data
To get started, download nuScenes V1.0 full dataset data and CAN bus expansion data [HERE](https://www.nuscenes.org/download).

Next, you should download the dataset of DriveLM. Note that to download the DriveLM dataset, you need fill out a google form and we will send you the download link afterwards.


- `DriveLM version-1.0 demo` [Download Link]().
## Prepare dataset

Follow the steps [HERE](https://github.com/fundamentalvision/BEVFormer/blob/master/docs/prepare_dataset.md) to prepare nuScenes dataset. Using the above code will generate `nuscenes_infos_temporal_{train,val}.pkl`.

#### Folder structure.
```
DriveLM
├── data/
│   ├── QA_dataset/
│   │   ├── train.json
│   │   ├── val.json
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

The QA pairs are in the `{train,val}.json`. Below is the json file structure.
```
train.json
├── scene_token:{
│   ├── scene_description:
│   ├── key_frames:{
│   │   ├── CAM_FRONT_timestamp_1:{
│   │   │   ├── Perception:{
│   │   │   │   ├──q:["Q: XXX", ...]
│   │   │   │   ├──a:["A: XXX", ...]
│   │   │   │   ├──description:{
│   │   │   │   │   ├── <c1>: <c1> is a moving car to the front of ego-car
│   │   │   │   ├──}
│   │   │   ├──}
│   │   │   ├── Prediction and Planning:{
│   │   │   │   ├──q:[]
│   │   │   │   ├──a:[]
│   │   │   ├──}
│   │   ├── CAM_FRONT_timestamp_2:{
│   │   │   ├── Perception:{
│   │   │   │   ├──q:[]
│   │   │   │   ├──a:[]
│   │   │   │   ├──description:{
│   │   │   │   │   ├── <c1>: <c1> is a moving car to the front of ego-car
│   │   │   │   ├──}
│   │   │   ├──}
│   │   │   ├── Prediction and Planning:{
│   │   │   │   ├──q:[]
│   │   │   │   ├──a:[]
│   │   │   ├──}
│   │   ├── ... }
│   │   ├──}
│   ├──}
├──}
```

- `scene_token` is the same as in nuScenes dataset.
- Under `key_frames`, each key frames are identified by the CAM_FRONT timestampt, which is the same as the CAM_FRONT timestamp in nuScenes dataset.
- `scene_description` is a one-sentence summary of ego-vehicle behavior in the 20-seconds video clip (the notion of scene in nuScenes dataset).
- `q` and `a` are python list, with each element a string of either `question` or `answer`.
- The `description` under `Perception` is a mapping between `c tag` (i.e. \<c1\>) and its textual description of visual appearance.

## Evaluation

To be announced in the future!
