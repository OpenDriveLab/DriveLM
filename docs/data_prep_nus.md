## Download data
We kindly hope you to fill out the [form](https://docs.google.com/forms/d/e/1FAIpQLSeX6CR3u-15IV-TKx2uPv1wiKjydjZ__NNW98H4nR5JZtQa2Q/viewform) before downloading. To get started, download nuScenes subset image data and DriveLM-nuScenes QA json files below. For v1.1 data, please visit the [DriveLM/challenge](https://github.com/OpenDriveLab/DriveLM/tree/main/challenge) folder.

<!-- <a href="https://docs.google.com/forms/d/e/1FAIpQLSfm8k7LjITLRdXgbURxk46dq5Q2n8qGoRX0nWqQNE1U_322wQ/viewform?usp=sf_link" target="_blank">
  <img src="https://img.shields.io/badge/Any%20comments%20welcome!-white?logo=google%20forms&label=Google%20Forms&labelColor=blue">
</a>  -->

| nuScenes subset images | DriveLM-nuScenes version-1.0|
|-------|-------|
| [Google Drive](https://drive.google.com/file/d/1DeosPGYeM2gXSChjMODGsQChZyYDmaUz/view?usp=sharing) | [Google Drive](https://drive.google.com/file/d/1LK7pYHytv64neN1626u6eTQBy1Uf4IQH/view?usp=sharing) |
|[Baidu Netdisk](https://pan.baidu.com/s/11xvxPzUY5xTIsJQrYFogqg?pwd=mk95)|[Baidu Netdisk](https://pan.baidu.com/s/1PAdotDY0MN3nkz8w_XhDsw?pwd=l4wf) |
|[HuggingFace](https://huggingface.co/datasets/OpenDriveLab/DriveLM/blob/main/drivelm_nus_imgs_train.zip)|[HuggingFace](https://huggingface.co/datasets/OpenDriveLab/DriveLM/blob/main/v1_0_train_nus.json)

You can also download the full nuScenes dataset [HERE](https://www.nuscenes.org/download) to enable video input. 
 
Our DriveLM dataset contains a collection of questions and answers. Currently, only the training set is publicly available. The dataset is named `v1_0_train_nus.json`.

<!-- - `v1_0_train.json`/`v1_0_val.json`: In this file, questions and answers are not augmented using GPT-3.5/4.0. The answers tend to follow relatively fixed patterns, resulting in straightforward and less diverse responses. -->

<!-- - `gpt_augmented_v1_0_train.json`/`gpt_augmented_v1_0_val.json`: Unlike the previous file, questions and answers in this version have been augmented using GPT. This optimization enhances the diversity of Q&A pairs. Consequently, responses are not limited to simple and direct Q&A, but may include richer expressions and content. -->
## Prepare the dataset

Organize the data structure as follows:

```
DriveLM
├── data/
│   ├── QA_dataset_nus/
│   │   ├── v1_0_train_nus.json
│   ├── nuscenes/
│   │   ├── samples/
```


#### File structure

The QA pairs are in the `v1_0_train_nus.json`. Below is the json file structure. All `coordinates` mentioned are referenced from the `upper-left` corner of the respective camera, with the `right` and `bottom` directions serving as the positive x and y axes, respectively.
```
v1_0_train_nus.json
├── scene_token:{
│   ├── "scene_description": "The ego vehicle proceeds along the current road, preparing to enter the main road after a series of consecutive right turns.",
│   ├── "key_frames":{
│   │   ├── "frame_token_1":{
│   │   │   ├── "key_object_infos":{"<c1,CAM_FRONT,258.3,442.5>": {"Category": "Vehicle", "Status": "Moving", "Visual_description": "White Sedan", "2d_bbox": [x_min, y_min, x_max, y_max]}, ...},
│   │   │   ├── "QA":{
│   │   │   │   ├── "perception":[
│   │   │   │   │   ├── {"Q": "What are the important objects in the current scene?", "A": "The important objects are <c1,CAM_FRONT,258.3,442.5>, <c2,CAM_FRONT,1113.3,505.0>, ...", "C": None, "con_up": None, "con_down": None, "cluster": None, "layer": None},
│   │   │   │   │   ├── {"Q": "xxx", "A": "xxx", "C": None, "con_up": None, "con_down": None, "cluster": None, "layer": None}, ...
│   │   │   │   ├── ],
│   │   │   │   ├── "prediction":[
│   │   │   │   │   ├── {"Q": "What is the future state of <c1,CAM_FRONT,258.3,442.5>?", "A": "Slightly offset to the left in maneuvering.", "C": None, "con_up": None, "con_down": None, "cluster": None, "layer": None}, ...
│   │   │   │   ├── ],
│   │   │   │   ├── "planning":[
│   │   │   │   │   ├── {"Q": "In this scenario, what are safe actions to take for the ego vehicle?", "A": "Brake gently to a stop, turn right, turn left.", "C": None, "con_up": None, "con_down": None, "cluster": None, "layer": None}, ...
│   │   │   │   ├── ],
│   │   │   │   ├── "behavior":[
│   │   │   │   │   ├── {"Q": "Predict the behavior of the ego vehicle.", "A": "The ego vehicle is going straight. The ego vehicle is driving slowly.", "C": None, "con_up": None, "con_down": None, "cluster": None, "layer": None}
│   │   │   │   ├── ]
│   │   │   ├── },
│   │   │   ├── "image_paths":{
│   │   │   │   ├── "CAM_FRONT": "xxx",
│   │   │   │   ├── "CAM_FRONT_LEFT": "xxx",
│   │   │   │   ├── "CAM_FRONT_RIGHT": "xxx",
│   │   │   │   ├── "CAM_BACK": "xxx",
│   │   │   │   ├── "CAM_BACK_LEFT": "xxx",
│   │   │   │   ├── "CAM_BACK_RIGHT": "xxx",
│   │   │   ├── }
│   │   ├── },
│   │   ├── "frame_token_2":{
│   │   │   ├── "key_object_infos":{"<c1,CAM_BACK,612.5,490.6>": {"Category": "Traffic element", "Status": "None", "Visual_description": "Stop sign", "2d_bbox": [x_min, y_min, x_max, y_max]}, ...},
│   │   │   ├── "QA":{
│   │   │   │   ├── "perception":[...],
│   │   │   │   ├── "prediction":[...],
│   │   │   │   ├── "planning":[...],
│   │   │   │   ├── "behavior":[...]
│   │   │   ├── },
│   │   │   ├── "image_paths":{...}
│   │   ├── }
│   ├── }
├── }
```

- `scene_token` is the same as in nuScenes dataset.
- `scene_description` is a one-sentence summary of ego-vehicle behavior in the about 20-second video clip (the notion of a scene in nuScenes dataset).
- Under `key_frames`, each key frame is identified by the `frame_token`, which corresponds to the `token` in the nuScenes dataset.
- The `key_object_infos` is a mapping between `c tag` (i.e. \<c1,CAM_FRONT,258.3,442.5\>) and more information about the related key objects such as the category, the status, the visual description, and the 2d bounding box.
- `QA` is divided into different tasks, and QA pairs under each task are formulated as a list of dictionaries. Each dictionary encompasses keys of `Q` (question), `A` (answer), `C` (context), `con_up`, `con_down`, `cluster`, and `layer`. Currently, the values of context related keys are set to None, serving as a tentative placeholder for future fields related to DriveLM-CARLA.


**Note:** The `c tag` label is used to indicate key objects selected during the annotation process. These objects include not only those present in the ground truth but also objects that are not, such as landmarks and traffic lights. Each key frame contains a minimum of three and a maximum of six key objects. The organization format of the `c tag` is `<c,CAM,x,y>`, where c is the identifier, CAM indicates the camera where the key object’s center point is situated, and x, y represent the horizontal and vertical coordinates of the 2D bounding box in the respective camera’s coordinate system with the `upper-left` corner as the `origin`, and the `right` and `bottom` as the `positive x and y axes`, respectively. 

In contrast to the `c tag`, for the question "Identify all the traffic elements in the front view," the output is presented as a list formatted as `[(c, s, x1, y1, x2, y2), ...]`. Here, `c` denotes the category, `s` represents the status, and `x1, y1, x2, y2` indicate the offsets of the top-left and bottom-right corners of the box relative to the center point.


<p align="center">
  <img width="671" alt="data" src="https://github.com/OpenDriveLab/DriveLM-new/assets/75412366/58d3a3f9-93b1-4899-a1c2-93c04a5978f0" width=90%>
</p>

