<div id="top" align="center">

<p align="center">
  <img src="assets/images/repo/title.jpg">
</p>
    
**Drive on Language:** *Unlocking the future where autonomous driving meets the unlimited potential of language.*
</div>

<div id="top" align="center">

![](https://komarev.com/ghpvc/?username=your-github-username)
  
<a href="#license-and-citation">
  <img alt="License: Apache2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"/>
</a>

<a href="#getting-start">
  <img src="https://img.shields.io/badge/Latest%20release-v1.0-yellow"/>
</a>

</div>


<div id="top" align="center">
<a href="https://opendrivelab.github.io/DriveLM" target="_blank">
    <img alt="Github Page" src="https://img.shields.io/badge/Project%20Page-white?logo=GitHub&color=green" />
  </a>
<a href="https://huggingface.co/datasets/OpenDrive/DriveLM" target="_blank">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DriveLM-ffc107?color=ffc107&logoColor=white" />
  </a>
<a href="https://twitter.com/OpenDriveLab" target="_blank">
    <img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/OpenDriveLab?style=social&color=brightgreen&logo=twitter" />
  </a>
<a href="https://twitter.com/AutoVisionGroup" target="_blank">
    <img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/Awesome Vision Group?style=social&color=brightgreen&logo=twitter" />
  </a>
<a href="https://opendrivelab.com" target="_blank">
<img src="https://img.shields.io/badge/contact%40opendrivelab.com-white?style=social&logo=gmail">
  </a>
</div>

<div id="top" align="center">
<a href="https://docs.google.com/forms/d/e/1FAIpQLSfm8k7LjITLRdXgbURxk46dq5Q2n8qGoRX0nWqQNE1U_322wQ/viewform?usp=sf_link" target="_blank">
  <img src="https://img.shields.io/badge/Any%20comments%20welcome!-white?logo=google%20forms&label=Google%20Forms&labelColor=blue">
</a>
</div>


<div id="top" align="center">
<p align="center">
 
</p>
</div>



https://github.com/OpenDriveLab/DriveLM/assets/103363891/3e40f63a-4873-4e7b-9f9d-4bfd254360e8



<!-- > demo scene token: cc8c0bf57f984915a77078b10eb33198 -->


## 🔥 Highlights of the DriveLM Dataset

#### In the view of general Vision Language Models
- 🌳 Structured reasoning, multi-modal **Graph-of-Thought** testbench.


https://github.com/OpenDriveLab/DriveLM/assets/103363891/f8018448-8a0a-4c50-9e0a-d5628147d4a8

 
#### In the view of full-stack autonomous driving
- 🛣 Completeness in functionality (covering **Perception**, **Prediction** and **Planning** QA pairs).


<p align="center">
  <img src="assets/images/repo/point_1.png">
</p>


- 🔜 Reasoning for future events that have not yet happened.
  - Many **"What If"**-style questions: imagine the future by language.
 

<p align="center">
  <img src="assets/images/repo/point_2.png" width=70%>
</p>

- ♻ Task-driven decomposition.
  - **One** scene-level text goal into **many** frame-level trajectories & planning text descriptions.

<p align="center">
  <img src="assets/images/repo/point_3.png">
</p>


## Table of Contents
- [News](#news)
- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [License and Citation](#license-and-citation)
- [Other Projects](#other-projects)

## News

- **`[2023/08/25]`** DriveLM dataset demo `v1.0` released.


<p align="right">(<a href="#top">back to top</a>)</p>

## Introduction

DriveLM is a project of driving on language, which contains both a `Dataset` and a `Model`. Through DriveLM, we introduce the reasoning ability of Large Language Models in autonomous driving (**AD**) to make decisions and ensure explainable planning.

Specifically, in the `Dataset` of DriveLM, we facilitate `Perception, Prediction, and Planning` (**P3**) with human-written reasoning logic as a connection. In the `Model`, we propose an AD Vision Language Model with the Graph-of-Thought ability to produce better planning results. Currently, a demo of the dataset has been released, and the full dataset and the model will be released in the future.

### What is Graph-of-Thoughts in AD?
The most exciting aspect of the dataset is that the questions and answers (`QA`) in `P3` are connected in a graph-style structure, with QA pairs as every node, and objects' relationships as the edges. Compared to [language-only Tree-of-Thought](https://github.com/princeton-nlp/tree-of-thought-llm) or [Graph-of-Thought](https://arxiv.org/abs/2305.16582), we go a step further towards multi-modality. The reason for doing this in the AD domain is that AD tasks are well-defined per stage, from raw sensor input to final control action.

### 📊 Comparison and stats: the *first* language-driving dataset facilitating P3 and logic

<center>
  
| Language Dataset  | Base Dataset |      Language Form    |   Perspectives | Scale      |  Release?|
|:---------:|:-------------:|:-------------:|:------:|:--------------------------------------------:|:----------:|
| [BDD-X 2018](https://github.com/JinkyuKimUCB/explainable-deep-driving)  |  [BDD](https://bdd-data.berkeley.edu/)  | Description | Planning Description & Justification    | 8M frames, 20k text strings   |**:heavy_check_mark:**|
| [HAD HRI Advice 2019](https://usa.honda-ri.com/had)  |  [HDD](https://usa.honda-ri.com/hdd)  | Advice | Goal-oriented & stimulus-driven advice | 5,675 video clips, 45k text strings   |**:heavy_check_mark:**|
| [Talk2Car 2019](https://github.com/talk2car/Talk2Car)   |      [nuScenes](https://www.nuscenes.org/)    | Description |  Goal Point Description | 30k frames, 10k text strings | **:heavy_check_mark:**|
| [DRAMA 2022](https://usa.honda-ri.com/drama)   |    - | Description |  QA + Captions | 18k frames, 100k text strings | **:heavy_check_mark:**|
| [nuScenes-QA 2023](https://arxiv.org/abs/2305.14836)   |   [nuScenes](https://www.nuscenes.org/)  | QA |  Perception Result     | 30k frames, 460k generated QA pairs| :x:|
| **DriveLM 2023** | [nuScenes](https://www.nuscenes.org/) | **:boom: QA + Scene Description** | **:boom:Perception, Prediction and Planning with Logic** | 30k frames, 360k annotated QA pairs |**:heavy_check_mark:** |

</center>

<p align="center">
  <img src="assets/images/repo/stats.jpeg">
</p>


### What is included in the DriveLM dataset?
We construct our dataset based on the prevailing nuScenes dataset. The most central element of DriveLM is frame-based `P3` `QA`. `Perception` questions require the model to recognize objects in the scene. `Prediction` questions ask the model to predict the future status of important objects in the scene. `Planning` questions prompt the model to give reasonable planning actions and avoid dangerous ones.


### How about the annotation process?

1️⃣ Keyframe selection. Given all frames in one clip, the annotator selects the keyframes that need annotation. The criterion is that those frames should involve changes in ego-vehicle movement status (lane changes, sudden stops, start after a stop, etc.).

2️⃣ Key objects selection. Given keyframes, the annotator needs to pick up key objects in the six surrounding images. The criterion is that those objects should be able to affect the action of the ego-vehicle (traffic signals, pedestrians crossing the road, other vehicles that move in the direction of the ego-vehicle, etc.).

3️⃣ Question and answer annotation. Given those key objects, we automatically generate questions regarding single or multiple objects about perception, prediction, and planning. More details can be found in our demo data.


<p align="right">(<a href="#top">back to top</a>)</p>


## Getting Started
- [Download Data](/docs/getting_started.md#download-data)
- [Prepare Dataset](/docs/getting_started.md#prepare-dataset)
- [Evaluation](/docs/getting_started.md#evaluation) **(TBA in the future)**

<p align="right">(<a href="#top">back to top</a>)</p>


## License and Citation
All assets and code in this repository are under the [Apache 2.0 license](./LICENSE) unless specified otherwise. The language data is under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Other datasets (including nuScenes) inherit their own distribution licenses. Please consider citing our project if it helps your research.

```BibTeX
@misc{drivelm2023,
  title={DriveLM: Drive on Language},
  author={DriveLM Contributors},
  howpublished={\url{https://github.com/OpenDriveLab/DriveLM}},
  year={2023}
}
```

<p align="right">(<a href="#top">back to top</a>)</p>

## Other Projects
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

**OpenDriveLab**
- [DriveAGI](https://github.com/OpenDriveLab/DriveAGI) | [UniAD](https://github.com/OpenDriveLab/UniAD) | [OpenLane-V2](https://github.com/OpenDriveLab/OpenLane-V2) | [Survey on E2EAD](https://github.com/OpenDriveLab/End-to-end-Autonomous-Driving)
- [Survey on BEV Perception](https://github.com/OpenDriveLab/BEVPerception-Survey-Recipe) | [BEVFormer](https://github.com/fundamentalvision/BEVFormer) | [OccNet](https://github.com/OpenDriveLab/OccNet)

**Autonomous Vision Group**
- [tuPlan garage](https://github.com/autonomousvision/tuplan_garage) | [CARLA garage](https://github.com/autonomousvision/carla_garage) | [Survey on E2EAD](https://github.com/OpenDriveLab/End-to-end-Autonomous-Driving)
- [PlanT](https://github.com/autonomousvision/plant) | [KING](https://github.com/autonomousvision/king)

<p align="right">(<a href="#top">back to top</a>)</p>
