<div id="top" align="center">

#  DriveLM  🚀

**Drive on Language:** *exploring the possibility of connecting autonomous driving and large language models*

<a href="#license-and-citation">
  <img alt="License: Apache2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"/>
</a>



<p align="center">
  <img src="demo.gif">
</p>

</div>

> - Point of contact: [contact@opendrivelab.com](mailto:contact@opendrivelab.com)

## News

- **`[2023/08/07]`** DriveLM `v1.0` released

## Table of Contents
- [Introduction](#introduction)
- [Highlights](#highlights)
- [More Details](#more-details)
- [Getting Started](#getting-started)
- [License and Citation](#license-and-citation)
- [Related Resources](#related-resources)


## Introduction
DriveLM is an open source project, which contains both `Dataset` and `Model`. Through DriveLM, we introduce the reasoning ability of large language model in AD to make decision and ensure explainable planning.


Specifically, in the `Dataset` of DriveLM, we facilitates Perception, Prediction and Planning (P3) with human-written reasoning logic as connection. And in the `Model` part, we propose an AD visual-language model with chain-of-thought ability to produce better planning result.


## Highlights


### :fire: DriveLM: The *first* language-driving dataset facilitating P3 and logic

<center>
  
|  Dataset  | Base Dataset |      Language Form    |   Content | Scale      |  Release?|
|:---------:|:-------------:|:--------------------:|:------:|:--------------------------------------------:|:----------:|
| [BDD-X 2018](https://github.com/JinkyuKimUCB/explainable-deep-driving)  |  BDD  | Description | Planning description & Justification    | 8M frame,20k text   |**:heavy_check_mark:**|
| [Talk2Car 2019](https://github.com/talk2car/Talk2Car)   |      nuScenes    | Description |  Goal point Description | 30k frame,10k text | **:heavy_check_mark:**|
| [nuScenes-QA 2023](https://arxiv.org/abs/2305.14836)   |   nuScenes  | VQA |  Perception result     | 30k frame, 460k text| :x:|
| **DriveLM 2023** | nuScenes| **:boom: VQA+Description** | **:boom:Perception, Prediction and Planning Logic** | 30k frame, 600k text|**:heavy_check_mark:** Mid August|

</center>


#### In the view of general VLM
- Structured-reasoning, Tree-of-Thought testbench

 
#### In the view of autonomous driving
- Full-stack, completeness in functionality (covering perception, prediction and planning)
- Reasoning for future events that does not even happened
  - Many "what if"-style question, imagine the future by language
- Task-driven Decomposition.One scene-level text-goal into multiple frame-level trajectory & planning-text




<p align="right">(<a href="#top">back to top</a>)</p>




## More Details

### What's included in DriveLM v1?
1️⃣In the version 1.0 of DriveLM, we construct our dataset based on the nuScenes. The most central element of DriveLM is scenario-based Q&A. Basically, we divide our Q&A pairs into three part: `Perception`, `Prediction` and `Planning`.


2️⃣On the other hand, DriveLM v1.0 contains two main parts:`Train` and `Validation`. `Train` contains 697 scenarios and corresponding Q&A pairs. And `Validation` contains 150 scenerios and  corresponding Q&A pairs.


3️⃣In each scene, there are about 40 keyframes（Sampling frequency is approximately 2 Hz）, and we select one every five keyframes for annotating.And in each keyframe selected,we give the labels and gt-boxes of the objects we are interested in.

### What's included in DriveLM v2?
1️⃣In the version 2.0 of DriveLM, we've made significant improvements: Instead of selecting the frames and objects to be annotated beforehand, we leave it up to the annotator to choose the frames and objects of interest among all the keyframes.


2️⃣Besides,we've increased the freedom of the Q&A pairs by adding customisable questions, which also increase the applicability of Q&A pairs in more diverse scenarios.

<p align="right">(<a href="#top">back to top</a>)</p>


## Getting Start
- [Download Data](/docs/getting_started.md#download-data)
- [Prepare Dataset](/docs/getting_started.md#prepare-dataset)
- [Train a Model](/docs/getting_started.md#train-a-model)


<p align="right">(<a href="#top">back to top</a>)</p>


## License and Citation
All assets (including figures and data) and code are under the [Apache 2.0 license](./LICENSE) unless specified otherwise.
Please consider citing our paper if the project helps your research with the following BibTex:

```bibtex
@misc{openscene2023,
      author = {DriveLM Contributors},
      title = {},
      url = {https://github.com/OpenDriveLab/DriveLM},
      year = {2023}
}

@article{sima2023_occnet,
      title={}, 
      author={},
      year={},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<p align="right">(<a href="#top">back to top</a>)</p>

## Related Resources
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
- [DriveAGI](https://github.com/OpenDriveLab/DriveAGI)  | [OpenLane-V2](https://github.com/OpenDriveLab/OpenLane-V2)
- [Survey on Bird's-eye-view Perception](https://github.com/OpenDriveLab/BEVPerception-Survey-Recipe) | [BEVFormer](https://github.com/fundamentalvision/BEVFormer) |  [OccNet](https://github.com/OpenDriveLab/OccNet)


<p align="right">(<a href="#top">back to top</a>)</p>
