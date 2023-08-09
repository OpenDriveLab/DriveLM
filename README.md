<div id="top" align="center">

# DriveLM

**Drive on Language: exploring the possibility of connecting autonomous driving and large language models**

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
- [Highlights](#highlights)
- [More Details](#more-details)
- [TODO](#todo)
- [Getting Started](#getting-started)
- [License and Citation](#license-and-citation)
- [Related Resources](#related-resources)

## Highlights

### :car: DriveLM:
Introduce the reasoning ability of large language model in AD to make decision and ensure explainable planning.


### :fire: DriveLM: The *first* language-driving dataset facilitating P3 and logic

In the dataset of DriveLM, we facilitates Perception, Prediction and Planning (P3) with human-written reasoning logic as connection.

<center>
  
|  Dataset  | Base Dataset |      Language Form    |   Content | Scale      |  Release?|
|:---------:|:-------------:|:--------------------:|:------:|:--------------------------------------------:|:----------:|
| [BDD-X 2018](https://github.com/JinkyuKimUCB/explainable-deep-driving)  |  BDD  | Description | Planning description & Justification    | 8M frame,20k text   |**:heavy_check_mark:**|
| [Talk2Car 2019](https://github.com/talk2car/Talk2Car)   |      nuScenes    | Description |  Goal point Description | 30k frame,10k text | **:heavy_check_mark:**|
| [nuScenes-QA 2023](https://arxiv.org/abs/2305.14836)   |   nuScenes  | VQA |  Perception result     | 30k frame, 460k text| :x:|
| **DriveLM 2023** | nuScenes| **:boom: VQA+Description** | **.:boom:Perception, Prediction and Planning Logic** | 30k frame, 600k text|**:heavy_check_mark:** Mid August|

</center>







### :fire: DriveLM:

AD visual-language model with chain-of-thought ability to produce better planning result





<p align="right">(<a href="#top">back to top</a>)</p>




## More Details

### Data Part
#### Perception
#### Prediction
#### Planning

### Model Part





## TODO 
- [x] DriveLM `v1.0`
- [ ] DriveLM `v1.1`
- [ ] DriveLM `v2.0`

<p align="right">(<a href="#top">back to top</a>)</p>


## Getting Started
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
