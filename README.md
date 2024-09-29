<div id="top" align="center">

<p align="center">
  <img src="assets/images/repo/title_v2.jpg">
</p>
    
**DriveLM:** *Driving with **G**raph **V**isual **Q**uestion **A**nswering*

</div>

## Highlights
 
🔥 We present datasets (**DriveLM-Data**) built on nuScenes and CARLA, and propose a VLM-based baseline approach (**DriveLM-Agent**) for jointly performing Graph VQA and end-to-end driving.

<!-- 🔥 **The key insight** is that with our proposed suite, we obtain a suitable proxy task to mimic the human reasoning process during driving.  -->

<p align="center">
  <img src="assets/images/repo/drivelm_teaser.jpg">
</p>

## Table of Contents
1. [DriveLM-Data](#drivelmdata)
   - [Comparison and Stats](#comparison)
   - [GVQA Details](docs/gvqa.md)
   - [Annotation and Features](docs/data_details.md)
2. [Dataset](#dataset)
3. [GVQA Generation](#gvqa_generation)
4. [Custom Dataset Generation & PDM-Lite](#custom_dataset_and_pdm_lite)
5. [Current Endeavors and Future Horizons](#timeline)
7. [License and Citation](#licenseandcitation)
8. [Other Resources](#otherresources)

## DriveLM-Data <a name="drivelmdata"></a>

We facilitate the `Perception, Prediction, Planning, Behavior, Motion` tasks with human-written reasoning logic as a connection between them. We propose the task of [GVQA](docs/gvqa.md) on the DriveLM-Data. 

### 📊 Comparison and Stats <a name="comparison"></a>
**DriveLM-Data** is the *first* language-driving dataset facilitating the full stack of driving tasks with graph-structured logical dependencies.

<p align="center">
  <img src="assets/images/repo/paper_data_comp.png">
</p>

For more details, see [GVQA task](docs/gvqa.md), [Dataset Features](docs/data_details.md/#features), and [Annotation](docs/data_details.md/#annotation).

<p align="right">(<a href="#top">back to top</a>)</p>

## Graph Visual Question Answering (GVQA) Dataset <a name="dataset"></a>
We provide a GVQA dataset, featuring 71,246 keyframes out of 214631 total frames across 1,759 routes with 100% completion and zero infractions. All scripts to reproduce the following can be found [HERE](vqa_dataset).

1. Download the PDM-Lite dataset (330+ GB extracted).
  **Note:** This dataset is based on the PDM-Lite expert with improvements integrated from ["Tackling CARLA Leaderboard 2.0 with
End-to-End Imitation Learning"](https://kashyap7x.github.io/assets/pdf/students/Zimmerlin2024.pdf)
```
bash download_pdm_lite_carla_lb2.sh
```
2. Get DriveLM-VGQA labels and keyframes:
```
wget https://huggingface.co/datasets/OpenDriveLab/DriveLM/resolve/main/drivelm_carla_keyframes.txt
wget https://huggingface.co/datasets/OpenDriveLab/DriveLM/resolve/main/drivelm_carla_vqas.zip
unzip drivelm_carla_vqas.zip
```

<p align="right">(<a href="#top">back to top</a>)</p>
  
## GVQA Generation (Optional) <a name="gvqa_generation"></a>

Extract keyframes:
```
python3 extract_keyframes.py --path-dataset /path/to/data --path-keyframes /path/to/save/keyframes.txt
```

Generate Graph-VQAs:
```
python3 generate_qas_drivelm_carla.py --path-keyframes /path/to/keyframes.txt --data-directory /path/to/data --output-graph-directory /path/to/output
```

Optional arguments:
- ```--sample-frame-mode```: Specify how to select frames, choose from 'all', 'keyframes', or 'uniform'.
- ```--sample-uniform-interval```: Specify interval for uniform sampling.
- ```--save-examples```: Save example images for debugging.
- ```--visualize-projection```: Visualize object centers in images.

<p align="right">(<a href="#top">back to top</a>)</p>

## Custom Dataset Generation & PDM-Lite <a name="custom_dataset_and_pdm_lite"></a>

For instructions on generating your own dataset with CARLA Leaderboard 2.0 and the PDM-Lite implementation, see [HERE](pdm_lite)

<p align="right">(<a href="#top">back to top</a>)</p>

## Current Endeavors and Future Directions  <a name="timeline"></a>
> - The advent of GPT-style multimodal models in real-world applications motivates the study of the role of language in driving.
> - Date below reflects the arXiv submission date.
> - If there is any missing work, please reach out to us!

<p align="center">
  <img src="assets/images/repo/drivelm_timeline_v3.jpg">
</p>

DriveLM attempts to address some of the challenges faced by the community.

- **Lack of data**: DriveLM-Data serves as a comprehensive benchmark for driving with language.
- **Embodiment**: GVQA provides a potential direction for embodied applications of LLMs / VLMs.
- **Closed-loop**: DriveLM-CARLA attempts to explore closed-loop planning with language.

<p align="right">(<a href="#top">back to top</a>)</p>

## License and Citation <a name="licenseandcitation"></a>
All assets and code in this repository are under the [Apache 2.0 license](./LICENSE) unless specified otherwise. The language data is under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Other datasets (including nuScenes) inherit their own distribution licenses. Please consider citing our paper and project if they help your research.

```BibTeX
@article{sima2023drivelm,
  title={DriveLM: Driving with Graph Visual Question Answering},
  author={Sima, Chonghao and Renz, Katrin and Chitta, Kashyap and Chen, Li and Zhang, Hanxue and Xie, Chengen and Luo, Ping and Geiger, Andreas and Li, Hongyang},
  journal={arXiv preprint arXiv:2312.14150},
  year={2023}
}
```

```BibTeX
@misc{contributors2023drivelmrepo,
  title={DriveLM: Driving with Graph Visual Question Answering},
  author={DriveLM contributors},
  howpublished={\url{https://github.com/OpenDriveLab/DriveLM}},
  year={2023}
}
```

<p align="right">(<a href="#top">back to top</a>)</p>

## Other Resources <a name="otherresources"></a>
<a href="https://twitter.com/OpenDriveLab" target="_blank">
    <img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/OpenDriveLab?style=social&color=brightgreen&logo=twitter" />
  </a>

<!-- <a href="https://opendrivelab.com" target="_blank">
  <img src="https://img.shields.io/badge/contact%40opendrivelab.com-white?style=social&logo=gmail">
</a> -->

<!--
 [![Page Views Count](https://badges.toozhao.com/badges/01H9CR01K73G1S0AKDMF1ABC73/blue.svg)](https://badges.toozhao.com/stats/01H9CR01K73G1S0AKDMF1ABC73 "Get your own page views count badge on badges.toozhao.com")
-->

**OpenDriveLab**
- [DriveAGI](https://github.com/OpenDriveLab/DriveAGI) | [UniAD](https://github.com/OpenDriveLab/UniAD) | [OpenLane-V2](https://github.com/OpenDriveLab/OpenLane-V2) | [Survey on E2EAD](https://github.com/OpenDriveLab/End-to-end-Autonomous-Driving)
- [Survey on BEV Perception](https://github.com/OpenDriveLab/BEVPerception-Survey-Recipe) | [BEVFormer](https://github.com/fundamentalvision/BEVFormer) | [OccNet](https://github.com/OpenDriveLab/OccNet)

<a href="https://twitter.com/AutoVisionGroup" target="_blank">
    <img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/Awesome Vision Group?style=social&color=brightgreen&logo=twitter" />
  </a>

**Autonomous Vision Group**
- [tuPlan garage](https://github.com/autonomousvision/tuplan_garage) | [CARLA garage](https://github.com/autonomousvision/carla_garage) | [Survey on E2EAD](https://github.com/OpenDriveLab/End-to-end-Autonomous-Driving)
- [PlanT](https://github.com/autonomousvision/plant) | [KING](https://github.com/autonomousvision/king) | [TransFuser](https://github.com/autonomousvision/transfuser) | [NEAT](https://github.com/autonomousvision/neat)

<p align="right">(<a href="#top">back to top</a>)</p>
