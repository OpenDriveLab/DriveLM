### TL;DR
* The purpose of this folder is to facilitate our CVPR 2024 challenge. Initially, we will use a small subset of training data (**demo train** in the following text) as an illustrative example, demonstrating how to obtain the **test data format** and the submission format, how to train the baseline and infer the baseline, and go through the evaluation pipeline. 

* For the purpose of the new **test data format**, it is essential that our primary intention is to create a specific test data format preventing possible cheating. 

<!-- > * Subsequently, we will demonstrate the process of conducting evaluations, encompassing the baseline methodology. -->

* For better illustration, we provide [google slides](https://docs.google.com/presentation/d/1bicxoR_L3t05p5xw-qZM0Dj5KdJhjynqLM0Rck0qdcI/edit?usp=sharing) for your reference. 

* **Official announcement about the DriveLM challenge is maintained in this folder**. Please raise an issue in the repo if you find anything unclear.

## How to Prepare Data

### DriveLM
Download the full DriveLM data [v1_0_train_nus.json](https://drive.google.com/file/d/1LK7pYHytv64neN1626u6eTQBy1Uf4IQH/view?usp=sharing), and the demo train DriveLM data [train_sample.json](https://drive.google.com/file/d/1pDikp6xoZGdyUS75qCqCM-Bh5-DWLyj-/view?usp=drive_link).
The code can run on both full and sampled data, and we provide the entire process of running on the demo train data as follows:

Follow the steps below to get the test data format as well as data for the baseline model.

### Extract Data

Extract fundamental question-and-answer (QA) pairs from the training dataset. 

**Note that** the number and the content of the fundamental QA pairs might change in the test server, but we ensure that **all the question types are limited in our provided test data format**. That being said, the question types are within 1) multi-choice question; 2) conversation question; 3) yes/no question;

```bash
# The following script assumes that you download the train data json under ./challenge/data
# make sure you are under ./challenge
mkdir data
mv train_sample.json data/train_sample.json
python extract_data.py
```
Then we will get the test.json in the challenge folder. The example of test.json can be found in [test.json](test.json)

### Convert Data

Transform the obtained test.json data into the required test format.

```bash
# The following script assumes that you download the train data json under ./challenge/data
# make sure you are under ./challenge
python convert_data.py
```
Then we will get the test_v1.json in challenge folder. The example of test_v1.json can be found in [test_v1.json](test_v1.json)

We use llama-adapter v2 as our baseline. If you want to convert data into llama-adapter format:
```bash
# The following script assumes that you prepare the test_v1.json under ./challenge
# make sure you are under ./challenge
python convert2llama.py
```
Then we will get the test_v2.json in challenge folder. The example of test_v2.json can be found in [test_v2.json](test_v2.json)

[test_v1.json](test_v1.json) is used for evaluation. [test_v2.json](test_v2.json) is used for training and inference of the baseline.

## How to run baseline

As we said above, we use [llama-adapter v2](https://github.com/OpenGVLab/LLaMA-Adapter/tree/main/llama_adapter_v2_multimodal7b) as our baseline.

### Setup
We provide a simple setup script below, and you can also refer to [docs](llama_adapter_v2_multimodal7b/README.md#L9) for more specific installation.
* setup up a new conda env and install necessary packages.
```bash
# make sure you are under ./challenge/llama_adapter_v2_multimodal7b
conda create -n llama_adapter_v2 python=3.8 -y
conda activate llama_adapter_v2
pip install -r requirements.txt
```

* Obtain the LLaMA pretrained weights using this [form](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform?usp=send_form). Please note that checkpoints from unofficial sources (e.g., BitTorrent) may contain malicious code and should be used with care. Organize the downloaded file in the following structure
```bash
/path/to/llama_model_weights
├── 7B
│   ├── checklist.chk
│   ├── consolidated.00.pth
│   └── params.json
└── tokenizer.model
```

### Train baseline
You should modify the [finetune_data_config.yaml](llama_adapter_v2_multimodal7b/finetune_data_config.yaml#L2) to specify the datasets for fine-tuning. 
The format of datasets refers to [test_v2.json](test_v2.json). 

The pre-trained checkpoint can be downloaded in [ckpts](https://github.com/OpenGVLab/LLaMA-Adapter/releases/tag/v.2.0.0).

First, prepare the [nuscenes](https://www.nuscenes.org/) dataset which can refer to [BEVFormer](https://github.com/fundamentalvision/BEVFormer/blob/master/docs/prepare_dataset.md). 

We also provide the sampled nuscenes images under challenge/data. If you just want to run through the sampled data. Please follow the instructions below.

```bash
# The following script assumes that you prepare the nuscenes under ./challenge/llama_adapter_v2_multimodal7b
mkdir -p data/nuscenes
mv ../data/samples data/nuscenes
```

```bash
data/nuscenes                                    
├── samples
│   ├── CAM_FRONT_LEFT                          
│   │   ├── n015-2018-11-21-19-58-31+0800__CAM_FRONT_LEFT__1542801707504844.jpg 
│   │   ├── n015-2018-11-21-19-58-31+0800__CAM_FRONT_LEFT__1542801708004844.jpg
```

Then link the nuscenes dataset under the folder llama_adapter_v2_multimodal7b/data/. 
```bash
# The following script assumes that you prepare the nuscenes under ./challenge
ln -s nuscenes llama_adapter_v2_multimodal7b/data
```

Then we can train baseline as follows. 
```bash
# /path/to/llama_model_weights, /path/to/pre-trained/checkpoint.pth and /output/path need to be modified by your path
# make sure you are under ./challenge/llama_adapter_v2_multimodal7b
.exps/finetune.sh \
/path/to/llama_model_weights /path/to/pre-trained/checkpoint.pth \
finetune_data_config.yaml /output/path
```

### Inference baseline

```bash
# /path/to/llama_model_weights and /path/to/pre-trained/checkpoint.pth need to be modified by your path
# make sure you are under ./challenge/llama_adapter_v2_multimodal7b
python demo.py --llama_dir /path/to/llama_model_weights --checkpoint /path/to/pre-trained/checkpoint.pth --data ../test_v2.json  --output ../llama-adapter-DriveLM.json
```
Then we will get the [llama-adapter-DriveLM.json](llama-adapter-DriveLM.json), which are the predicted answers used for evaluation purposes.


## How to Eval

We implement diverse evaluation metrics tailored to different question types as mentioned [above](https://github.com/OpenDriveLab/DriveLM-private/blob/test/challenge/README.md?plain=1#L19).

### Setup
Install the language-evaluation package

Following [https://github.com/bckim92/language-evaluation](https://github.com/bckim92/language-evaluation) (skip the FIRST STEP if related libraries are already installed)

```bash
# FIRST STEP
# Oracle Java
sudo add-apt-repository ppa:webupd8team/java
sudo apt upadte
apt-get install oracle-java8-installer

# libxml-parser-perl
sudo apt install libxml-parser-perl
```
Then run:
```bash
# SECOND STEP
pip install git+https://github.com/bckim92/language-evaluation.git
python -c "import language_evaluation; language_evaluation.download('coco')"
```

### Evaluation
**The number and the content of the questions are subject to change in later version, but the question types are limited and provided.**

We have implemented three types of evaluation methods: Accuracy, ChatGPT Score, Language Evaluation and Match Score. The [final score](evaluation.py#L157) is the weighted average of four metrics.

The inputs required for evaluation are [llama-adapter-DriveLM.json](llama-adapter-DriveLM.json) and [test_v1.json](test_v1.json).

1. Replace [root_path1](evaluation.py#L97) with the path of your models' output. The example of models' output can be found in [output](llama-adapter-DriveLM.json).
2. Replace [root_path2](evaluation.py#L101) with the path of test_v1.json. The example of test_v1.json can be found in [test_v1.json](test_v1.json)
3. Replace [API-KEY](chatgpt.py#L17) with your own chatGPT api key.

```bash
# The following script assumes that you prepare the llama-adapter-DriveLM.json and test_v1.json under ./challenge
# make sure you are under ./challenge
python evaluation.py --root_path1 ./llama-adapter-DriveLM.json --root_path2 ./test_v1.json
```

### Results
The zero-shot results of baseline on the sampled data are as follows:
```
accuracy:  0.0
chatgpt:  78.5
match:  23.75
language score:  {'val/Bleu_1': 0.029183757177883535, 'val/Bleu_2': 0.00017003737042789148, 'val/Bleu_3': 3.066026234534233e-05, 'val/Bleu_4': 1.3024512157157705e-05, 'val/ROUGE_L': 0.05928706665796174, 'val/CIDEr': 0.05818698178494484}
final score:  0.36633034231114403
```


