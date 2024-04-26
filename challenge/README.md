### TL;DR
* The purpose of this folder is to facilitate our CVPR 2024 challenge. Initially, we will use a small subset of training data (**demo train** in the following text) as an illustrative example, demonstrating how to obtain the **test data format** and the submission format, how to train the baseline and infer the baseline, and go through the evaluation pipeline. 
  
* Thrilled to announce that the test server is online and the test questions are release! Check `DriveLM-nuScenes version-1.1 val` and [How-to-submit](#submit-to-test-server).

* For the purpose of the new **test data format**, it is essential that our primary intention is to create a specific test data format preventing possible cheating. 

<!-- > * Subsequently, we will demonstrate the process of conducting evaluations, encompassing the baseline methodology. -->

* For better illustration, we provide [google slides](https://docs.google.com/presentation/d/1bicxoR_L3t05p5xw-qZM0Dj5KdJhjynqLM0Rck0qdcI/edit?usp=sharing) for your reference. 

* **Official announcement about the DriveLM challenge is maintained in this folder**. Please raise an issue in the repo if you find anything unclear.

## How to Prepare Data

### DriveLM
We provide three options for you to prepare the dataset:
1. If you just want to run through the demo. 
We provide the demo train DriveLM data [train_sample.json](data/train_sample.json) and the [sampled images](llama_adapter_v2_multimodal7b/data/nuscenes) in the repo.

2. If you already have nuscenes dataset similar with the [bevformer](https://github.com/fundamentalvision/BEVFormer/blob/master/docs/prepare_dataset.md). Then you just need to
```bash
rm -rf llama_adapter_v2_multimodal7b/data/nuscenes
ln -s /path/to/your/nuscenes llama_adapter_v2_multimodal7b/data/
```

3. If you do not have the nuscenes dataset, but you want to run through the whole DriveLM dataset. Then you need to Download the following dataset.

| nuScenes subset images | DriveLM-nuScenes version-1.1| DriveLM-nuScenes version-1.1 val |
|:-------:|:-------:|:------:|
| [Google Drive Train](https://drive.google.com/file/d/1DeosPGYeM2gXSChjMODGsQChZyYDmaUz/view?usp=sharing)  & [Google Drive Val](https://drive.google.com/file/d/18f8ygNxGZWat-crUjroYuQbd39Sk9xCo/view?usp=sharing) | [Google Drive](https://drive.google.com/file/d/1CvTPwChKvfnvrZ1Wr0ZNVqtibkkNeGgt/view?usp=sharing) | [Google Drive](https://drive.google.com/file/d/1fsVP7jOpvChcpoXVdypaZ4HREX1gA7As/view?usp=sharing) |
|[Baidu Netdisk Train](https://pan.baidu.com/s/11xvxPzUY5xTIsJQrYFogqg?pwd=mk95) & [Baidu Netdisk Val](https://pan.baidu.com/s/1mUbHmtc4c5WhUQACW8_Blw?pwd=54vz) |[Baidu Netdisk](https://pan.baidu.com/s/1Vojg73jviguki0yvAB6nUg?pwd=73s8) | [Baidu Netdisk](https://pan.baidu.com/s/1p01Szh5QTtSAzSXdhLCTwQ?pwd=h9hi) |
|[HuggingFace Train](https://huggingface.co/datasets/OpenDriveLab/DriveLM/blob/main/drivelm_nus_imgs_train.zip) & [HuggingFace Val](https://huggingface.co/datasets/OpenDriveLab/DriveLM/blob/main/drivelm_nus_imgs_val.zip) |[HuggingFace](https://huggingface.co/datasets/OpenDriveLab/DriveLM/blob/main/v1_1_train_nus.json) |[HuggingFace](https://huggingface.co/datasets/OpenDriveLab/DriveLM/blob/main/v1_1_val_nus_q_only.json) |

Please follow the instructions below.
```bash
# The following script assumes that you prepare the nuscenes under ./challenge/llama_adapter_v2_multimodal7b
mv ../drivelm_nus_imgs_train.zip .
unzip drivelm_nus_imgs_train.zip
mv nuscenes data
```
Then the format of the data will be the same as the following.
```bash
data/nuscenes                                    
├── samples
│   ├── CAM_FRONT_LEFT                          
│   │   ├── n015-2018-11-21-19-58-31+0800__CAM_FRONT_LEFT__1542801707504844.jpg 
│   │   ├── n015-2018-11-21-19-58-31+0800__CAM_FRONT_LEFT__1542801708004844.jpg
```


Follow the steps below to get the test data format as well as data for the baseline model.

### Extract Data

Extract fundamental question-and-answer (QA) pairs from the training dataset. 

**Note that** the number and the content of the fundamental QA pairs might change in the test server, but we ensure that **all the question types are limited in our provided test data format**. That being said, the question types are within 1) multi-choice question; 2) conversation question; 3) yes/no question;

```bash
# The following script assumes that you download the train data json under ./challenge/data
# make sure you are under ./challenge
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
Then we will get the test_eval.json in challenge folder. The example of test_eval.json can be found in [test_eval.json](test_eval.json)

We use llama-adapter v2 as our baseline. If you want to convert data into llama-adapter format:
```bash
# The following script assumes that you prepare the test_eval.json under ./challenge
# make sure you are under ./challenge
python convert2llama.py
```
Then we will get the test_llama.json in challenge folder. The example of test_llama.json can be found in [test_llama.json](test_llama.json)

[test_eval.json](test_eval.json) is used for evaluation. [test_llama.json](test_llama.json) is used for training and inference of the baseline.

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

The cost related to finetuning and inference can be found below.

| Process   | Data         | Data Quantity   | Frame | Setting        | VRAM Requirement | Training Time per Epoch | Inference Time for Data |
| --------- | ------------ | --------------- | ----- | -------------- | ---------------- | ----------------------- | ----------------------- |
| Finetune  | Training Set | 29,448 QA pairs | 4072  | Batch size = 4 | 34G              | 10 minutes              | -                       |
| Inference | Training Set | 29,448 QA pairs | 4072  | Batch size = 8 | 35G              | -                       | Approximately 2 hours   |



### Train baseline
Here, we offer examples for fine-tuning the model. If you are interested in pretraining the model, you can find detailed information in the [llama-adapter](https://github.com/OpenGVLab/LLaMA-Adapter) repository. You should modify the [finetune_data_config.yaml](llama_adapter_v2_multimodal7b/finetune_data_config.yaml#L2) to specify the datasets for fine-tuning. 
The format of datasets refers to [test_llama.json](test_llama.json). 

The pre-trained checkpoint can be downloaded in [ckpts](https://github.com/OpenGVLab/LLaMA-Adapter/releases/tag/v.2.0.0). You can choose any one of them.

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
# num_processes is the number of the gpu you will use to infer the data.
# make sure you are under ./challenge/llama_adapter_v2_multimodal7b
python demo.py --llama_dir /path/to/llama_model_weights --checkpoint /path/to/pre-trained/checkpoint.pth --data ../test_llama.json  --output ../output.json --batch_size 4 --num_processes 8
```
Then we will get the [output.json](output.json), which are the predicted answers used for evaluation purposes.


## How to Eval

We implement diverse evaluation metrics tailored to different question types as mentioned [above](https://github.com/OpenDriveLab/DriveLM-private/blob/test/challenge/README.md?plain=1#L19).

### Setup
Install the language-evaluation package.

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

The inputs required for evaluation are [output.json](output.json) and [test_eval.json](test_eval.json).

1. Replace [root_path1](evaluation.py#L97) with the path of your models' output. The example of models' output can be found in [output](output.json).
2. Replace [root_path2](evaluation.py#L101) with the path of test_eval.json. The example of test_eval.json can be found in [test_eval.json](test_eval.json)
3. Replace [API-KEY](gpt_eval.py#L17) with your own chatGPT api key.

```bash
# The following script assumes that you prepare the output.json and test_eval.json under ./challenge
# make sure you are under ./challenge
python evaluation.py --root_path1 ./output.json --root_path2 ./test_eval.json
```

### Results
The zero-shot results of baseline on the sampled data are as follows:
```
"accuracy":  0.0
"chatgpt":  65.11111111111111
"match":  28.25
"language score":  {
  'val/Bleu_1': 0.0495223110147729, 
  'val/Bleu_2': 0.00021977465683011536, 
  'val/Bleu_3': 3.6312541763196866e-05, 
  'val/Bleu_4': 1.4776149283286042e-05, 
  'val/ROUGE_L': 0.08383567940883102, 
  'val/CIDEr': 0.09901486412073952
}
"final_score":  0.3240234750718823
```

The zero-shot results of baseline on the test data are as follows:
```
"accuracy": 0.0
"chatgpt": 67.7535896248263, 
"match": 18.83
"language score": {
  "test/Bleu_1": 0.2382764794460423,
  "test/Bleu_2": 0.09954243471154352,
  "test/Bleu_3": 0.03670697545241351,
  "test/Bleu_4": 0.011298629095627342,
  "test/ROUGE_L": 0.1992858115225957,
  "test/CIDEr": 0.0074352082312374385
}
"final_score": 0.32843094354141145
```

## Submit to Test Server

### Submission Instruction

The competition server is held on [Hugging Face space](https://huggingface.co/spaces/AGC2024/driving-with-language-2024). Our test server is now open for submission!


Please infer your model on the `DriveLM-nuScenes version-1.1 val`(this is our test question and we will **NOT** release their GT answer) and get your output as `output.json`. You need to evaluate `output.json` locally first before submitting to test server!

1. Prepare your result

    Open [prepare_submission.py](prepare_submission.py) and fill in the following information starting line 4:
    ```
    method = ""  # <str> -- name of the method
    team = ""  # <str> -- name of the team, !!!identical to the Google Form!!!
    authors = [""]  # <list> -- list of str, authors
    email = ""  # <str> -- e-mail address
    institution = ""  # <str> -- institution or company
    country = ""  # <str> -- country or region
    ```
    While other fields can change between different submissions, make sure you <font color=red> always use your team name submitted on Google registration form for the `team` field, NOT the anonymous team name to be shown on the leaderboard</font>.
    Then run this file:
    ```bash
    # make sure you are under ./challenge
    python prepare_submission.py
    ```
    This will generate `submission.json` with your information and result. An [example](submission.json) is given in this folder. 

2. Upload your result as **a Hugging Face model**

    Click your profile picture on the top right of the Hugging Face website, and select `+ New Model`. Create a new model repository, and upload the `submission.json` file.
    
    Note that private models are also acceptable by the competition space.

3. Submit your result and evaluate on test server

    Go to [competition space](https://huggingface.co/spaces/AGC2024/driving-with-language-2024), click `New Submission` on the left panel. Paste the link of the Hugging Face model you created under `Hub model`. Then click `Submit`. 

    <font color=red> Note: you can make up to 3 submissions per day. </font>

### How to View My Submissions?

You can check the status of your submissions in the `My submissions` tab of the competition space.

Please refer to [these slides](https://docs.google.com/presentation/d/1bicxoR_L3t05p5xw-qZM0Dj5KdJhjynqLM0Rck0qdcI/edit?usp=sharing) for explaination of each score.

You can select a submission and click `Update Selected Submissions` on the bottom to update its evaluation status to the private leaderboard. Please note that <font color=red>public score and private score are exactly the same</font> in our case. So please ignore the descriptions in `My Submissions` tab. 

## FAQ

### The `New Submission` page shows `Invalid Token` when I click `Submit`, what should I do?

This means you are no longer logged in to the current competition space, or the space has automatically logged you out due to inactivity (more than a day). 

Please refresh the page, click `Login with Hugging Face` at the bottom of the left panel, and resubmit.

### Can I Submit Without Making My Submission Public?

Of course. The competition space accepts Hugging Face private models. in fact, we recommend participants to submit as private models to keep their submissions private.

### Will My Evaluation Status Be Visible to Others?

The public leaderboard will be open with the best results of all teams about a week before the competition ends.

**Note that** you can change your team name even after the competition ends. Thus, if you want to stay anonymous on the public leaderboard, you can first use a temporary team name and change it to your real team name after the competition ends.

### My evaluation status shows `Failed`, how can I get the error message?

First, make sure your submission is in the correct format as in [submission preparation](#submission-preparation) and you upload the correct Hugging Face **model** link (in the format of `Username/model`) in `New Submission`.

If you confirm that the submission format is correct, please contact the challenge host [Chonghao Sima](mailto:simachonghao@pjlab.org.cn) via email. Please include the **Submission ID** of the corresponding submission in the email. The Submission ID can be found in the `My Submissions` tab.

```
Email Subject:
[CVPR DRIVELM] Failed submission - {Submission ID}
Body:
  Your Name: {}
  Team Name: {}
  Institution / Company: {}
  Email: {}
```

### I could not visit `My Submissions` page, what should I do?

Chances are that you are not logged in to the current competition space. 

Please refresh the page, click `Login with Hugging Face` at the bottom of the left panel.

### If I encounter a reshape error, what should I do?

You should first refer to this [location](https://github.com/OpenDriveLab/DriveLM/blob/main/challenge/evaluation.py#L90). Most of the reshape errors occur here.

### Finally, which dataset do we submit to the competition?

Please refrain from using demo data. Instead, utilize the [validation data](https://drive.google.com/file/d/1fsVP7jOpvChcpoXVdypaZ4HREX1gA7As/view?usp=sharing)  for inference and submission to the evaluation server.


## Citation
Please consider citing our project and staring our [huggingface repo](https://huggingface.co/spaces/AGC2024/driving-with-language-2024) if they help your competition and research.
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
