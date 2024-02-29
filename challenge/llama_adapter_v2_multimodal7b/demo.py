import cv2
import llama
import torch
from PIL import Image
from tqdm import tqdm
import json
import argparse
import torchvision.transforms as transforms
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# add args
parser = argparse.ArgumentParser(description='LLAMA Adapter')
parser.add_argument('--llama_dir', type=str, default="/path/to/llama_model_weights", help='path to llama model weights')
parser.add_argument('--checkpoint', type=str, default="/path/to/pre-trained/checkpoint.pth", help='path to pre-trained checkpoint')
parser.add_argument('--data', type=str, default="../test_v2.json", help='path to test data')
parser.add_argument('--output', type=str, default="../llama-adapter-DriveLM.json", help='path to output file')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
llama_dir = args.llama_dir

# choose from BIAS-7B, LORA-BIAS-7B, CAPTION-7B.pth
model, preprocess = llama.load(args.checkpoint, llama_dir, llama_type="7B", device=device)
model.eval()


data_dict = []

transform_train = transforms.Compose([
    transforms.Resize(
                    (224, 224), interpolation=InterpolationMode.BICUBIC
                ), # 3 is bicubic
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])


with open(args.data, 'r') as f:
    data_all = json.load(f)


for data_item in tqdm(data_all):
    filename = data_item['image']
    ids = data_item['id']
    question = data_item['conversations'][0]['value']
    answer = data_item['conversations'][1]['value']
    
    prompt = llama.format_prompt(question)

    if isinstance(filename, list):
        image_all = []
        for img_path in filename:
            image = cv2.imread(img_path)
            image = Image.fromarray(image)
            image = transform_train(image).unsqueeze(0).to(device)
            image_all.append(image)
        image = torch.stack(image_all, dim=1)
    else:
        image = cv2.imread(filename)
        image = Image.fromarray(image)
        image = transform_train(image).unsqueeze(0).to(device)

    result = model.generate(image, [prompt], temperature=0.2, top_p=0.1)[0]
    print(result)

    data_dict.append({'id': ids, 'question':question, 'gt_answer':answer, 'answer':result})

with open(args.output, "w") as f:
    json.dump(data_dict, f, indent=4)  
