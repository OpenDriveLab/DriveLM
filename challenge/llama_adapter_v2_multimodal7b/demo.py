import cv2
import llama
import torch
from PIL import Image
from tqdm import tqdm
import json
import argparse
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.multiprocessing import Process, Queue, set_start_method
import math

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

class LLamaDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
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
                if self.transform:
                    image = self.transform(image)
                image_all.append(image)
            image = torch.stack(image_all, dim=0)
        else:
            image = cv2.imread(filename)
            image = Image.fromarray(image)
            if self.transform:
                image = self.transform(image)

        return image, prompt, ids, question, answer

def worker(rank, gpu_id, args, queue):
    torch.cuda.set_device(gpu_id)
    
    device = torch.device("cuda")
    llama_dir = args.llama_dir

    # Choose from BIAS-7B, LORA-BIAS-7B, CAPTION-7B.pth
    model, preprocess = llama.load(args.checkpoint, llama_dir, llama_type="7B", device=device)
    model.eval()

    transform_train = transforms.Compose([
        transforms.Resize((224, 224), interpolation=BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

    with open(args.data, 'r') as f:
        data_all = json.load(f)

    num_processes = args.num_processes
    data_per_process = math.ceil(len(data_all) / num_processes)
    start_idx = rank * data_per_process
    end_idx = min((rank + 1) * data_per_process, len(data_all))
    data_to_process = data_all[start_idx:end_idx]

    dataset = LLamaDataset(data_to_process, transform=transform_train)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    data_dict = []

    for batch in tqdm(dataloader):
        images, prompts, ids, questions, gt_answers = batch
        images = images.to(device)
        results = model.generate(images, prompts, temperature=0.2, top_p=0.1)
        
        for i, result in enumerate(results):
            data_dict.append({'id': ids[i], 'question': questions[i], 'gt_answer': gt_answers[i], 'answer': result})
    
    queue.put(data_dict)

# add args
parser = argparse.ArgumentParser(description='LLAMA Adapter')
parser.add_argument('--llama_dir', type=str, default="/path/to/llama_model_weights", help='path to llama model weights')
parser.add_argument('--checkpoint', type=str, default="/path/to/pre-trained/checkpoint.pth", help='path to pre-trained checkpoint')
parser.add_argument('--data', type=str, default="../test_llama.json", help='path to test data')
parser.add_argument('--output', type=str, default="../output.json", help='path to output file')
parser.add_argument('--batch_size', type=int, default=8, help='batch size for parallel processing')
parser.add_argument('--num_processes', type=int, default=8, help='number of processes to use')
args = parser.parse_args()

if __name__ == '__main__':
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs")
    
    queue = Queue()
    processes = []
    for rank in range(num_gpus):
        p = Process(target=worker, args=(rank, rank, args, queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    data_dict = []
    while not queue.empty():
        data_dict.extend(queue.get())

    with open(args.output, "w") as f:
        json.dump(data_dict, f, indent=4)
