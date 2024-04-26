import re
import argparse
import json
import numpy as np
import torch.nn as nn
import language_evaluation
from multiprocessing import Pool

import sys
sys.path.append(".")
from gpt_eval import GPTEvaluation


class evaluation_suit():
    def __init__(self):
        self.language_eval = language_evaluation.CocoEvaluator(coco_types=["BLEU", "ROUGE_L", "CIDEr"])
        self.chatgpt_eval = GPTEvaluation()
        self.GPT = []
        self.accuracy = {"answer": [], "GT": []}
        self.language = {"answer": [], "GT": []}
        self.match = {"match": {"answer": [], "GT": []}, "GPT": []}

    def eval_acc(self):
        scores = []
        for i in range(len(self.accuracy["answer"])):
            answer = self.accuracy["answer"][i]
            GT = self.accuracy["GT"][i]
            if answer == GT:
                scores.append(1.0)
            else:
                scores.append(0.0)

        scores = sum(scores) / len(scores)
        return scores

    def eval_chatGPT(self, data):
        with Pool(32) as p:  # Change the number based on your CPU cores
            scores = p.map(self.chatgpt_eval.forward, data)

        scores = list(map(float, scores))
        scores = sum(scores) / len(scores)
        return scores

    def eval_language(self):
        """
        return the dict evaluation results
        """
        answer = self.language["answer"]
        GT = self.language["GT"]
        results_gen = self.language_eval.run_evaluation(answer, GT)
        results_gen_dict = {
            f"val/{k}": v for k, v in results_gen.items()
        }
        return results_gen_dict

    def eval_match(self):
        outs1 = []
        for i in range(len(self.match["match"]["answer"])):
            answer = self.match["match"]["answer"][i]
            GT = self.match["match"]["GT"][i]
            _, F1_score = self.match_result(answer, GT)
            outs1.append(F1_score * 100)
        
        outs1 = sum(outs1) / len(outs1)
        outs2 = self.eval_chatGPT(self.match["GPT"])
        scores = (outs1 + outs2) / 2.0
        return scores

    def eval_graph(self, question):
        # check if answer in self.graph  
        question_nums = re.findall(r'\d+\.\d+', question)
        question_nums = np.array([list(map(float, x.split()))[0] for x in question_nums]).reshape(-1, 2)
        question_nums = [list(i) for i in question_nums]
        for q in question_nums:
            if q not in self.graph:
                return False
        return True

    def match_result(self, answer, GT):
        """
        answer: [[1.,2.], [2., 3.]]
        GT: [[1., 2.], [2., 3.]]
        """
        answer_nums = re.findall(r'\d+\.\d+', answer)
        GT_nums = re.findall(r'\d+\.\d+', GT)
        # transform string into float
        answer_nums = np.array([list(map(float, x.split()))[0] for x in answer_nums]).reshape(-1, 2)
        GT_nums = np.array([list(map(float, x.split()))[0] for x in GT_nums]).reshape(-1, 2)
        length = len(GT_nums)

        matched_out = []
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        for pred in answer_nums:
            closest_distance = float('inf')
            closest_gt = None
            closest_id = None
            for i, gt in enumerate(GT_nums):
                distance = np.sum(np.abs(pred - gt))
                if distance < closest_distance:
                    closest_distance = distance
                    closest_gt = gt
                    closest_id = i

            if closest_distance < 16:
                true_positives += 1
                matched_out.append(closest_gt)  
                GT_nums = np.delete(GT_nums, closest_id, axis=0) 
            else:
                false_positives += 1
            
        false_negatives = length - true_positives
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        F1 = 2 * precision * recall / (precision + recall + 1e-8)

        return matched_out, F1

    def set_graph(self, answer, GT):
        self.graph, _ = self.match_result(answer, GT)
        self.graph = [list(i) for i in self.graph]

    def forward(self, tag, answer, GT):
        if 0 in tag:
            self.accuracy["answer"].append(answer)
            self.accuracy["GT"].append(GT)
        if 1 in tag:
            self.GPT.append((answer, GT))
        if 2 in tag:
            self.language["GT"].append(GT)
            self.language["answer"].append(answer)
        if 3 in tag:
            self.match["match"]["GT"].append(GT)
            self.match["match"]["answer"].append(answer)
            self.match["GPT"].append((answer, GT))

            
    def evaluation(self):
        print("evaluation start!")
        scores = {}
        scores["accuracy"] = self.eval_acc()
        scores["chatgpt"] = self.eval_chatGPT(self.GPT)
        scores["language"] = self.eval_language()
        scores["match"] = self.eval_match()

        return scores

if __name__ == '__main__':
    # get args
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--root_path1', type=str, default="./llama-adapter-DriveLM.json", help='path to prediction file')
    parser.add_argument('--root_path2', type=str, default="./test_v1.json", help='path to test file')
    args = parser.parse_args()
    
    with open(args.root_path1, 'r') as f :#, \    
        pred_file = json.load(f)
    pred_file = {pred_file[i]["id"]: pred_file[i] for i in range(len(pred_file))}
    
    with open(args.root_path2, 'r') as f:
        test_file = json.load(f)

    evaluation = evaluation_suit()
    for scene_id in test_file.keys():
        scene_data = test_file[scene_id]['key_frames']

        for frame_id in scene_data.keys():
            frame_data_qa = scene_data[frame_id]['QA']
            first_flag = True

            for i, qa in enumerate(frame_data_qa["perception"] + frame_data_qa["prediction"] + frame_data_qa["planning"] + frame_data_qa["behavior"]):
                question = qa['Q']
                GT = qa['A']
                tag = qa['tag']
                idx = scene_id + "_" + frame_id + "_" + str(i)
                predict = pred_file[idx]["answer"]
                # assert pred_file[idx]["gt_answer"] == GT, print(pred_file[idx]["gt_answer"], GT)
                if first_flag:
                    first_flag = False
                    evaluation.set_graph(predict, GT)
                    evaluation.forward(tag, predict, GT)
                else:
                    if evaluation.eval_graph(question):
                        res = evaluation.forward(tag, predict, GT)

    output = evaluation.evaluation()
    print("accuracy score: ", output["accuracy"])
    print("chatgpt score: ", output["chatgpt"])
    print("match score: ", output["match"])
    print("language score: ", output["language"])
    
    # Normalize to 0-1 and combine the scores: chatgpt, language, match, accuracy
    scores = []
    weights = [0.4, 0.2, 0.2, 0.2]
    
    # chatGPT
    score = output["chatgpt"] / 100.
    scores.append(score)

    # language
    score = 0
    for idx, key in enumerate(output["language"].keys()):
        if idx < 4:
            score += output["language"][key] / 4. / 3.
        elif idx == 4:
            score += output["language"][key] / 3. 
        else:
            score += output["language"][key] / 10. / 3.

    scores.append(score)
    
    # match
    score = output["match"] / 100.
    scores.append(score)

    # accuracy
    score = output["accuracy"]
    scores.append(score)

    final_score = sum([x * y for x, y in zip(scores, weights)])
    print("final score: ", final_score)
    

