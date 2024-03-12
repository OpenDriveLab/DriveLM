import re
import argparse
import json
import numpy as np
import torch.nn as nn
import language_evaluation

import sys
sys.path.append(".")
from chatgpt import ChatGPT


class evaluation_suit():
    def __init__(self):
        self.language_eval = language_evaluation.CocoEvaluator(coco_types=["BLEU", "ROUGE_L", "CIDEr"])
        self.chatgpt_eval = ChatGPT()
        self.GTs = []
        self.answers = []

    def eval_acc(self, answer, GT):
        if answer == GT:
            return 1
        else:
            return 0

    def eval_chatGPT(self, answer, GT):
        scores = self.chatgpt_eval.forward(answer, GT)
        scores = float(scores)
        return scores

    def eval_language(self):
        """
        return the dict evaluation results
        """
        results_gen = self.language_eval.run_evaluation(self.answers, self.GTs)
        results_gen_dict = {
            f"val/{k}": v for k, v in results_gen.items()
        }
        return results_gen_dict

    def eval_match(self, answer, GT):
        matched = self.match_result(answer, GT)
        GT_nums = re.findall(r'\d+\.\d+', GT)
        GT_nums = np.array([list(map(float, x.split()))[0] for x in GT_nums]).reshape(-1, 2)
        GT_nums = [list(i) for i in GT_nums]

        return len(matched) / len(GT_nums) * 100

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

        matched_out = []
        for ans in answer_nums:
            for gt in GT_nums:
                distance = np.sum(np.abs(ans - gt))
                if distance < 16:
                    matched_out.append(gt)
                    break

        return matched_out

    def set_graph(self, answer, GT):
        self.graph = self.match_result(answer, GT)
        self.graph = [list(i) for i in self.graph]

    def forward(self, tag, answer, GT):
        scores = {}
        if 0 in tag:
            scores["accuracy"] = self.eval_acc(answer, GT)
        if 1 in tag:
            scores["chatgpt"] = self.eval_chatGPT(answer, GT)
        if 2 in tag:
            self.GTs.append(GT)
            self.answers.append(answer)
        if 3 in tag:
            outs1 = self.eval_match(answer, GT)
            outs2 = self.eval_chatGPT(answer, GT)
            scores["match"] = (outs1 + outs2) / 2.0

        return scores

if __name__ == '__main__':
    # get args
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--root_path1', type=str, default="./output.json", help='path to prediction file')
    parser.add_argument('--root_path2', type=str, default="./test_eval.json", help='path to test file')
    args = parser.parse_args()
    
    with open(args.root_path1, 'r') as f :#, \    
        pred_file = json.load(f)
    pred_file = {pred_file[i]["id"]: pred_file[i] for i in range(len(pred_file))}
    
    with open(args.root_path2, 'r') as f:
        test_file = json.load(f)

    evaluation = evaluation_suit()
    output = {"accuracy": [], "chatgpt": [], "language": [], "match": []}
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
                assert pred_file[idx]["gt_answer"] == GT, print(pred_file[idx]["gt_answer"], GT)
                if first_flag:
                    first_flag = False
                    evaluation.set_graph(predict, GT)
                    res = evaluation.forward(tag, predict, GT)
                    for key in output.keys():
                        if key in res:
                            output[key].append(res[key])
                else:
                    if evaluation.eval_graph(question):
                        res = evaluation.forward(tag, predict, GT)
                        for key in output.keys():
                            if key in res:
                                output[key].append(res[key])
    
    output["language"] = evaluation.eval_language()
    if len(output["accuracy"]) != 0:
        output["accuracy"] = sum(output["accuracy"]) / len(output["accuracy"])
        print("accuracy: ", output["accuracy"])
    if len(output["chatgpt"]) != 0:
        output["chatgpt"] = sum(output["chatgpt"]) / len(output["chatgpt"])
        print("chatgpt: ", output["chatgpt"])
    if len(output["match"]) != 0:
        output["match"] = sum(output["match"]) / len(output["match"])
        print("match: ", output["match"])

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
    

