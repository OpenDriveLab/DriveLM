import numpy as np
import json
import random


def rule_based1(question, answer):
    rule = ["Going ahead.", "Turn right.", "Turn left.", "Stopped.", "Back up.", "Reverse parking.", "Drive backward."]
    rule.remove(answer)
    choices = random.sample(rule, 3)
    choices.append(answer)
    random.shuffle(choices)
    idx = choices.index(answer)
    question += f" Please select the correct answer from the following options: A. {choices[0]} B. {choices[1]} C. {choices[2]} D. {choices[3]}"
    mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
    return {"Q": question, "A": mapping[idx]}

def rule_based2(question, answer):
    rule = ['The ego vehicle is slightly steering to the left. The ego vehicle is driving very fast.', 'The ego vehicle is steering to the left. The ego vehicle is driving with normal speed.', 'The ego vehicle is steering to the left. The ego vehicle is driving fast.', 'The ego vehicle is slightly steering to the right. The ego vehicle is driving fast.', 'The ego vehicle is going straight. The ego vehicle is driving slowly.', 'The ego vehicle is going straight. The ego vehicle is driving with normal speed.', 'The ego vehicle is slightly steering to the left. The ego vehicle is driving with normal speed.', 'The ego vehicle is slightly steering to the left. The ego vehicle is driving slowly.', 'The ego vehicle is slightly steering to the right. The ego vehicle is driving slowly.', 'The ego vehicle is slightly steering to the right. The ego vehicle is driving very fast.', 'The ego vehicle is steering to the right. The ego vehicle is driving fast.', 'The ego vehicle is steering to the right. The ego vehicle is driving very fast.', 'The ego vehicle is slightly steering to the left. The ego vehicle is driving fast.', 'The ego vehicle is steering to the left. The ego vehicle is driving very fast.', 'The ego vehicle is going straight. The ego vehicle is not moving.', 'The ego vehicle is slightly steering to the right. The ego vehicle is driving with normal speed.', 'The ego vehicle is steering to the right. The ego vehicle is driving slowly.', 'The ego vehicle is steering to the right. The ego vehicle is driving with normal speed.', 'The ego vehicle is going straight. The ego vehicle is driving very fast.', 'The ego vehicle is going straight. The ego vehicle is driving fast.', 'The ego vehicle is steering to the left. The ego vehicle is driving slowly.']
    rule.remove(answer)
    choices = random.sample(rule, 3)
    choices.append(answer)
    random.shuffle(choices)
    idx = choices.index(answer)
    question += f" Please select the correct answer from the following options: A. {choices[0]} B. {choices[1]} C. {choices[2]} D. {choices[3]}"
    mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
    return {"Q": question, "A": mapping[idx]}
    

def loop_test(root, dst):
    with open(root, 'r') as f:
        test_file = json.load(f)

    for scene_id in test_file.keys():
        scene_data = test_file[scene_id]['key_frames']

        for frame_id in scene_data.keys():
            # frame_data_infos = scene_data[frame_id]['key_object_infos']
            frame_data_qa = scene_data[frame_id]['QA']
            image_paths = scene_data[frame_id]['image_paths']

            test_file[scene_id]['key_frames'][frame_id] = dict()
            # test_file[scene_id]['key_frames'][frame_id]['key_object_infos'] = frame_data_infos
            test_file[scene_id]['key_frames'][frame_id]['QA'] = dict()
            test_file[scene_id]['key_frames'][frame_id]['QA']['perception'] = []
            # add all prediction and planning
            test_file[scene_id]['key_frames'][frame_id]['QA']['prediction'] = frame_data_qa["prediction"]
            test_file[scene_id]['key_frames'][frame_id]['QA']['planning'] = frame_data_qa["planning"]

            test_file[scene_id]['key_frames'][frame_id]['QA']['behavior'] = []
            test_file[scene_id]['key_frames'][frame_id]['image_paths'] = image_paths

            for qa in frame_data_qa["perception"]:
                question = qa['Q']
                answer = qa['A']
                if "What is the moving status of object".lower() in question.lower():
                    qa.update(rule_based1(question, answer))
                    test_file[scene_id]['key_frames'][frame_id]['QA']['perception'].append(qa)
                else:
                    test_file[scene_id]['key_frames'][frame_id]['QA']['perception'].append(qa)

            for qa in frame_data_qa["behavior"]:
                question = qa['Q']
                answer = qa['A']
                qa.update(rule_based2(question, answer))
                test_file[scene_id]['key_frames'][frame_id]['QA']['behavior'].append(qa)

    with open(dst, 'w') as f:
        json.dump(test_file, f, indent=4)



if __name__ == '__main__':
    root = "test.json"
    dst = "test_eval.json"
    loop_test(root, dst)
