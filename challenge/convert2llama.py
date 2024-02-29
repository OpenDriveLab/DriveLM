import numpy as np
import json


def convert2llama(root, dst):
    with open(root, 'r') as f:
        test_file = json.load(f)

    output = []
    for scene_id in test_file.keys():
        scene_data = test_file[scene_id]['key_frames']

        for frame_id in scene_data.keys():
            image_paths = scene_data[frame_id]['image_paths']
            image_paths = [image_paths[key].replace("..", "data") for key in image_paths.keys()]

            frame_data_qa = scene_data[frame_id]['QA']
            QA_pairs = frame_data_qa["perception"] + frame_data_qa["prediction"] + frame_data_qa["planning"] + frame_data_qa["behavior"]
            
            for idx, qa in enumerate(QA_pairs):
                question = qa['Q']
                answer = qa['A']
                output.append(
                    {
                        "id": scene_id + "_" + frame_id + "_" + str(idx),
                        "image": image_paths,
                        "conversations": [
                            {
                                "from": "human",
                                "value": "<image>\n" + question
                            },
                            {
                                "from": "gpt",
                                "value": answer
                            },
                        ]
                    }
                )

    with open(dst, 'w') as f:
        json.dump(output, f, indent=4)


if __name__ == '__main__':
    root = "test_v1.json"
    dst = "test_v2.json"
    convert2llama(root, dst)
