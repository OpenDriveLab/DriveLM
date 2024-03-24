import openai
import pickle
import pdb
import numpy as np
import torch
import json
import argparse
from multiprocessing import Pool


class GPTEvaluation:
    def __init__(self):
        openai.api_key = "you need to use your own openai key for evaluation on your local machine"

    def call_chatgpt(self, chatgpt_messages, max_tokens=40, model="gpt-3.5-turbo"):
        response = openai.ChatCompletion.create(
            model=model, messages=chatgpt_messages, temperature=0.6, max_tokens=max_tokens
        )
        reply = response["choices"][0]["message"]["content"]
        total_tokens = response["usage"]["total_tokens"]
        return reply, total_tokens
    
    def prepare_chatgpt_message(self, prompt):
        system_message = "an evaluator who rates my answer based on the correct answer"
        messages = [{"role": "system", "content": system_message}]
        messages.append({"role": "user", "content": "{}".format(prompt)})
        
        return messages
    
    def forward(self, data):
        answer, GT = data
        prompts = "Rate my answer based on the correct answer out of 100, with higher scores indicating that the answer is closer to the correct answer, and you should be accurate to single digits like 62, 78, 41,etc. Output the number only"
        prompts = prompts + "This is the correct answer: " + GT + "This is my answer: " + answer
        
        output = ""
        messages = self.prepare_chatgpt_message(prompts)
        reply, total_tokens = self.call_chatgpt(messages, max_tokens=3000)

        output += reply
        output += "\n\n"

        output = output[:-2]

        return output
    

if __name__ == "__main__":
    data = [
        ("The ego vehicle should notice the bus next, as it is the third object in the image. The bus is stopped at the intersection, and the ego vehicle should be cautious when approaching the intersection to ensure it does not collide with the bus.", "Firstly, notice <c3,CAM_FRONT_LEFT,1075.5,382.8>. The object is a traffic sign, so the ego vehicle should continue at the same speed. Secondly, notice <c2,CAM_FRONT,836.3,398.3>. The object is a traffic sign, so the ego vehicle should accelerate and continue ahead. Thirdly, notice <c1,CAM_BACK,991.7,603.0>. The object is stationary, so the ego vehicle should continue ahead at the same speed."),
        # Add more data here
    ]

    eval = GPTEvaluation()

    with Pool(5) as p:  # Change the number based on your CPU cores
        scores = p.map(eval.forward, data)

    print(scores)