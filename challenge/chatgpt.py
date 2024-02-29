from tenacity import (
    retry,
    stop_after_attempt,
    wait_incrementing,
)  
import openai
import pickle
import pdb
import numpy as np
import torch
import json
import argparse


class ChatGPT:
    def __init__(self):
        openai.api_key = "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"

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
    
    def forward(self, answer, GT):
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
    prediction = "Keep going at the same speed."
    GT = "Keep going at the same speed, decelerate gradually without braking."

    eval = ChatGPT()
    scores = eval.forward(prediction, GT)
    print(scores)