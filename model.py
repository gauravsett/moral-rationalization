import json
import openai
from openai import Completion
import re
from transformers import pipeline, set_seed
from transformers import GPT2TokenizerFast


def clean(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text.strip())
    return text


class Model(object):
    
    def __init__(self) -> None:
        raise NotImplementedError
    
    def generate(self, input: str) -> str:
        raise NotImplementedError
    
    def classify(self, input: str) -> int:
        raise NotImplementedError


class OpenAIModel(Model):
    
    def __init__(self, version: str) -> None:
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        with open("secrets.json", "r") as f:
            openai.api_key = json.load(f)["OPENAI_API_KEY"]
        self.version = version
        self.logit_bias = self.set_bias()
        
    def set_bias(self):
        logit_bias = dict()
        labels = ["moral", "immoral"]
        for label in labels:
            logit_bias[self.tokenizer.encode(label)[0]] = 100
            logit_bias[self.tokenizer.encode(" " + label)[0]] = 100
        return logit_bias
    
    def generate(self, input: str | list) -> str | list:
        if isinstance(input, list):
            return [self.generate(i) for i in input]
        g = clean(
            Completion.create(
                model=self.version, 
                prompt=input, 
                max_tokens=128,
                temperature=0, 
            ).choices[0].text
        )
        return g
    
    def classify(self, input: str | list) -> str | list:
        if isinstance(input, list):
            return [self.classify(i) for i in input]
        c = Completion.create(
                model=self.version, 
                prompt=input, 
                max_tokens=1,
                temperature=0, 
                logit_bias=self.logit_bias,
                logprobs=1,
        ).choices[0]
        return clean(c.text), c.logprobs["top_logprobs"][0][c.text]


class HuggingFaceModel(Model):
    
    def __init__(
        self, 
        version: str, 
        device: str="cpu", 
        batch_size: int=1
    ) -> None:
        self.version = version
        self.pipeline = pipeline(
            "text-generation", 
            model=self.version, 
            device=device, 
            batch_size=batch_size,
        )
        set_seed(0)
    
    def generate(self, input: list) -> str:
        responses = self.pipeline(
            input, 
            max_new_tokens=128, 
            return_full_text=False,
            do_sample=False,
        )
        return [clean(r[0]["generated_text"]) for r in responses]
    
    def classify(self, input: list) -> str:
        responses = self.pipeline(
            input, 
            max_new_tokens=1, 
            return_full_text=False,
            do_sample=False,
        )
        return [clean(r[0]["generated_text"]) for r in responses]
