import json
import openai
from openai import Completion
from transformers import pipeline, set_seed


class Model():
    
    def __init__(self) -> None:
        raise NotImplementedError
    
    def generate(self, input: str) -> str:
        raise NotImplementedError
    
    def classify(self, input: str) -> int:
        raise NotImplementedError


class OpenAIModel(Model):
    
    def __init__(self, version: str) -> None:
        with open("secrets.json", "r") as f:
            openai.api_key = json.load(f)["OPENAI_API_KEY"]
        self.version = version
    
    def generate(self, input: str) -> str:
        response = Completion.create(
            model=self.version, prompt=input, temperature=0, max_tokens=128
        )
        return response.choices[0].text.strip()
    
    def classify(self, input: str) -> int:
        response = Completion.create(
            model=self.version, prompt=input, temperature=0, max_tokens=1, logprobs=1
        )
        return response.choices[0].text.strip()


class HuggingFaceModel(Model):
    
    def __init__(self, version: str) -> None:
        self.version = version
        self.pipeline = pipeline("text-generation", model=self.version)
        set_seed(7)
    
    def generate(self, input: str) -> str:
        response = self.pipeline(input, max_new_tokens=128, do_sample=False)
        return response[0]["generated_text"][len(input):].strip()
    
    def classify(self, input: str) -> int:
        response = self.pipeline(input, max_new_tokens=1, do_sample=False)
        return response[0]["generated_text"][len(input):].strip()
    