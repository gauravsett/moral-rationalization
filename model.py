from secrets import OPENAI_API_KEY
import openai
from openai import Completion


class Model():
    
    def __init__(self) -> None:
        pass
    
    def generate(self, input: str) -> str:
        pass
    
    def classify(self, input: str, classes: list = [0, 1]) -> int:
        pass


class GPT3(Model):
    
    def __init__(self, version: str) -> None:
        openai.api_key = OPENAI_API_KEY
        self.version = version
    
    def generate(self, input: str) -> str:
        response = Completion.create(
            model=self.version, prompt=input, temperature=0, max_tokens=256
        )
        return response.choices[0].text
    
    def classify(self, input: str, classes: list = [0, 1]) -> int:
        response = Completion.create(
            model=self.version, prompt=input, temperature=0, max_tokens=1
        )
        code = {"yes": 1, "no": 0}
        return code[response.choices[0].text.lower()]
   