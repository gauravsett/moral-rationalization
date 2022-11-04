import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class Data():
    
    def __init__(self) -> None:
        raise NotImplementedError
        
    def _load_data(self) -> pd.DataFrame:
        raise NotImplementedError


class CommonsenseData(Data):
    
    def __init__(self) -> None:
        self.path = "data/commonsense/"
        self.data = self._load_data()
    
    def _load_data(self) -> pd.DataFrame:
        data = pd.DataFrame(
            columns=["label", "input", "is_short", "edited"]
        )
        for file in os.listdir(self.path):
            if "ambig" in file:
                continue
            subset = pd.read_csv(self.path + file)
            data = pd.concat([data, subset]).reset_index(drop=True)
        data = data.loc[data["is_short"] == True][["input", "label"]]
        return data
