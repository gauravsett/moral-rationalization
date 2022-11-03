import os
import pandas as pd


class Data():
    
    def __init__(self) -> None:
        pass
        
    def load_data(self) -> pd.DataFrame:
        pass


class CommonsenseData(Data):
    
    def __init__(self) -> None:
        self.path = "data/commonsense/"
        self.data = self.load_data()
    
    def load_data(self) -> pd.DataFrame:
        data = pd.DataFrame(
            columns=["label", "input", "is_short", "edited", "subset"]
        )
        for file in os.listdir(self.path):
            if "ambig" in file:
                continue
            subset = pd.read_csv(self.path + file)
            subset["subset"] = file.split("_", maxsplit=1)[1][:-4]
            data = pd.concat([data, subset]).reset_index(drop=True)
        return data
