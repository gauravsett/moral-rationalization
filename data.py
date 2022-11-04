import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class CommonsenseDataset(Dataset):
    
    def __init__(self) -> None:
        self.path = "data/commonsense/"
        self.data = self._load_data()
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> dict:
        return {
            "input": self.data.iloc[index]["input"],
            "label": self.data.iloc[index]["label"],
        }
    
    def _load_data(self) -> pd.DataFrame:
        data = pd.DataFrame(
            columns=["label", "input", "is_short", "edited"]
        )
        for file in os.listdir(self.path):
            if "ambig" in file:
                continue
            subset = pd.read_csv(self.path + file)
            data = pd.concat([data, subset]).reset_index(drop=True)
        data = data.loc[data["is_short"] == True][["label", "input"]]
        return data


commonsense_dataloader = DataLoader(
    CommonsenseDataset(),
    batch_size=1,
    shuffle=False,
)