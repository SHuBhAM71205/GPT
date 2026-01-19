from torch.utils.data import Dataset
import os
import array
import mmap as mm
import torch
from typing import List,Tuple

def collat_fn(batch:List[Tuple[array.array,array.array]]) -> Tuple[torch.Tensor,torch.Tensor]:
    x_tensor = torch.stack([torch.tensor(x[0],dtype = torch.long) for x in batch],dim = 0)
    y_tensor = torch.stack([torch.tensor(y[1],dtype = torch.long) for y in batch],dim = 0)
    return x_tensor,y_tensor

class TokenizedDataSet(Dataset):
    
    def __init__(self,tokenized_data_location: str,max_len) -> None:
        super().__init__()
        
        self.loc = tokenized_data_location
        assert os.path.exists(self.loc) ,"tokenized file dont exist wrong path is passesd"
        self.tokens = array.array("i")
        self.max_len = max_len
        
        with open(f"{self.loc}","rb") as f:
            mmp = mm.mmap(f.fileno(),0,access=mm.ACCESS_READ)
            self.tokens.frombytes(mmp)
            mmp.close()
        
    def __len__(self):
        
        # it is because self.tokens - (self.max_len + 1) + 1
        return len(self.tokens) - self.max_len
    
    
    def __getitem__(self, index:int):
        chunk = self.tokens[index:index+self.max_len + 1]
        
        x = chunk[:-1]
        y = chunk[1:]

        return x,y



if __name__ == "__main__":

    # for the testing purposes
    tk = TokenizedDataSet("./Tokenized/data.tok", 3)

    datasetLoader = torch.utils.data.DataLoader(
        dataset = tk,
        batch_size = 2,
        shuffle = True,
        drop_last = True,
        collate_fn = lambda batch: collat_fn(batch)
    )
    
    print(next(iter(datasetLoader)))
    pass