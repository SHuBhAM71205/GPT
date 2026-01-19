import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.GPT import GPT,GPTConfig
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from DataSet.DatasetLoader import TokenizedDataSet,collat_fn

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# constant
batch_size = 128
lr = 1e-3
epoch = 1
block_size = 1024

# DataLoader

TinySakesphere = DataLoader(
    dataset = TokenizedDataSet("./Tokenized/data.tok", max_len=block_size),
    batch_size = batch_size,
    shuffle = True,
    drop_last = True,
    collate_fn= lambda batch : collat_fn(batch)
)

#model
model = GPT(GPTConfig())
model = model.to(device)


# place for the lr scheduler if needed probably later usefull
optimizer = optim.AdamW(model.parameters(),lr=1e-3)


loss_history = []
throuput_history = []

# trainning loop
for i in range(epoch):
    model.train()
    for x,y in TinySakesphere:
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)

        logits,loss = model(x,y)

        loss_history.append(loss.item())
    
        loss.backward()
        
        optimizer.step()
        
        if i % 100 == 0:
            print(f"Iteration {i} : Loss {loss.item()}")