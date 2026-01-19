from dataclasses import dataclass
import torch
import tiktoken

from Model.GPT import GPT, GPTConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")


gpt = GPT(GPTConfig()).to(device)
gpt = GPT.from_pretrained("gpt2")

gpt.eval()
gpt.to(device)

num_return_sequence = 5
max_seq_len = 30

enc = tiktoken.get_encoding('gpt2')

tokenized = enc.encode("Hello I am a Language Model")

tokenized_tensor = torch.tensor(data=tokenized,dtype=torch.long)

tokenized_tensor = tokenized_tensor.unsqueeze(0).repeat(num_return_sequence,1)

x = tokenized_tensor.to(device)


torch.cuda.manual_seed(42)
with torch.inference_mode():
    
    while x.shape[1] < max_seq_len:
        
        logits = gpt(x)
        
        logits = logits[:,-1,:] # taking the logits of the next _token in row
        
        probs = torch.softmax(logits,dim=-1)
        
        topk_probs,topk_indices = torch.topk(probs,50,dim=-1)
        
        ix = torch.multinomial(topk_probs,1)
        print(ix)
        xcol = torch.gather(topk_indices,-1,ix)
        
        x = torch.cat((x,xcol),dim = 1)
    
    for i in range(x.shape[0]):
        print(enc.decode(x[i,:].tolist(),errors="replace"))
        