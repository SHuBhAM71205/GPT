from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as f
import torch

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class Attention(nn.Module):
    
    def __init__(self,config: GPTConfig):
        super().__init__()
        self.config = config
        assert self.config.n_embd % self.config.n_head == 0, "Embedding dimension must be divisible by number of heads"
        
        # not needed for the flash attention but still let it stay here
        self.c_attn = nn.Linear(self.config.n_embd, 3 * self.config.n_embd)
        self.c_proj = nn.Linear(self.config.n_embd, self.config.n_embd)
        
        self.register_buffer("bias", 
                                torch.tril(
                                    torch.ones(config.block_size, config.block_size)
                                ).view(1, 1, config.block_size, config.block_size)
                            )
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        self.flash_available = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self,x:torch.Tensor):
        
        '''
        Docstring for forward
        
        :param self: Description
        :param x: Input tensor of shape (B,T,C)
        :type x: torch.Tensor
        :return: Output tensor after applying attention mechanism
        :rtype: torch.Tensor
        '''
        B, T, C = x.size()
        
        qkv = self.c_attn(x)
        
        q,k,v = torch.split(qkv,self.config.n_embd,dim=-1)
        
        q = q.view(size=(B, T, self.n_head, self.n_embd // self.n_head)).transpose(1,2)
        k = k.view(size=(B, T, self.n_head, self.n_embd // self.n_head)).transpose(1,2)
        v = v.view(size=(B, T, self.n_head, self.n_embd // self.n_head)).transpose(1,2)
        
        if self.flash_available:
            out = f.scaled_dot_product_attention(
                q,k,v,
                dropout_p=0.0,
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-1,-2)) * (1.0 / torch.sqrt(torch.tensor(self.n_embd // self.n_head,dtype=torch.float32)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float("-inf")) # type: ignore
            att = f.softmax(att,dim=-1)
            out = att @ v

        out = out.transpose(1,2).contiguous().view(B,T,C)
        return self.c_proj(out)
    
class MLP(nn.Module):
    
    def __init__(self,config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.fc_c = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.GeLU = nn.GELU(approximate='tanh')
        self.fc_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        

    def forward(self,x:torch.Tensor):
        '''
        Docstring for forward
        
        :param self: Description
        :param x: Input tensor of shape (B,T,C)
        :type x: torch.Tensor
        :return: Output tensor after applying MLP
        :rtype: torch.Tensor
        '''
        out = self.fc_proj(self.GeLU(self.fc_c(x)))
        return out

class Block(nn.Module):
    
    def __init__(self,config : GPTConfig):
        super().__init__()
        self.config = config
        
        self.att = Attention(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        
    
    def forward(self,x: torch.Tensor):
        '''
        Docstring for forward
        
        :param self: Description
        :param x: Input tensor of shape (B,T,C) 
        :type x: torch.Tensor
        :return: Output tensor after applying the block
        :rtype: torch.Tensor
        '''
        x  = self.att(self.ln1(x)) + x
        x = self.mlp(self.ln2(x)) + x
        return x

# Ha Ha Ha IGNORE type: ignore just pylance things here just want clean things
class GPT(nn.Module):
    def __init__(self,config:GPTConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(self.config.vocab_size,self.config.n_embd),
                wpe = nn.Embedding(self.config.block_size, self.config.n_embd),
                h = nn.ModuleList([ Block(config) for _ in range(self.config.n_layer)])
            )
        )
        
        self.lm_head = nn.Linear(self.config.n_embd,self.config.vocab_size,bias=False)
        
        self.lm_head.weights = transformer.wte.weights # type:ignore
        
    
    def forward(self,x:torch.Tensor,y=None):
        '''
            Docstring for forward
            :param self: Description
            :param x: Input tensor of shape (B,T)
            :type x: torch.Tensor
            :param y: Target tensor of shape (B,T), defaults to None
            :type y: torch.Tensor, optional
            :return: Tuple of logits and loss (if y is provided)
            :rtype: Tuple[torch.Tensor, Optional[torch.Tensor]]
        '''
        
        B,T = x.size()
        
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted" 
        
        token_embeddings = self.transformer.wte(x) #type: ignore
        
        positions = torch.arange(0,T,dtype=torch.long,device=x.device)
        positional_embeddings = self.transformer.wpe(positions) # type: ignore
        
        embeddings = token_embeddings + positional_embeddings # (B,T,C)
        
        for block in self.transformer.h: # type: ignore
            embeddings = block(embeddings)
        
        logits = self.lm_head(embeddings) # give the #(B,T,Vocab_size ) 
        
        loss = None
        if y is not None:
            loss = f.cross_entropy(
                logits.view(B*T, self.config.vocab_size),
                y.view(B*T)
            )
        
        return logits,loss

    
    # just following karpathy's implementation here
    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {"gpt2","gpt2-medium,gpt2-large,gpt2-xl"}
        
        from transformers import GPT2LMHeadModel
        
        print("loading weights from pretrained gpt: %s" %model_type)

        config_args ={
        "gpt2":
        dict(n_layer=12, n_head=12, n_embd=768), #124M params
        "gpt2-medium":
        dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
        'gpt2-large':
        dict(n_layer=36, n_head=28, n_embd=1280), #774M params
        'gpt2-xl':
        dict(n_layer=48, n_head=25, n_embd=1600), #1558M params
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024 
        config  = GPTConfig(**config_args)
        model  = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]
        
        model_hf  = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

if __name__ == "__main__":
    # for testing the code

    model: GPT = GPT(GPTConfig())
    pass
