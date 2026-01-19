import tiktoken
import os
import mmap
import array

data_folder = "./Data"
save_folder = "./Tokenized"

tiktoken_name = "gpt2"

tokenizer = tiktoken.get_encoding(tiktoken_name)




with open(f"{data_folder}/input.txt","r") as in_f,\
    open(f"{save_folder}/data.tok","wb") as out_f:
    
    mmp = mmap.mmap(in_f.fileno(),0,access=mmap.ACCESS_READ)
    data = tokenizer.encode(in_f.read())

    byte_arr = array.array('i', data)
    out_f.write(byte_arr)
    mmp.close()
    

