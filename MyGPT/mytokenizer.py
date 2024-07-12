import torch
import tiktoken
from random import randrange
import random

class mytokenizer():
    def __init__(self,B,T,device_type ="cuda", textUrl='D:\\repositories\\test ai assitance\\lighmcheni test\\myNanoGPT\\Hafezfull.txt') -> None:
        self.B = B
        self.T = T
        self.linesWord = open(textUrl, 'r', encoding="utf8").read().splitlines()
        self.allword = []
        for line in self.linesWord:
            result=line.split(" ")
            self.allword= self.allword+result
        
        print('All Words',len(self.allword))

        self.clean_dict = []
        for word in self.allword:
            word=word.replace("\u200c","") 
            # word = self.clean_word(word)
            if not word in self.clean_dict:
                self.clean_dict.append(word)
    
        # self.clean_dict = self.create_clean_dict(self,self.allword)
        # print(clean_dict)
        print('clean_dict',len(self.clean_dict))

        self.stoi = {s:i+1 for i,s in enumerate(self.clean_dict)}
        self.itos = {i:s for s,i in self.stoi.items()}

        self.tokens =[]
        for iword in self.allword:
            iword=iword.replace("\u200c","") 
            # iword = self.clean_word(iword)
            self.tokens.append(self.stoi[iword])
            # self.tokens.append(self.stoi[' '])

        self.tokens = torch.tensor(self.tokens)
        self.tokens.to(device_type)

        self.current_position = 0

    def GetToken(self,string):
        return self.stoi[string]
    
    def GetRandomToken(self):
        return randrange(len(self.clean_dict))
    
    def GetWords(self,token):
        curentsords = ""
        for item in token:
            curentsords += self.itos[item]+" "
        return curentsords
    
    def GetRandomWords(self):
        return random.choice(self.clean_dict)
    
    def _clean_word(word):
        inword=word.replace("\u200c","") 
        return inword
    
    def create_clean_dict(self,word_array):
        clean_dict = []
        for word in word_array:
            word = self.clean_word(word)
            if not word in clean_dict:
                clean_dict.append(word)
        return clean_dict
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        # if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
        if self.current_position + (B * T * 1) > len(self.tokens):
            # self.current_shard = (self.current_shard + 1) % len(self.shards)
            # self.tokens = load_tokens(self.shards[self.current_shard])
            # self.current_position = B * T * self.process_rank
            self.current_position = 0
        return x, y
    

# myii = mytokenizer(2,3)
# print(myii.tokens)

class myDataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open("D:\\repositories\\test ai assitance\\lighmcheni test\\myNanoGPT\\Hafez.txt", 'r',encoding='utf-8') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        self.tokens.to('cuda')
        print(f"loaded {len(self.tokens)} tokens")
        print(f"l epoch =  {len(self.tokens) // B*T}  batches")

        self.current_position = 0
       
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        # if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
        if self.current_position + (B * T * 1) > len(self.tokens):
            # self.current_shard = (self.current_shard + 1) % len(self.shards)
            # self.tokens = load_tokens(self.shards[self.current_shard])
            # self.current_position = B * T * self.process_rank
            self.current_position = 0
        return x, y
