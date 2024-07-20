import torch
import tiktoken
from random import randrange
import random
import json
import sentencepiece as spm

class mytokenizer():
    def __init__(self,B,T,device_type ="cuda", textUrl='D:\\repositories\\test ai assitance\\lighmcheni test\\myNanoGPT\\Hafezfull2.txt') -> None:
        self.B = B
        self.T = T
        self.allword = []
        self.tokens =[]
        self.clean_dict = []

        useload  = True
        tokenloader = False

        #  check compelete Tokens

        if useload == True:
                    try:
                        with open('D:\\repositories\\test ai assitance\\lighmcheni test\\myNanoGPT\\allword.json', 'r') as file:
                            # Perform any operations you need here
                            self.allword = json.load(file)
                    except IOError:
                        self.linesWord = open(textUrl, 'r', encoding="utf8").read().splitlines()
                        for line in self.linesWord:
                            result=line.split(" ")
                            self.allword= self.allword+result
                        
                        with open('D:\\repositories\\test ai assitance\\lighmcheni test\\myNanoGPT\\allword.json', 'w') as file:
                            json.dump(self.allword, file)

        if useload == True:
                    try:
                        with open('D:\\repositories\\test ai assitance\\lighmcheni test\\myNanoGPT\\clean_dict.json', 'r') as file:
                            # Perform any operations you need here
                            self.clean_dict = json.load(file)
                        
                        self.stoi = {s:i+1 for i,s in enumerate(self.clean_dict)}
                        self.itos = {i:s for s,i in self.stoi.items()}

                    except IOError:
                        for word in self.allword:
                            word=word.replace("\u200c","") 
                            #    word = self.clean_word(word)
                            if not word in self.clean_dict:
                                self.clean_dict.append(word)
                        
                        with open('D:\\repositories\\test ai assitance\\lighmcheni test\\myNanoGPT\\clean_dict.json', 'w') as file:
                            json.dump(self.clean_dict, file)
                        
                        self.stoi = {s:i+1 for i,s in enumerate(self.clean_dict)}
                        self.itos = {i:s for s,i in self.stoi.items()}

        if useload == True:
            try:
                with open('D:\\repositories\\test ai assitance\\lighmcheni test\\myNanoGPT\\iTokens.json', 'r') as file:
                    # Perform any operations you need here
                    self.tokens = json.load(file)
                    self.tokens = torch.tensor(self.tokens)
                    self.tokens.to(device_type)
                    tokenloader = True

            except IOError:
                    for iword in self.allword:
                        iword=iword.replace("\u200c","") 
                        # iword = self.clean_word(iword)
                        self.tokens.append(self.stoi[iword])
                        # self.tokens.append(self.stoi[' '])
                    
                    with open('D:\\repositories\\test ai assitance\\lighmcheni test\\myNanoGPT\\iTokens.json', 'w') as file:
                        json.dump(self.tokens, file)
                    
                    self.tokens = torch.tensor(self.tokens)
                    self.tokens.to(device_type)
        
        
        print('All Words',len(self.allword))
        print('clean_dict',len(self.clean_dict))

        self.current_position = 0

    def GetToken(self,string):
        return self.stoi[string]
    
    
    def GetTokenlist(self,string):
        allword = []
        # for line in string:
        #     result=line.split(" ")
        #     allword= allword+result
        
        result=string.split(" ")
        allword= allword+result
        
        itokens =[]
        for iword in allword:
            iword = iword.replace("<S2><S2>","<S2>") 
            iword = iword.replace("<S1><S1>","<S1>") 
            if iword != '':
                itokens.append(self.stoi[iword])

        itokens = torch.tensor(itokens)
        return itokens


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
        # self.current_position += 1
        # self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        # if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
        if self.current_position + (B * T * 1) > len(self.tokens):
            # self.current_shard = (self.current_shard + 1) % len(self.shards)
            # self.tokens = load_tokens(self.shards[self.current_shard])
            # self.current_position = B * T * self.process_rank
            self.current_position = 0
        return x, y,self.current_position
    

# myii = mytokenizer(2,3)
# print(myii.tokens)

class myDataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open("D:\\repositories\\test ai assitance\\lighmcheni test\\myNanoGPT\\Hafezfull.txt", 'r',encoding='utf-8') as f:
            text = f.read()
        enc = spm.SentencePieceProcessor(model_file='spm.model')
        # enc = tiktoken.get_encoding('gpt2')
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
