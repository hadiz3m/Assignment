import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures
import numpy as np
# %matplotlib inline

# read in all the words
words = open('D:\\repositories\\test ai assitance\\lighmcheni test\\name3.txt', 'r', encoding="utf8").read().splitlines()
words[:8]

print(len(words))
print(min(len(w) for w in words))
print(max(len(w) for w in words))

b = {}
for w in words:
  chs = ['<S>'] + list(w) + ['<E>']
  for ch1, ch2 in zip(chs, chs[1:]):
    bigram = (ch1, ch2)
    b[bigram] = b.get(bigram, 0) + 1

sorted(b.items(), key = lambda kv: -kv[1])
print(len(b))

N = torch.zeros((31, 31), dtype=torch.int32)
# print('N',N)

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

for w in words:
  chs = ['.'] + list(w) + ['.']
#   print('chs',chs)
  for ch1, ch2 in zip(chs, chs[1:]):
    # print(ch1, ch2)
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    N[ix1, ix2] += 1


import matplotlib.pyplot as plt
# matplotlib inline


plt.figure(figsize=(31,31))
plt.imshow(N, cmap='Blues')
for i in range(31):
    for j in range(31):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off');
# plt.show()

# print(N[0])


p = N[0].float()
p = p / p.sum()
p

g = torch.Generator().manual_seed(2147483647)
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
print('ix',ix)
itos[ix]

print('itos[ix]',itos[ix])

g = torch.Generator().manual_seed(2147483647)
p = torch.rand(3, generator=g)
p = p / p.sum()
p

mysample = torch.multinomial(p, num_samples=100, replacement=True, generator=g)


P = (N+1).float()
P /= P.sum(1, keepdims=True)

plt.figure(figsize=(31,31))
plt.imshow(P, cmap='Blues')
for i in range(31):
    for j in range(31):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, P[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off');
# plt.show()


g = torch.Generator().manual_seed(2147483647)

for i in range(15):
  
  out = []
  ix = 0
  while True:
    p = P[ix]
    print(p)
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
      break
  print(''.join(out))