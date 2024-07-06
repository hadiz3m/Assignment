import torch
import torch.nn as nn
import torch.optim as optim
import unidecode
import string
import random

# Read a file and split into lines
def readLines(filename):
    with open(filename, encoding='utf-8') as f:
        return f.read().strip().split('\n')

# Turn a string into a list of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

# Define the RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

# Get the list of names
names = readLines('D:\\repositories\\test ai assitance\\lighmcheni test\\name3.txt')

# Build the category_lines dictionary, a list of names per category
category_lines = {}
words = open('D:\\repositories\\test ai assitance\\lighmcheni test\\name3.txt', 'r', encoding="utf8").read().splitlines()
chars = sorted(list(set(''.join(words))))
# all_letters = string.ascii_letters + " .,;'"
all_letters2 = chars[:29] #+ [" ",".",",",";","'"]
all_letters = ""

for i in all_letters2:
    all_letters+= i

n_letters = len(all_letters)

for name in names:
    category = name[0]  # Category will be the first letter of the name
    if category not in category_lines:
        category_lines[category] = []
    category_lines[category].append(name)

# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# Random training example
def randomTrainingExample():
    category = 'آ'
    line = randomChoice(category_lines[category])
    while(True):
        try:
            category = randomChoice(all_letters)
            line = randomChoice(category_lines[category])
            break
        except():
            category = randomChoice(all_letters)
            line = randomChoice(category_lines[category])
            break

    category_tensor = torch.tensor([all_letters.find(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor
   

# Training loop
def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()
    rnn.zero_grad()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()

# Initialize the RNN
n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_letters)

# Loss function and optimizer
criterion = nn.NLLLoss()
learning_rate = 0.005

# Train the model
for i in range(10000):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    if i % 500 == 0:
        print(f'Epoch {i} Loss: {loss}')

# Save the model
torch.save(rnn, 'char_rnn.pth')
