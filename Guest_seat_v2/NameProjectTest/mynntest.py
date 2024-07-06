import torch
import torch.nn as nn
import torch.optim as optim


class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() 
        self.hidden_liyer1 = nn.Linear(24,24)
        #self.hidden_liyer2 = nn.ReLU()
        self.output_liyer = nn.Softmax()

    def forward(self,x):
        x=self.flatten(x)
        x=self.hidden_liyer1(x)
       # x= self.hidden_liyer2(x)
        output = self.output_liyer(x)
        return output


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyNeuralNetwork().to(device)

print(model)


def validate_input(input_array):
    unique_numbers = set(input_array)
    if len(unique_numbers) != 24:
        raise ValueError("Input array must contain unique numbers from 1 to 24.")
    return torch.tensor(input_array, dtype=torch.float32)

# Example usage:
input_date = validate_input([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
# input_date = torch.rand(1,24,device=device)
print(input_date)
output = model(input_date)
print(output)

