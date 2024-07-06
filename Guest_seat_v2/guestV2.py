import torch
import torch.nn as nn
import random
import GuestConfilictBuilder as confilict
import torch.optim as optim
import numpy as np

num_guests = 200

GuestTable = confilict.GuestTable.get()
print(GuestTable)

CurentConfilict = confilict.ConfilictTable.get()
print(CurentConfilict)

class GuesModel(nn.Module):
    def __init__(self) -> None:
        super(GuesModel,self).__init__()
        self.layer1 = nn.Linear(200,200)
        self.layer2 = nn.Linear(200,2)

    def forward(self,x):
     x=self.layer1(x)
     x=self.layer2(x)
     return x   


def GetConfilict(guest1,Guest2):
    curentrelation = 0
    try:
       curentrelation = CurentConfilict[guest1][Guest2]
    except:
        curentrelation =0
    return curentrelation

def calculateRealConfilictDependOnGuestPosition(input):
    
    seatConfilict = np.zeros((20, 10))
    
    for i in range(20):
        for j in range(10):
            curentGuest = round(input[i,j] *100)
            # get around curent Guest
            # [front,back,left,right]
            aroundGuest = [[i-1,j],[i+1,j],[i,j-1],[i,j+1]]
            # print('aroundGuest:',aroundGuest)

            for guestPo in aroundGuest:
                poi,poj = guestPo
                # print('guestPo',guestPo)
                nextGuest = 0
                
                try:
                     if poi>=0 and poj>=0:
                        nextGuest = round(input[poi,poj] * 100)
                except:
                     nextGuest = 0
                
                if nextGuest != 0:
                   gustconfilict = GetConfilict(curentGuest,nextGuest)
                #    print('guest1:',curentGuest, ' via guest2:' , nextGuest , ' has :',gustconfilict)
                   if gustconfilict == 1:
                       seatConfilict[i,j] = gustconfilict
                       break
    
    return seatConfilict

def checkGroupJoinLoss():
    GuestTable
    

# # Loss function to minimize conflicts
def conflict_loss(input,output):
    loss = 0

    guestSeat = input.detach().numpy().reshape(20,10)
    GuestSeatConfilict =  calculateRealConfilictDependOnGuestPosition(guestSeat)

    Groupjoin = input.detach().numpy().reshape(40,5)
    CompairPersentage = CheckSimilarityRealtoTrineConfilict(GuestSeatConfilict)
    
    GuestSeatConfilictFlatten =torch.from_numpy(np.array(GuestSeatConfilict.flatten()))
    # print('output:',output)
    # print('GuestSeatConfilictFlatten',GuestSeatConfilictFlatten)

    mse_loss = torch.mean((output - GuestSeatConfilictFlatten) ** 2)
    # print('mse_loss:',mse_loss)
    return mse_loss


def CheckSimilarityRealtoTrineConfilict(RealChair,TraiChair):
    CheckConfilict = np.zeros((20, 10))
    for i in range(20):
        for j in range(10):
            if RealChair[i,j] == TraiChair[i,j]:
                CheckConfilict[i,j] =1
    persent = np.sum(CheckConfilict) / 200
    return persent


def NewGuestArenge():
    tensor_values = []
    counter =0
    while(True):
        random_number = random.randint(1, num_guests)/100
        
        if random_number not in tensor_values:
            tensor_values.append(random_number)
            counter +=1
        
        if counter >= num_guests:
            break

    return torch.tensor(tensor_values)

def triningModel(model,epochs =1000 , LeraningRate = 0.1):
   optimizer = optim.Adam(model.parameters(), lr=LeraningRate)

   for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Random input for guests
    input_data = NewGuestArenge()
    print('input_data',input_data)

    # Forward pass
    output = model(input_data)

    # Compute loss
    loss = conflict_loss(input_data,output)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epoch}], Loss: {loss.item():.4f}')

        
    curentloss = loss.item()
    if curentloss < BestMin:
        BestMin = curentloss
        BestSeatArenge = input_data

    ShowBestResult(BestMin,BestSeatArenge)


# Initialize and train the model
model = GuesModel()
triningModel(model)

model.eval()
input_data =  NewGuestArenge()

seating_arrangement = model(input_data).detach().numpy()
print("Seating Arrangement:", seating_arrangement)