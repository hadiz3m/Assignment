
import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Number of guests and seats
num_guests = 24
num_seats = 24
confilictcount = 40

guesnamelist = ["jeme","zahra","tome","stiv","fati","javad","ston","mahdi","sobhan","tohid","esi","tiyam","messi","ashor","nahid","ziba","mijica","ana","ali","ayoub","solmaz","sabori","mohsa","sohyla"]
space = [" "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "]
gaslistname1 = np.random.choice(guesnamelist,24)
gaslistname2 =  np.random.choice(guesnamelist,24)

invatedlistspace = np.char.add(gaslistname1,space)
invatedlist = np.char.add(invatedlistspace,gaslistname2)

Confilict_matrix =np.array([[0 for x in range(num_guests)] for y in range(num_guests)])

counter1 = 0

while True:
    i = random.randint(0,num_guests-1)
    j = random.randint(0,num_guests-1)

    if i!=j:
        if Confilict_matrix[i,j] == 0:
            Confilict_matrix[i,j] = 1
            counter1 +=1

    if counter1 >=confilictcount:
        break


print(Confilict_matrix)

# Neural Network Model
class SeatingModel(nn.Module):
    def __init__(self, num_guests, num_seats):
        super(SeatingModel, self).__init__()
        hideleyercount = num_guests*num_guests
        self.fc1 = nn.Linear(num_guests, hideleyercount)
        self.fc2 = nn.Linear(hideleyercount,num_seats)
        # self.fc3 = nn.Linear(num_seats, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        # x = self.fc3(x)
        return x


def GetConfilict(guest1,Guest2):
    curentrelation = 0
    try:
       curentrelation = Confilict_matrix[guest1][Guest2]
    except:
        curentrelation =0
    return curentrelation

def calculateRealConfilictDependOnGuestPosition(input):
    
    seatConfilict = np.zeros((6, 4))
    
    for i in range(6):
        for j in range(4):
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

def CheckSimilarityRealtoTrineConfilict(RealChair,TraiChair):
    CheckConfilict = np.zeros((6, 4))
    
    for i in range(6):
        for j in range(4):
            if RealChair[i,j] == TraiChair[i,j]:
                CheckConfilict[i,j] =1
    persent = np.sum(CheckConfilict) / 24
    return persent

# # Loss function to minimize conflicts
def conflict_loss(input,output):
    loss = 0

    ChairConfilictarrange  = output.detach().numpy().reshape(6,4)
    ChairConfilictarrangenormal = np.where(ChairConfilictarrange > 0, 1, 0)
    # print('Chair Confilict traine:',ChairConfilictarrangenormal)

    guestSeat = input.detach().numpy().reshape(6,4)
    # print('guestSeat',guestSeat)
    GuestSeatConfilict =  calculateRealConfilictDependOnGuestPosition(guestSeat)
    # print('GuestSeatConfilict',GuestSeatConfilict)

    CompairPersentage = CheckSimilarityRealtoTrineConfilict(GuestSeatConfilict,ChairConfilictarrangenormal)
    # print('Compair',CompairPersentage)
    # for conflict in conflicts:
    #     print('conflict:',conflict)
    #     guest1, guest2 = conflict
    #     loss += torch.abs(output[guest1] - output[guest2])

    GuestSeatConfilictFlatten =torch.from_numpy(np.array(GuestSeatConfilict.flatten()))
    # print('output:',output)
    # print('GuestSeatConfilictFlatten',GuestSeatConfilictFlatten)

    mse_loss = torch.mean((output - GuestSeatConfilictFlatten) ** 2)
    # print('mse_loss:',mse_loss)
    return mse_loss

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

def ShowBestResult(BestMin,BestSeatArenge):
    print('Best Lost:',BestMin) 
    print('Best Arrange Seat:',BestSeatArenge)
    guestSeat = BestSeatArenge.detach().numpy().reshape(6,4)
    # print('guestSeat',guestSeat)
    GuestSeatConfilict =  calculateRealConfilictDependOnGuestPosition(guestSeat)
    print('Guest Seat Confilict',GuestSeatConfilict)

# # Training the model
def train_model(model, num_epochs=1000, learning_rate=0.1):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    BestMin = 10
    BestSeatArenge = 0
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        
        # Random input for guests
        input_data = NewGuestArenge()
        # print('input_data',input_data)

        # Forward pass
        output = model(input_data)

        # Compute loss
        loss = conflict_loss(input_data,output)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        
        curentloss = loss.item()
        if curentloss < BestMin:
            BestMin = curentloss
            BestSeatArenge = input_data

    ShowBestResult(BestMin,BestSeatArenge)
    
    
# Initialize and train the model
model = SeatingModel(num_guests, num_seats)
train_model(model)

# Get the seating arrangement
model.eval()
input_data =  NewGuestArenge()

# seating_arrangement = model(input_data).detach().numpy()
# print("Seating Arrangement:", seating_arrangement)