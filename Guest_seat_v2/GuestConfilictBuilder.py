import random
import numpy as np
import os


class ConfilictTable():
    
    def Build(num_member=5,num_Group=40,confilictcount = 600) -> np.array:
        num_member = num_member
        num_Group = num_Group
        GuestNumber = num_member *num_Group
        confilictcount = confilictcount

        guesnamelist = ["jeme","zahra","tome","stiv","fati","javad","ston","mahdi","sobhan","tohid","esi","tiyam","messi","ashor","nahid","ziba","mijica","ana","ali","ayoub","solmaz","sabori","mohsa","sohyla"]
        space = [" "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "]
        gaslistname1 = np.random.choice(guesnamelist,24)
        gaslistname2 =  np.random.choice(guesnamelist,24)

        invatedlistspace = np.char.add(gaslistname1,space)
        invatedlist = np.char.add(invatedlistspace,gaslistname2)

        Confilict_matrix =np.array([[0 for x in range( GuestNumber)] for y in range( GuestNumber)])

        print(len(Confilict_matrix))

        counter1 = 0

        while True:
            i = random.randint(0, GuestNumber-1)
            j = random.randint(0, GuestNumber-1)

            if i!=j:
                if Confilict_matrix[i,j] == 0:
                    Confilict_matrix[i,j] = 1
                    counter1 +=1

            if counter1 >=confilictcount:
                break


        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        confilictAddress = ROOT_DIR+"\\ConfilictList.npy"

        np.save(confilictAddress,Confilict_matrix)
        return Confilict_matrix
        #open and read the file after the appending:
        # Confilict_matrix = np.load(confilictAddress)
        # print(Confilict_matrix.sum())

    def get()-> np.array:
        try:   
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            confilictAddress = ROOT_DIR+"\\ConfilictList.npy"
            Confilict_matrix = np.load(confilictAddress)
            return Confilict_matrix
        except:
           Confilict_matrix = ConfilictTable.Build()
           return Confilict_matrix


class GuestTable():
    
    def Build(num_member=5,num_Group=40) -> np.array:
        num_member = num_member
        num_Group = num_Group
        GuestNumber = num_member *num_Group

        guesnamelist = ["jeme","zahra","tome","stiv","fati","javad","ston","mahdi","sobhan","tohid","esi","tiyam","messi","ashor","nahid","ziba","mijica","ana","ali","ayoub","solmaz","sabori","mohsa","sohyla"]
        space = [" "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "," "]
        gaslistname1 = np.random.choice(guesnamelist,24)
        gaslistname2 =  np.random.choice(guesnamelist,24)

        invatedlistspace = np.char.add(gaslistname1,space)
        invatedlist = np.char.add(invatedlistspace,gaslistname2)

        GuestList =np.arange(1,GuestNumber+1)
        GuestList = GuestList.reshape(num_Group,num_member)

        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        confilictAddress = ROOT_DIR+"\\GuestList.npy"

        np.save(confilictAddress,GuestList)
        return GuestList

    def get()-> np.array:
        try:   
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            GuestList = ROOT_DIR+"\\GuestList.npy"
            GuestList = np.load(GuestList)
            return GuestList
        except:
           GuestList = GuestTable.Build()
           return GuestList


# curentconfilict = ConfilictTable.get()
# print(curentconfilict)

# CurentGustList2 = GuestTable.get()
# print(CurentGustList2)