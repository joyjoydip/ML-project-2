import pickle as pk

model=pk.load(open('bengaluru_house_price_prediction.pkl','rb'))

import warnings
warnings.filterwarnings("ignore", category=UserWarning)  
#their is warning coming that's why i am using the above function to ignore it 

import json
col=json.load(open('column_names.json','r'))
# print(col)
#the above file contains the columns names but here 
#the location is from the 3 index to last index 
print(len(col['data_columns']))
locations=col['data_columns'] #storing list of the location in the list 
print(locations)

#it means their are 240 location in the columns 
#and 0 1 2 columns and only one location is the input 
import numpy as np

inp=np.zeros(len(col['data_columns']))
print(inp.shape)

 #here we get the array of the 243 len with filling with the 0 
#so we have to take 
#total_sqft,'bath',bhk and 1 location as input and predict the price as the output 
def find_location(location):

    for i in range(3,243):


        if locations[i]==location.lower():  #our location are present in the smaller case that
            #'s why we have to use the lower() 
            
            
                return i 
         
    return -1
            #because their is 1 location which is not present in the location's list 


# print(find_location('1st Block Jayanagar'))
# print(find_location('1st Phase JP Nagar'))
def predict_price(location,sqft,bath,bhk):    
    loc_index=find_location(location)

    inp=np.zeros(len(locations))  #here we are generating the input same as the 
    #input len 
    
    inp[0] = sqft
    inp[1] = bath
    inp[2] = bhk 
    if loc_index !=-1:
        inp[loc_index] = 1

    inp=inp.reshape(1,-1)

    return model.predict(inp)[0]

print(predict_price('1st Block Jayanagar',2850,4,4))


# print(predict_price('1st Block Jayanagar', 2850, 4, 4))
# print(predict_price('Non Existent Location', 2850, 4, 4))
