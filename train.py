'''
Created on Nov 16, 2017

@author: 
'''

protos = dict()
#protos['lm']= 

def Train(epoch):
    return 1


startEpoch=0
nEpochs=1
for epoch in range(startEpoch, nEpochs):
    learning_rate = 0.001
    train_loss = Train(epoch)
    
    print('training loss for epoch ' + str(epoch)  + ' is : ' + str(train_loss))
    
    
