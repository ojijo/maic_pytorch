'''
Created on Nov 16, 2017

@author: zhj
'''

from net import lm
from net.dl import DataLoader

opt = dict()
opt['input_h5'] = "./data/ai-data/output/cocotalk_challenge.h5"
opt['input_json'] = "./data/ai-data/output/cocotalk_challenge.json"
opt['nn_neighbor'] = ""
opt['batch_size'] = "5"
opt['seq_per_img'] = "5"
opt['thread_num'] = "1"

protos = dict()
protos['lm']= lm.LanguageModel()
loader = DataLoader(opt)

def Train(epoch):
    return 1


startEpoch=0
nEpochs=1
for epoch in range(startEpoch, nEpochs):
    learning_rate = 0.001
    train_loss = Train(epoch)
    
    print('training loss for epoch ' + str(epoch)  + ' is : ' + str(train_loss))
    
    
