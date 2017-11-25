'''
Created on Nov 20, 2017

@author: zhj
'''
import h5py
import json
import torch

class DataLoader(object):

  def __init__(self,opt):
    #load the json file which contains additional information about the dataset
    print('DataLoader loading json file: ', opt['input_json'])
    with open(opt['input_json'], "r") as f:
      self.info = json.load(f)
    self.ix_to_word = self.info['ix_to_word']
    self.vocab_size = self.ix_to_word.__len__()

    self.batch_size = int(opt['batch_size']) # how many images get returned at one time (to go through CNN)
    self.seq_per_img = int(opt['seq_per_img']) # number of sequences to return per image

    print('vocab size is ' + str(self.vocab_size))
    
    self.h5_file = h5py.File(opt['input_h5'], 'r')
    images= self.h5_file.get('/images')
    images_size = images.shape
    self.num_images = images_size[0]
    self.num_channels = images_size[1]
    self.max_image_size = images_size[2]
    print(images_size)
    
    #load in the sequence data
    labels = self.h5_file.get('/labels')
    labels_size = labels.shape
    self.seq_length = labels_size[1]
    print('max sequence length in data is ' + str(self.seq_length))
    
    #load the pointers in full to RAM (should be small enough)
    self.label_start_ix = self.h5_file.get('/label_start_ix').value
    self.label_end_ix = self.h5_file.get('/label_end_ix').value
    self.labels = labels
    self.label_lens = self.h5_file.get('/label_length').value
    
    self.split_ix = dict()
    self.iterator = dict()
    self.image_ids = torch.LongTensor(self.num_images).zero_()
    for i,img in self.info['images']:
      split = img.split
      if (not self.split_ix[split]):
        #initialize new split
        self.split_ix[split] = dict()
        self.iterator[split] = 1
      self.split_ix[split][i]= i
      self.image_ids[i] = img.id    

    self.__size = dict()
    for k,v in self.split_ix:
      print('assigned' + str(v) + 'images to split' + str(k))

    #self.meanstd = {
    #  mean = { 0.485, 0.456, 0.406 },
    #  std = { 0.229, 0.224, 0.225 },
    #}

    #self.transform = t.Compose{
    # t.ColorNormalize(self.meanstd)
    #}    
    