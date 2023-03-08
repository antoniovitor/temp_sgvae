import csv
import h5py
import random

import pandas as pd
import parameters
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler

params = parameters.load_params()

def write_csv(d, path):
  """
  Writes the results to the dict d
  """
  with open(path, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(d.keys())
    writer.writerows(zip(*d.values()))
    

class LoadSmilesTestData:
    def __init__(self, labels_path):
        random.seed(42)  # To ensure the same labels are picked everytime
        f = pd.read_csv(labels_path)
        self.smiles = list(f.loc[:, 'smiles'])
        self.prop1 = list(f.loc[:, params['prop'][0]])
        self.prop2 = list(f.loc[:, params['prop'][1]])

class LoadSmilesData:
    
    def __init__(self, labels_path, normalization):
        
        random.seed(42)  # To ensure the same labels are picked everytime
        f = pd.read_csv(labels_path)
            
        smiles = list(f.loc[:, 'smiles'])
        prop1 = list(f.loc[:, params['prop'][0]])
        prop2 = list(f.loc[:, params['prop'][1]])
        
        # choosing random ids for train and test
        ids = list(range(len(f.index)))
        random.shuffle(ids)
        chunk = int(0.15 * len(smiles))
        
        self._ids_train = sorted(ids[chunk:])
        self._ids_test = sorted(ids[0:chunk])

        self._smiles_train = [smiles[i] for i in self._ids_train]
        self._smiles_test = [smiles[i] for i in self._ids_test]
        # property 1
        property_train_1 = np.asarray([prop1[i] for i in self._ids_train]).reshape(-1, 1)
        property_test_1 = np.asarray([prop1[i] for i in self._ids_test]).reshape(-1, 1)
        # property 2
        property_train_2 = np.asarray([prop2[i] for i in self._ids_train]).reshape(-1, 1)
        property_test_2 = np.asarray([prop2[i] for i in self._ids_test]).reshape(-1, 1)
        
        # normalizing the property values
        if normalization.lower().strip() == 'minmax':
            minmaxscaler = MinMaxScaler()
            self._property_train_normalized_1 = minmaxscaler.fit_transform(property_train_1)
            self._property_test_normalized_1 = minmaxscaler.transform(property_test_1)
            self._property_train_normalized_2 = minmaxscaler.fit_transform(property_train_2)
            self._property_test_normalized_2 = minmaxscaler.transform(property_test_2)
            self.scaler = minmaxscaler
            
        elif normalization.lower().strip() == 'standard':
            standardscaler = StandardScaler()
            self._property_train_normalized_1 = standardscaler.fit_transform(property_train_1)
            self._property_test_normalized_1 = standardscaler.transform(property_test_1)
            self._property_train_normalized_2 = standardscaler.fit_transform(property_train_2)
            self._property_test_normalized_2 = standardscaler.transform(property_test_2)
            self.scaler = standardscaler
            
        elif normalization.lower().strip() == 'none':
            self._property_train_normalized_1 = property_train_1
            self._property_test_normalized_1 = property_test_1
            self._property_train_normalized_2 = property_train_2
            self._property_test_normalized_2 = property_test_2
            self.scaler = None
        
        else:
            raise ValueError('Invalid normalization method. Current options are minmax, standard and none')
    
    def smiles_train(self):
        return self._smiles_train
        
    def smiles_test(self):
        return self._smiles_test
  
    def property_train(self):
        return self._property_train_normalized_1.flatten().tolist(), self._property_train_normalized_2.flatten().tolist()
        
    def property_test(self):
        return self._property_test_normalized_1.flatten().tolist(), self._property_test_normalized_2.flatten().tolist()
        
    def ids_train(self):
        return self._ids_train
        
    def ids_test(self):
        return self._ids_test


class AnnealKL:
  """
  Anneals weight to the kl term in a cyclical manner
  """
  def __init__(self, n_epoch, start=0, stop=1, n_cycle=2, ratio=0.5):
    self.L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) 

    for c in range(n_cycle):
      v , i = start , 0
      while v <= stop:
        self.L[int(i+c*period)] = 1.0/(1.0 + np.exp(-(v*12.-6.)))
        v += step
        i += 1
        
  def beta(self, epoch):
    return self.L[epoch]
    
    
class TrainAniondataset(Dataset):
  """
  Loads the QM9 training data set for the dataloader
  """
  def __init__(self, dataset_path, labels_path, normalization):
    train_data = []
    h5f = h5py.File(dataset_path, 'r')
    data = h5f['data'][:].astype(np.float32)
    h5f.close()
    
    train_labels = LoadSmilesData(labels_path, normalization)
    labels_1, labels_2 = train_labels.property_train()

    for i in range(len(data)):
        train_data.append([data[i], labels_1[i], labels_2[i]])
    self.data = train_data
    
  def __len__(self):
    return len(self.data)
 
  def __getitem__(self, idx):
    return self.data[idx]