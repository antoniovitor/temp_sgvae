import nltk
import pdb
from models import grammar
from models import grammar_model
import numpy as np
import h5py
import pickle
import random
import parameters
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
random.seed(42)

def to_one_hot(smiles, MAX_LEN, NCHARS):
    """ Encode a list of smiles strings to one-hot vectors """
    assert type(smiles) == list
    prod_map = {}
    for ix, prod in enumerate(grammar.GCFG.productions()):
        prod_map[prod] = ix
    tokenize = grammar_model.get_zinc_tokenizer(grammar.GCFG)
    tokens = map(tokenize, smiles)
    parser = nltk.ChartParser(grammar.GCFG)
    parse_trees = [parser.parse(t).__next__() for t in tokens]
    productions_seq = [tree.productions() for tree in parse_trees]
    indices = [np.array([prod_map[prod] for prod in entry], dtype=int) for entry in productions_seq]
    one_hot = np.zeros((len(indices), MAX_LEN, NCHARS), dtype=np.float32)
    for i in range(len(indices)):
        num_productions = len(indices[i])
        one_hot[i][np.arange(num_productions),indices[i]] = 1.
        one_hot[i][np.arange(num_productions, MAX_LEN),-1] = 1.
    return one_hot


def smiles_list_to_one_hot(smiles):
    params = parameters.load_params()
    MAX_LEN = params['max_length']
    NCHARS = len(grammar.GCFG.productions())

    OH = np.zeros((len(smiles),MAX_LEN,NCHARS))
    for i in tqdm(range(0, len(smiles), 100)):
        onehot = to_one_hot(smiles[i:i+100], MAX_LEN, NCHARS)
        OH[i:i+100,:,:] = onehot

    return OH

# def main():
#     data = smiles_list_to_one_hot(L)

#     h5f = h5py.File('data/grammar_dataset_new.h5','w')
#     h5f.create_dataset('data', data=data)
#     h5f.close()
    
# if __name__ == '__main__':
#     main()

