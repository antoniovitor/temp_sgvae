import pandas as pd
from pathlib import Path
import h5py
import numpy as np
import utils.make_dataset_grammar as dataset_maker

class Dataset():
    def __init__(self, subset, properties) -> None:
        self.subset = subset
        self.properties = properties
        self.base_path = Path(f'datasource/ions/{subset}/')
        self.sets = {}
        self.encoded_sets = {}

        # for split in ['train', 'val']:
        #     self.process_subset(split)

    def process_subset(self, split):
        dest_path = self.base_path / f'{self.subset}.{split}.grammar.h5'
        if dest_path.is_file():
            return        
        
        print(f'Processing {split} partition of the dataset with {self.subset}.')
        smiles = self.get_split(split).loc[:, 'smiles'].to_list()

        data = dataset_maker.smiles_list_to_one_hot(smiles)
        with h5py.File(dest_path,'w') as file:
            file.create_dataset('data', data=data)

    def get_split(self, split):
        if split not in self.sets:
            self.sets[split] = pd.read_csv(self.base_path / f'{self.subset}.{split}.csv')
        return self.sets[split]

    def get_splits(self, splits):
        return [self.get_split(split) for split in splits]

    def get_encoded_split(self, split):
        self.process_subset(split)

        if split not in self.encoded_sets:
            labels = self.get_split(split).loc[:, self.properties].astype(np.float64).to_numpy()

            source_path = self.base_path / f'{self.subset}.{split}.grammar.h5'
            with h5py.File(source_path, 'r') as file:
                smiles_one_hot = file['data'][:].astype(np.float32)
        
            self.encoded_sets[split] = [[one_hot, *labels[i]] for i, one_hot in enumerate(smiles_one_hot)]

        return self.encoded_sets[split]

    def get_encoded_splits(self, splits):
        return tuple([self.get_encoded_split(split) for split in splits])