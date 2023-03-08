import argparse

# Parsing the arguments
parser = argparse.ArgumentParser()

parser.add_argument('--path', help='Path to the weights file', type=str)
parser.add_argument('--plot', help='Plot the latent space', action='store_true')
parser.add_argument('--evaluation', help='Evaluate the model on prior validity, novelty and uniqueness', action='store_true')
parser.add_argument('--train_property', help='Train and evaluate the property prediction model', action='store_true')
parser.add_argument('--test_property', help='Test the property prediction model in a hold-out set', action='store_true')
parser.add_argument('--split', help='Split set to be used', type=str)

args = parser.parse_args()

def get_args():
    return args


# TODO: finish args parser class 
args_dict = {
    'path': {
        'help': 'Path to the training folder',
        'type': str
    },
    'plot': {
        'help': 'Plot the latent space',
        'action': 'store_true',
    },
    'train_property': {
        'help': 'Train and evaluate the property prediction model',
        'action': 'store_true',
    },
    'test_property': {
        'help': 'Test the property prediction model in a hold-out set',
        'action': 'store_true',
    },
    'split': {
        'help': 'Split set to be used (train, val or test)',
        'type': str
    },
    'subset': {
        'help': 'Subset to be used (anions or cations)',
        'type': str
    },
}

class ArgsParser():
    def requires(self, require_list):
        self.require_list = require_list
        return self

    def get_args(self):
        parser = argparse.ArgumentParser()
        for arg in self.require_list:
            options = args_dict[arg]
            parser.add_argument() # TODO: pass parameters

        return args