import args_parser 
import parameters
from bootstrap import bootstrap
from pathlib import Path
from analyses.error_distribution import analyze_error_distribution
from analyses.bond_lengths import run_bond_length_analysis

########## ARGUMENTS
args = args_parser.get_args()

########## GLOBAL VARIABLES
params = parameters.load_params()

########## MAIN
def main():
    # TODO: include filter to select method to be executed
    # analyze_error_distribution()
    run_bond_length_analysis()


if __name__ == "__main__":
    bootstrap()
    main()