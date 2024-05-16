from scenario_converter import mFSTSPRoute
import argparse
import os

def main(input_dir, sol_file, solutions_name = 'tbl_solutions'):

    if sol_file.upper() == 'ALL':
        files = os.listdir(input_dir)
        solutions_files = [filename for filename in files if solutions_name in filename]
    else:
        solutions_files = [sol_file]
    for sol in solutions_files:
        # Initialize the mFSTSPRoute object with the given parameters
        routes = mFSTSPRoute(input_dir, sol)
        
        # Perform operations
        routes.construct_truckroute()
        routes.get_deliveries()
        routes.get_sorties()
        routes.construct_scenario(sol.split('.')[0] + '.scn')

# if __name__ == "__main__":
#     # Set up argument parser
#     parser = argparse.ArgumentParser(description="Convert mFSTSPRoute operations to scn file")
#     parser.add_argument('input_dir', type=str, help="Path to the input directory.")
#     parser.add_argument('sol_file', type=str, help="Name of the solution file.")
#     main(args.input_dir, args.sol_file)

main("/Users/raven/Documents/TU/MSc/Thesis/Code/mFSTSP/Problems/20170606T123331779163",
        "tbl_solutions_103_1_Heuristic.csv")