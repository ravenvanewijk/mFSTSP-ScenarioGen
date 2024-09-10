from scenario_converter import mFSTSPRoute
from DC_scn_converter import DCScenario
from uncertainty import uncertainty_settings
import argparse
import os
import re
import osmnx as ox
import tqdm
import multiprocessing as mp

mp.set_start_method('fork')

def main(input_dir='ALL', sol_file='ALL', uncertainty=False, solutions_name='tbl_solutions',
            file_pattern=r'^\d{8}T\d{6}\d{6}$', problem_dir='../mFSTSP/Problems/',
            drone_type='101'):
    # disable caching, reduce clutter
    ox.config(use_cache=False)  
    input_arr = []
    if uncertainty:
        if uncertainty.lower() not in list(uncertainty_settings.keys()):
            raise ValueError('Select valid uncertainty level: ' +\
                                f'{list(uncertainty_settings.keys())}')
    
    if input_dir == 'ALL':
        problem_folder = '../mFSTSP/Problems'
        all_files = os.listdir(problem_folder)
        input_dirs = [os.path.join(problem_dir, filename) for filename in \
                        all_files if re.match(file_pattern, filename)]
    
    else:
        input_dirs = [input_dir]

    for input_dir in input_dirs:
        if sol_file.upper() == 'ALL':
            files = os.listdir(input_dir)
            solutions_files = [filename for filename in files if 
                                solutions_name in filename and
                                drone_type in filename and
                                'Heuristic' in filename]
        else:
            solutions_files = [sol_file]
        for sol in solutions_files:
            print(input_dir, sol)
            input_arr.append([input_dir, sol, uncertainty])

    return input_arr

def make_scen(input_):
    input_dir, sol, uncertainty = input_
    routes = mFSTSPRoute(input_dir, sol, uncertainty)
    # Perform operations
    routes.construct_truckroute()
    routes.get_deliveries()
    routes.get_sorties()
    routes.construct_scenario(sol.split('.')[0] + '.scn')

    react = DCScenario(input_dir, sol, uncertainty)
    react.construct_scenario(sol.split('.')[0] + '_DC.scn')

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert mFSTSPRoute operations to scn file")
    parser.add_argument('input_dir', type=str, nargs='?', \
            default='ALL', help="Path to the input directory.")
    parser.add_argument('sol_file', type=str, nargs='?', \
            default='ALL', help="Name of the solution file.")
    # Parse the arguments
    args = parser.parse_args()
    # args.input_dir = '../mFSTSP/Problems/20170607T111828707735'
    # args.sol_file = 'tbl_solutions_102_3_Heuristic.csv'
    input_arr = main(args.input_dir, args.sol_file)

    # make_scen(input_arr[0])
    
    with mp.Pool(16) as p:
        results = list(tqdm.tqdm(p.imap(make_scen, input_arr), total = len(input_arr)))