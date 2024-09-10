import re
import os
import pandas as pd
import numpy as np
import graph_gen as gg
from utils import get_map_lims, m2ft, ms2kts, find_nearest_city, city_coords
from uncertainty import generate_delivery_times, uncertainty_settings

class DCScenario:

    def __init__(self, input_dir, sol_file, uncertainty=False):
        self.input_dir = input_dir
        self.sol_file = sol_file
        self.vehicle_group = re.search(r'\d+', self.sol_file).group()
        # Load customer locations
        self.customers = pd.read_csv(self.input_dir + '/tbl_locations.csv')
        self.customers.columns = self.customers.columns.str.strip()
        self.uncertainty = uncertainty
        if self.uncertainty:
            self.customers['del_unc'] = [0] + list(generate_delivery_times(
                        len(self.customers) - 2, 
                        uncertainty_settings[self.uncertainty]['mu_del'],
                        uncertainty_settings[self.uncertainty]['min_delay'])) + [0]
        else:
            self.customers['del_unc'] = np.zeros(len(self.customers))

        customer_latlons = self.customers[['latDeg', 'lonDeg']].to_numpy().tolist()
        # 4 km border for the map is sufficient
        lims = get_map_lims(customer_latlons, 4)
        self.city = find_nearest_city((np.mean((lims[0], lims[1])), 
                                np.mean((lims[2], lims[3]))), 
                                city_coords)
        # try:
        #     self.city = gg.get_city_from_bbox(lims[0], lims[1], lims[2], lims[3])
        # except gg.CityNotFoundError:
        #     print("Customers are in an unknown location")
        
        self.truckname = 'TRUCK'


    def construct_scenario(self, save_name):
        # Load all necessary implementations
        self.scen_text = "00:00:00>IMPL ACTIVEWAYPOINT TDActWp\n"
        self.scen_text += "00:00:00>IMPL AUTOPILOT TDAutoPilot\n"
        self.scen_text += "00:00:00>IMPL ROUTE TDRoute\n"
        self.scen_text += f"00:00:00>LOADGRAPH {os.getcwd()}/graphs/{self.city}.graphml\n"
        # Initiate logging with correct args to track results
        log_file = str(len(self.customers) - 1) + '_' + self.sol_file.rstrip('_Heuristic.csv') +\
                        '_DC_' + str(self.uncertainty)
        self.scen_text += f"00:00:00>LOG {log_file} {self.input_dir.split('/')[-1]}\n"
        # Extract the number of drones from the sol_file name
        M = re.search(r'_[0-9]+_([0-9]+)_', self.sol_file)[1]
        self.scen_text += f"00:00:00>DELIVER {self.vehicle_group} {M} "
        self.scen_text += (f"{self.input_dir.split('/')[-1]}")
        for index, customer in self.customers.iterrows():
            self.scen_text += f",{customer['latDeg']},{customer['lonDeg']},{customer['del_unc']}"

        destination_tolerance = 3/1852 
        self.scen_text += (
            f"\n00:{'{:02}'.format(len(self.customers)//5)}:00>"
            f"{self.truckname} ATDIST {self.customers.loc[0]['latDeg']} "
            f"{self.customers.loc[0]['lonDeg']} {destination_tolerance} "
            f"TRKDEL {self.truckname}"
                        )

        # Change directory to scenario folder
        scenariofolder = '/scenario'

        try:
            os.chdir(os.getcwd() + scenariofolder)
        except:
            if os.getcwd().split('/')[-1] == 'scenario':
                pass
            else:
                raise Exception('Scenario folder not found')

        # Save the text in a scenario file
        with open(save_name, 'w') as f:
            f.write(self.scen_text)

        os.chdir(os.getcwd().rsplit(scenariofolder, 1)[0])