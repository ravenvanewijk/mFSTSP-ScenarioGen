import re
import os
import pandas as pd
import numpy as np
import graph_gen as gg
from utils import get_map_lims, m2ft, ms2kts, find_nearest_city, city_coords
from uncertainty import generate_delivery_times, uncertainty_settings, \
                        generate_drone_speed

class DCScenario:

    def __init__(self, input_dir, sol_file, uncertainty=False):
        self.input_dir = input_dir
        self.sol_file = sol_file
        self.vehicle_group = re.search(r'\d+', self.sol_file).group()
        # Retreive vehicle data from input and sol file
        self.get_vehicle_data()
        specs = self.vehicle_data[self.vehicle_data['% vehicleID'] == '2']
        self.cruise_speed = ms2kts(specs['cruiseSpeed [m/s]'].item())
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

    def get_vehicle_data(self):
        """Load the vehicle data that corresponds with the solution file"""
        # Load vehicle data from CSV
        self.vehicle_group = re.search(r'\d+', self.sol_file).group()
        self.vehicle_data = pd.read_csv(self.input_dir.rsplit('/', 1)[0] + '/' + f"tbl_vehicles_{self.vehicle_group}.csv")
        # Set the correct row as column names
        self.vehicle_data.columns = self.vehicle_data.iloc[0]
        # Drop the column that has been set as column names
        self.vehicle_data = self.vehicle_data.drop(self.vehicle_data.index[0]) 

    def construct_scenario(self):
        # Load all necessary implementations
        self.scen_text = "00:00:00>IMPL ACTIVEWAYPOINT TDActWp\n"
        self.scen_text += "00:00:00>IMPL AUTOPILOT TDAutoPilot\n"
        self.scen_text += "00:00:00>IMPL ROUTE TDRoute\n"
        self.scen_text += f"00:00:00>LOADGRAPH ../{os.getcwd().rsplit('/')[-1]}/graphs/{self.city}.graphml\n"
        # Initiate logging with correct args to track results
        log_file = str(len(self.customers) - 1) + '_' + self.city + '_' + self.sol_file.rstrip('_Heuristic.csv') +\
                        '_DC_' + str(self.uncertainty)
        self.scen_text += f"00:00:00>LOG {log_file} {self.input_dir.split('/')[-1]}\n"
        # Extract the number of drones from the sol_file name
        M = re.search(r'_[0-9]+_([0-9]+)_', self.sol_file)[1]
        self.scen_text += f"00:00:00>UNCERTAINTY {bool(self.uncertainty)} \n"
        self.scen_text += f"00:00:00>DRONEUNC "
        if self.uncertainty:
            spd_vars = generate_drone_speed(uncertainty_settings[self.uncertainty]['spd_change_prob'],
                                uncertainty_settings[self.uncertainty]['spd_change_mag'], length=len(self.customers)*int(M))
            self.scen_text += ", ".join(str(v) for v in spd_vars)
        self.scen_text += "\n"
        self.scen_text += f"00:00:00>DELIVER {self.vehicle_group} {self.cruise_speed} {M} "
        self.scen_text += (f"{self.input_dir.split('/')[-1]}")
        for index, customer in self.customers.iterrows():
            self.scen_text += f",{customer['latDeg']},{customer['lonDeg']},{customer['del_unc']}"

        approx_max_endtime = 60*60*len(self.customers) // 4

        if self.uncertainty:
            time = uncertainty_settings[self.uncertainty]['stop_interval']
            while time < approx_max_endtime:
                minutes, seconds = divmod(time, 60)
                hours, minutes = divmod(minutes, 60)
                self.scen_text += (
                f"\n{'{:02}'.format(hours)}:{'{:02}'.format(minutes)}:{'{:02}'.format(seconds)}>"
                f"ADDOPERATIONPOINTS {self.truckname} CURLOC STOP "
                f"{uncertainty_settings[self.uncertainty]['stop_length']}"
                            )
                time += uncertainty_settings[self.uncertainty]['stop_interval']

        # Change directory to scenario folder
        scenariofolder = '/scenario'

        try:
            os.chdir(os.getcwd() + scenariofolder)
        except:
            if os.getcwd().split('/')[-1] == 'scenario':
                pass
            else:
                raise Exception('Scenario folder not found')

        save_dir = self.input_dir.rsplit('Problems/')[-1]
        if not os.path.exists(save_dir):
            # Create the directory if it doesn't exist
            os.makedirs(save_dir)
        
        # Change the current working directory to the input_dir
        os.chdir(save_dir)

        # Save the text in a scenario file
        with open(log_file + '.scn', 'w') as f:
            f.write(self.scen_text)

        os.chdir(os.getcwd().rsplit(scenariofolder, 1)[0])