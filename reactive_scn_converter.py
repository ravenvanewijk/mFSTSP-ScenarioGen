import re
import os
import pandas as pd

class ReactiveScenario:

    def __init__(self, input_dir, sol_file):
        self.input_dir = input_dir
        self.sol_file = sol_file
        # Load customer locations
        self.customers = pd.read_csv(self.input_dir + '/tbl_locations.csv')
        self.customers.columns = self.customers.columns.str.strip()
        # Retreive vehicle data from input and sol file
        self.get_vehicle_data()

    def get_vehicle_data(self):
        """Load the vehicle data that corresponds with the solution file"""
        # Load vehicle data from CSV
        self.vehicle_group = re.search(r'\d+', self.sol_file).group()
        self.vehicle_data = pd.read_csv(self.input_dir.rsplit('/', 1)[0] + '/' + f"tbl_vehicles_{self.vehicle_group}.csv")
        # Set the correct row as column names
        self.vehicle_data.columns = self.vehicle_data.iloc[0]
        # Drop the column that has been set as column names
        self.vehicle_data = self.vehicle_data.drop(self.vehicle_data.index[0])

    def construct_scenario(self, save_name):
        self.scen_text = "00:00:00>IMPL ACTIVEWAYPOINT TDActWp\n"
        self.scen_text += "00:00:00>IMPL AUTOPILOT TDAutoPilot\n"
        self.scen_text += "00:00:00>IMPL ROUTE TDRoute\n"
        self.scen_text += f"00:00:00>LOG {self.input_dir.split('/')[-1]} {self.sol_file}\n"
        self.scen_text += f"00:00:00>REACT {self.vehicle_group}"
        for index, customer in self.customers.iterrows():
            self.scen_text += f",{customer['latDeg']},{customer['lonDeg']},{customer['parcelWtLbs']}"

        # Change directory to scenario folder
        try:
            os.chdir(os.getcwd() + '/scenario')
        except:
            if os.getcwd().split('/')[-1] == 'scenario':
                pass
            else:
                raise Exception('Scenario folder not found')

        # Save the text in a scenario file
        with open(save_name, 'w') as f:
            f.write(self.scen_text)