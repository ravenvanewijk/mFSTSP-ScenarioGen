import os
import geopy.distance
import pandas as pd
import osmnx as ox
import numpy as np
import taxicab as tc
import re
import matplotlib.pyplot as plt
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge
from utils import kwikqdrdist, plot_linestring, reverse_linestring, \
                shift_circ_ls, simplify_graph, m2ft, ms2kts, get_map_lims

class mFSTSPRoute:
    def __init__(self, input_dir, sol_file):
        self.input_dir = input_dir
        self.sol_file = sol_file
        self.route_waypoints = {}
        self.route_lats = {}
        self.route_lons = {}
        self.data = {}
        # extract all data from solution file
        data = pd.read_csv(self.input_dir + '/' + self.sol_file)
        # metadata is first few lines
        metadata = data.head(3)
        # strip all spaces
        metadata.columns = metadata.columns.str.strip()
        # routing solution is embedded in everything after the fourth row
        solution = data.tail(-4)
        solution.columns = data.iloc[3]
        solution.columns.name = None
        solution.columns = solution.columns.str.strip()
        # string space stripping
        solution.loc[:, 'vehicleType'] = solution['vehicleType'].str.strip()
        solution.loc[:, 'startTime'] = pd.to_numeric(solution['startTime'], errors='coerce')
        solution.loc[:, 'endTime'] = pd.to_numeric(solution['endTime'], errors='coerce')
        solution.loc[:, 'startNode'] = pd.to_numeric(solution['startNode'], errors='coerce')
        solution.loc[:, 'endNode'] = pd.to_numeric(solution['endNode'], errors='coerce')
        # Store the data in the dictionary
        self.data['problemName'] = metadata['problemName'].iloc[0]
        self.data['numUAVs'] = int(metadata['numUAVs'].iloc[0]),
        self.data['requireTruckAtDepot'] = bool(metadata['requireTruckAtDepot'].iloc[0]),
        self.data['requireDriver'] = bool(metadata['requireDriver'].iloc[0]),
        self.data['solution'] = solution

        # Load customer locations
        self.customers = pd.read_csv(self.input_dir + '/tbl_locations.csv')
        self.customers.columns = self.customers.columns.str.strip()
        # Create return depot node (equal to first node (= depot))
        self.customers = pd.concat([self.customers, self.customers.iloc[[0]]], ignore_index=True)
        self.depot_return_id = self.customers.index[-1]
        # Give correct index to this node
        self.customers.at[self.depot_return_id, '% nodeID'] = self.depot_return_id

        customer_latlons = self.customers[['latDeg', 'lonDeg']].to_numpy().tolist()
        # 4 km border for the map is sufficient
        lims = get_map_lims(customer_latlons, 4)
        # Adjusted box sizes to include the entire map
        self.G = ox.graph_from_bbox(bbox=lims, network_type='drive')
        # Simplify the graph using osmnx
        self.G = simplify_graph(self.G)
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

    def construct_truckroute(self):
        # Extract all truck route nodes
        self.truckactivities = self.data['solution'][self.data['solution']['vehicleType'] == 'Truck']
        # Should already be sorted, redundancy
        self.truckactivities = self.truckactivities.sort_values('startTime')
        # Get all traveling activities
        self.truckdriving = self.truckactivities[self.truckactivities['Description'].str.startswith(\
                                                                                    ' Travel from node')]
        # Extract node sequence from traveling activities
        self.cust_nodes = self.truckdriving['Description'].str.extractall(r'(\d+)')
        self.cust_nodes = self.cust_nodes.astype(int)
        self.cust_nodes = self.cust_nodes[0].unique().tolist()
        route = None 
        # zip the customers to create pairs
        cust_pairs = zip(self.cust_nodes[:-1], self.cust_nodes[1:])

        for U, V in cust_pairs:
            custroute = []
            # Now use the taxicab package to find the shortest path between 2 customer nodes.
            # Args come in this order: 0: distance, 1: nodes 2: unfinished linepart begin 3: unfinished linepart end
            # Look up location of customer nodes U and V in the custnode df
            routepart = tc.distance.shortest_path(self.G,   (self.customers.iloc[U]['latDeg'], 
                                                        self.customers.iloc[U]['lonDeg']), 
                                                        (self.customers.iloc[V]['latDeg'], 
                                                        self.customers.iloc[V]['lonDeg']))
            
            # Use the nodes to extract all edges u, v of graph G that the vehicle completely traverses
            routepart_edges = zip(routepart[1][:-1], routepart[1][1:])

            # routepart at beginning
            custroute.append(routepart[2])
            try:
                # For every pair of edges, append the route with the Shapely LineStrings
                for u, v in routepart_edges:
                    # Some edges have this attribute embedded, when geometry is curved
                    if 'geometry' in self.G.edges[(u, v, 0)]:
                        custroute.append(self.G.edges[(u, v, 0)]['geometry'])
                    # Other edges don't have this attribute. These are straight lines between their two nodes.
                    else:
                        # So, get a straight line between the nodes and append that line piece
                        custroute.append(LineString([(self.G.nodes[u]['x'], self.G.nodes[u]['y']), 
                                                (self.G.nodes[v]['x'], self.G.nodes[v]['y'])]))
            except IndexError:
                pass
            
            try:
                # Additional check for first linepart directionality. Sometimes it might be facing the wrong way.
                # The end of the beginning (incomplete) linestring should match
                try:
                    if not custroute[1].coords[0] == routepart[2].coords[-1]:
                        # Check if flipped version does align
                        if custroute[1].coords[0] == routepart[2].coords[0]:
                            custroute[0] = reverse_linestring(custroute[0])
                        else:
                            raise Exception('Taxicab alignment Error: Coordinates of beginning LineString does not align')
                except IndexError:
                    pass
            except AttributeError:
                pass

            try:
                # Check whether final incomplete linestring is in proper direction, similar check
                try:
                    if not custroute[-1].coords[-1] == routepart[3].coords[0]:
                        # Check if flipped version does align
                        if custroute[-1].coords[-1] == routepart[3].coords[-1]:
                            custroute.append(reverse_linestring(routepart[3]))
                        else:
                            raise Exception('Taxicab alignment Error: Coordinates of final LineString does not align')
                    else:
                        custroute.append(routepart[3])
                except IndexError:
                    pass
            except AttributeError:
                pass
            # for ls in custroute:
            #     plot_linestring(ls)
            # add customer route total global route
            custroute_linestring = linemerge(custroute)  # Combine all parts into a single LineString
            self.customers.loc[U, 'Route_lat'] = custroute[0].coords[0][1]
            self.customers.loc[U, 'Route_lon'] = custroute[0].coords[0][0]
            # Merge the current custroute into the main route
            if route is None:
                route = custroute_linestring
            else:
                if route.coords[0] == custroute_linestring.coords[-1]:
                    # If a circular loop is generated, very slightly shift the last coordinates of custroute
                    # Ensures proper directionality of the route
                    custroute_linestring = shift_circ_ls(custroute_linestring)
                route = linemerge([route, custroute_linestring])

            if type(route) == MultiLineString:
                raise Exception(f'Resulting route is a MultiLineString from nodes {U} to {V}.'
                                + ' Possibly the segments do not connect properly.')

        # Now that we have the line, we can get the waypoints in LAT LON.
        # The graph is in LON LAT, but BlueSky works with LAT LON.
        self.route_waypoints = list(zip(route.xy[1], route.xy[0]))
        self.route_lats = route.xy[1]
        self.route_lons = route.xy[0]
        
        # Assign new values to nodeID, Route_lat, and Route_lon for return depot node
        # create slightly shifted depot node in customers, to account for the shift_circ_ls (preventing circular LS)
        self.customers.at[self.depot_return_id, 'Route_lat'] = self.route_lats[-1]
        self.customers.at[self.depot_return_id, 'Route_lon'] = self.route_lons[-1]

    def get_sorties(self):
        """This method generates a set of trips that are used to generate the scenarios texts for the sorties.
        Trips consist of i (launch loc), j (delivery cust loc), and k (rendezvous loc)"""

        # Get all drone activities
        droneactivities = self.data['solution'][self.data['solution']['vehicleType'] == 'UAV']
        # Filter the sorties
        sorties = droneactivities[droneactivities['Description'].str.contains('Fly to UAV customer')]
        rendezvouss = droneactivities[droneactivities['activityType'] == (' UAV travels empty')]
        rendezvouss.loc[:, 'endNode'] = rendezvouss['endNode'].astype(int)
        # Convert rendezvouss to dictionary for quicker lookup
        rendezvouss_dict = {int(row['startNode']): int(row['endNode']) for index, row in rendezvouss.iterrows()}
        self.trips = {}
        self.UAVs = sorties['vehicleID'].unique()
        for UAV in self.UAVs:
            self.trips[UAV] = []

        # Process sorties
        for index, sortie in sorties.iterrows():
            i = int(sortie['startNode'])
            j = int(sortie['endNode'])
            
            # Check if j is in rendezvouss_dict
            if j in rendezvouss_dict:
                k = rendezvouss_dict[j]
                self.trips[sortie['vehicleID']].append((i, j, k))
            else:
                raise ValueError(f"No corresponding rendezvous found for endNode {j} starting from node {i}")
    
    def get_deliveries(self):
        """This method generates a set of delivery nodes based on the data of the solution.
        Can be used to make a scenario text with it."""

        self.truckdeliveries = self.truckactivities[self.truckactivities['Status']==' Making Delivery ']
        self.delivery_nodes = self.truckdeliveries['startNode'].tolist()

    def construct_scenario(self, save_name):
        route_waypoints = self.route_waypoints
        route_lats = self.route_lats
        route_lons = self.route_lons

        i = 1 # Start at second waypoint
        turns = ['turn'] # Doesn't matter what the first waypoint is designated as, so just have it as true.
        for lat_cur, lon_cur in route_waypoints[1:-1]:
            # Get the previous and the next waypoint
            lat_prev, lon_prev = route_waypoints[i-1]
            lat_next, lon_next = route_waypoints[i+1]
            # Get the angle
            a1, _ = kwikqdrdist(lat_prev,lon_prev,lat_cur,lon_cur)
            a2, _ = kwikqdrdist(lat_cur,lon_cur,lat_next,lon_next)
            angle=abs(a2-a1)
            if angle>180:
                angle=360-angle
            # In general, we noticed that we don't need to slow down if the turn is smaller than 25 degrees
            # If the angle is larger, then a more severe slowdown is required
            #  However, this will depend on the cruise speed of the vehicle.
            if angle > 35:
                turns.append('sharpturn')
            elif angle > 25:
                turns.append('turn')
            else:
                turns.append('straight')
            i += 1

        # Let the vehicle slow down for the depot
        turns.append(True)
        # Add some commands to pan to the correct location and zoom in, and use the modified active wp package.
        self.scen_text = "00:00:00>IMPL ACTIVEWAYPOINT TDActWp\n"
        self.scen_text += "00:00:00>IMPL AUTOPILOT TDAutoPilot\n"
        self.scen_text += "00:00:00>IMPL ROUTE TDRoute\n"
        self.scen_text += f'00:00:00>PAN {route_lats[0]} {route_lons[0]}\n' # Pan to the origin
        self.scen_text += "00:00:00>ZOOM 50\n" # Zoom in
        self.scen_text += f"00:00:00>LOG {self.input_dir.split('/')[-1]} {self.sol_file}\n"

        # Get angle of first direction to ensure correct orientation
        achdg, _ = kwikqdrdist(route_lats[0], route_lons[0], route_lats[1], route_lons[1])

        # CRE command takes Aircraft ID, Aircraft Type, Lat, Lon, hdg, alt[ft], spd[kts]
        trkid = 'TRUCK'
        actype = 'Truck'
        acalt = 0 # ft, ground altitude
        acspd = 0 # start with 0 speed from depot
        self.scen_text += f'00:00:00>CRE {trkid} {actype} {route_lats[0]} {route_lons[0]} {achdg} {acalt} {acspd}\n'
        self.scen_text += f'00:00:00>COLOUR {trkid} RED'

        # After creating it, we want to add all the waypoints. We can do that using the ADDTDWAYPOINTS command.
        # ADDTDWAYPOINTS can chain waypoint data in the following way:
        # ADDTDWAYPOINTS ACID LAT LON ALT SPD Turn? TurnSpeed
        # SPD here can be set as the cruise speed so the vehicle knows how fast to go
        # cruise_spd = 25 #kts
        cruise_alt = acalt # Keep it constant throughout the flight
        # Turn speed of 5 kts usually works well
        turn_spd = 10 #kts
        sharpturn_spd = 5 #kts
        # Initiate adddtwaypoints command
        self.scen_text += f'00:00:00>ADDTDWAYPOINTS {trkid}' # First add the root of the command
        # Loop through waypoints
        # Refresh command after 100 iterations, BlueSky cannot handle everything in one go
        i = 0
        j = 0
        self.scn_lim = 400
        for wplat, wplon, turn in zip(route_lats, route_lons, turns):
            # Check if this waypoint is a turn
            if turn == 'turn' or turn == 'sharpturn':
                wptype = 'TURNSPD'
                wp_turnspd = turn_spd if turn == 'turn' else sharpturn_spd
            else:
                wptype = 'FLYBY'
                # Doesn't matter what we pick here, as long as it is assigned. 
                # Will be ignored
                wp_turnspd = turn_spd
            if i > self.scn_lim:
                j += 1
                self.scen_text += f'\n00:00:00>ADDTDWAYPOINTS {trkid}'
                i = 0
            i += 1
            # Add the text for this waypoint. It doesn't matter if we always add a turn speed, as BlueSky will
            # ignore it if the wptype is set as FLYBY
            self.scen_text += f',{wplat},{wplon},{cruise_alt},,{wptype},{wp_turnspd}'

        # Add delivery commands
        self.delivery_scen(trkid)
        # Add sortie commands
        for UAV in self.UAVs:
            self.sortie_scen(trkid, UAV)
        self.add_truck_timing(trkid)
        self.scen_text += '\n'
        # Check whether depot is an operation point. If so, delay the LNAV and VNAV
        # LNAV and VNAV will cause the truck to start moving before the operation has taken place
        if 'ADDOPERATIONPOINTS TRUCK, ' + str(route_lats[0]) + '/' + str(route_lons[0]) not in self.scen_text:
            # Enable vertical and horizontal navigation manually when first wp is not an operation
            self.scen_text += f'00:00:00>LNAV {trkid} ON\n'
            self.scen_text += f'00:00:00>VNAV {trkid} ON\n'
        # Turn trail on, tracing for testing
        self.scen_text += f'00:00:00>TRAIL ON'

        # Add a newline at the end of the addtdwaypoints command
        self.scen_text += '\n'

        # Delete truck at route end
        # It is considered it arrived if it is within 3 metres of the destination, converted to nautical miles.
        destination_tolerance = 3/1852 
        self.scen_text += f'00:{"{:02}".format((j * self.scn_lim + i)//100)}:00>{trkid} ATDIST {route_lats[-1]} {route_lons[-1]} {destination_tolerance} TRKDEL {trkid}\n'

        # Change directory to scenario folder
        try:
            os.chdir(os.getcwd() + '/scenario')
        except:
            raise Exception('Scenario folder not found')

        # Save the text in a scenario file
        with open(save_name, 'w') as f:
            f.write(self.scen_text)

    def delivery_scen(self, trkid):
        specs = self.vehicle_data[self.vehicle_data['% vehicleID'] == '1']
        for node in self.delivery_nodes:
            self.scen_text += f"\n00:00:00>ADDOPERATIONPOINTS {trkid} {self.customers.loc[node]['Route_lat']}/" + \
                            f"{self.customers.loc[node]['Route_lon']}, DELIVERY, {specs['serviceTime [sec]'].item()}"

    def sortie_scen(self, trkid, UAVnumber):
        """Function that writes the text of sorties on top of existing text. This should come after the waypoints have
        been added, otherwise the sorties will be added to the stack before the waypoints exist.
        args: type, description
        - trkid: str, identifyer of the truck that will perform the operation
        - UAVnumber: str, identifyer of the UAV that will be dispatched"""
        UAVtrips = self.trips[UAVnumber]
        UAV_type = f'M{self.vehicle_group}'
        specs = self.vehicle_data[self.vehicle_data['% vehicleID'] == UAVnumber]
        for node in self.cust_nodes:
            matching_trip = next((trip for trip in UAVtrips if trip[0] == node), None)
            if matching_trip:
                i = matching_trip[0]
                j = matching_trip[1]
                k = matching_trip[2]
                i_coords = f"{self.customers.loc[i]['Route_lat']}/{self.customers.loc[i]['Route_lon']}"
                j_lat = f"{self.customers.loc[j]['latDeg']}"
                j_lon = f"{self.customers.loc[j]['lonDeg']}"
                k_coords = f"{self.customers.loc[k]['Route_lat']}/{self.customers.loc[k]['Route_lon']}"
                # Truck is ID one so subtract 1 from UAV number
                # Use data in specsheet to add the command to the stack
                self.scen_text += f"\n00:00:00>ADDOPERATIONPOINTS {trkid}, {i_coords}, SORTIE, "+\
                            f"{specs['launchTime [sec]'].item()}, {UAV_type}, {int(UAVnumber) - 1}, {j_lat}, "+\
                            f"{j_lon}, {k_coords}, {m2ft(specs['cruiseAlt [m]'].item())}, "+\
                            f"{ms2kts(specs['cruiseSpeed [m/s]'].item())} {specs['serviceTime [sec]'].item()}, " +\
                            f"{specs['recoveryTime [sec]'].item()}"

    def add_truck_timing(self, trkid):
        self.scen_text += f'\n00:00:00>TDRTAs {trkid}'
        cust_idx = 1
        for truckwp in self.cust_nodes:
            if truckwp == 0:
                continue
            target_custtime = self.truckdriving[self.truckdriving['endNode'] == truckwp]['endTime'].values[0]
            lat = self.customers.loc[self.cust_nodes[cust_idx]]['Route_lat']
            lon = self.customers.loc[self.cust_nodes[cust_idx]]['Route_lon']

            self.scen_text += f", {lat}/{lon},{target_custtime}"
            cust_idx += 1