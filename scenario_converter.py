import os
import geopy.distance
import math
import pandas as pd
import osmnx as ox
import numpy as np
import taxicab as tc
import matplotlib.pyplot as plt
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge, transform
from utils import kwikqdrdist, plot_linestring, reverse_linestring, shift_circ_ls, simplify_graph

class mFSTSPRoute:
    def __init__(self, input_dir, sol_file):
        self.route_waypoints = {}
        self.route_lats = {}
        self.route_lons = {}
        self.data = {}
        # extract all data from solution file
        data = pd.read_csv(input_dir + '/' + sol_file)
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
        # Store the data in the dictionary
        self.data['problemName'] = metadata['problemName'].iloc[0]
        self.data['numUAVs'] = int(metadata['numUAVs'].iloc[0]),
        self.data['requireTruckAtDepot'] = bool(metadata['requireTruckAtDepot'].iloc[0]),
        self.data['requireDriver'] = bool(metadata['requireDriver'].iloc[0]),
        self.data['solution'] = solution

        # Load customer locations
        self.customers = pd.read_csv(input_dir + '/tbl_locations.csv')
        # Create return depot node (equal to first node (= depot))
        self.customers = pd.concat([self.customers, self.customers.iloc[[0]]], ignore_index=True)
        self.depot_return_id = self.customers.index[-1]
        # Give correct index to this node
        self.customers.at[self.depot_return_id, '% nodeID'] = self.depot_return_id

        # 4 km border for the map is sufficient
        lims = self.get_map_lims(4)
        # Adjusted box sizes to include the entire map
        self.G = ox.graph_from_bbox(bbox=lims, network_type='drive')
        # Simplify the graph using osmnx
        self.G = simplify_graph(self.G)

    def get_map_lims(self, margin, unit='km'):
        """Function to get map limits where all customers fit in.
        Args: type, description
        margin: float or int, margin for borders of the map
        unit: string, unit for provided margin"""
        
        # Conversion factors
        unit_conversion = {
            'km': 1,
            'm': 1 / 1000,             # 1000 meters in a kilometer
            'mi': 1.60934,             # 1 mile is approximately 1.60934 kilometers
            'nm': 1.852                # 1 nautical mile is approximately 1.852 kilometers
        }

        # Convert margin to kilometers
        if unit in unit_conversion:
            margin_km = margin * unit_conversion[unit]
        else:
            raise ValueError(f"Unsupported unit: {unit}. Use 'km', 'm', 'mi', or 'nm'.")

        # Get the max and min latitudes and longitudes
        latmax = self.customers[' latDeg'].max()
        latmin = self.customers[' latDeg'].min()
        lonmax = self.customers[' lonDeg'].max()
        lonmin = self.customers[' lonDeg'].min()

        # Convert margin from km to degrees
        lat_margin_deg = margin_km / 111.32  # 1 degree latitude is approximately 111.32 km
        avg_lat = (latmax + latmin) / 2
        lon_margin_deg = margin_km / (111.32 * math.cos(math.radians(avg_lat)))  # Adjust longitude margin by latitude

        # Calculate the new limits
        box_latmax = latmax + lat_margin_deg
        box_latmin = latmin - lat_margin_deg
        box_lonmax = lonmax + lon_margin_deg
        box_lonmin = lonmin - lon_margin_deg

        # Return the coordinates as a tuple
        return (box_latmax, box_latmin, box_lonmax, box_lonmin)

    def construct_truckroute(self):
        # Extract all truck route nodes
        self.truckactivities = self.data['solution'][self.data['solution']['vehicleType'] == 'Truck']
        # Should already be sorted, redundancy
        self.truckactivities = self.truckactivities.sort_values('startTime')
        # Get all traveling activities
        truckdriving = self.truckactivities[self.truckactivities['Description'].str.startswith(' Travel from node')]
        # Extract node sequence from traveling activities
        self.cust_nodes = truckdriving['Description'].str.extractall(r'(\d+)')
        self.cust_nodes = self.cust_nodes.astype(int)
        self.cust_nodes = self.cust_nodes[0].unique().tolist()
        route = None 
        # zip the customers to create pairs
        cust_pairs = zip(self.cust_nodes[:-1], self.cust_nodes[1:])

        for U, V in cust_pairs:
            print(U,V)
            custroute = []
            # Now use the taxicab package to find the shortest path between 2 customer nodes.
            # Args come in this order: 0: distance, 1: nodes 2: unfinished linepart begin 3: unfinished linepart end
            # Look up location of customer nodes U and V in the custnode df
            # ____ figure out if this uses distance (length) of edges.
            # ____ find out about one way streets! does taxicab take this into account?
            # ____ if it's time, need for additional extra step. Time to cross edge. new weights. use that as new weight
            routepart = tc.distance.shortest_path(self.G,   (self.customers.iloc[U][' latDeg'], 
                                                        self.customers.iloc[U][' lonDeg']), 
                                                        (self.customers.iloc[V][' latDeg'], 
                                                        self.customers.iloc[V][' lonDeg']))
            
            # Use the nodes to extract all edges u, v of graph G that the vehicle completely traverses
            routepart_edges = zip(routepart[1][:-1], routepart[1][1:])

            # routepart at beginning
            custroute.append(routepart[2])
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
                    
            # Additional check for first linepart directionality. Sometimes it might be facing the wrong way.
            # The end of the beginning (incomplete) linestring should match
            if not custroute[1].coords[0] == routepart[2].coords[-1]:
                # Check if flipped version does align
                if custroute[1].coords[0] == routepart[2].coords[0]:
                    custroute[0] = reverse_linestring(custroute[0])
                else:
                    raise Exception('Taxicab alignment Error: Coordinates of beginning LineString does not align')

            # Check whether final incomplete linestring is in proper direction, similar check
            if not custroute[-1].coords[-1] == routepart[3].coords[0]:
                # Check if flipped version does align
                if custroute[-1].coords[-1] == routepart[3].coords[-1]:
                    custroute.append(reverse_linestring(routepart[3]))
                else:
                    raise Exception('Taxicab alignment Error: Coordinates of final LineString does not align')
            else:
                custroute.append(routepart[3])
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
        self.trips = []

        # Process sorties
        for index, sortie in sorties.iterrows():
            i = int(sortie['startNode'])
            j = int(sortie['endNode'])
            
            # Check if j is in rendezvouss_dict
            if j in rendezvouss_dict:
                k = rendezvouss_dict[j]
                self.trips.append((i, j, k))
            else:
                raise ValueError(f"No corresponding rendezvous found for endNode {j} starting from node {i}")
    
    def get_deliveries(self):
        """This method generates a set of delivery nodes based on the data of the solution.
        Can be used to make a scenario text with it."""

        self.truckdeliveries = self.truckactivities[self.truckactivities['Status']==' Making Delivery ']
        self.delivery_nodes = self.truckdeliveries['startNode'].str.strip().astype(int).tolist()

    def construct_scenario(self, save_name, sorties=False):
        route_waypoints = self.route_waypoints
        route_lats = self.route_lats
        route_lons = self.route_lons

        i = 1 # Start at second waypoint
        turns = [True] # Doesn't matter what the first waypoint is designated as, so just have it as true.
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
            #  However, this will depend on the cruise speed of the vehicle.
            if angle > 25:
                turns.append(True)
            else:
                turns.append(False)
            i += 1

        # Let the vehicle slow down for the depot
        turns.append(True)

        # Add some commands to pan to the correct location and zoom in, and use the modified active wp package.
        self.scen_text = "00:00:00>IMPL ACTIVEWAYPOINT TDActWp"
        self.scen_text += f'00:00:00>PAN {route_lats[0]} {route_lons[0]}\n' # Pan to the origin
        self.scen_text += '00:00:00>ZOOM 50\n\n' # Zoom in

        # Get angle of first direction to ensure correct orientation
        achdg, _ = kwikqdrdist(route_lats[0], route_lons[0], route_lats[1], route_lons[1])

        # CRE command takes Aircraft ID, Aircraft Type, Lat, Lon, hdg, alt[ft], spd[kts]
        trkid = 'TRUCK'
        actype = 'Truck'
        acalt = 0 # ft, ground altitude
        acspd = 0 # start with 0 speed from depot
        self.scen_text += f'00:00:00>CRE {trkid} {actype} {route_lats[0]} {route_lons[0]} {achdg} {acalt} {acspd}\n'

        # After creating it, we want to add all the waypoints. We can do that using the ADDTDWAYPOINTS command.
        # ADDTDWAYPOINTS can chain waypoint data in the following way:
        # ADDTDWAYPOINTS ACID LAT LON ALT SPD Turn? TurnSpeed
        # SPD here can be set as the cruise speed so the vehicle knows how fast to go
        cruise_spd = 25 #kts
        cruise_alt = acalt # Keep it constant throughout the flight
        # Turn speed of 5 kts usually works well
        turn_spd = 5 #kts

        # Initiate adddtwaypoints command
        self.scen_text += f'00:00:00>ADDTDWAYPOINTS {trkid}' # First add the root of the command
        # Loop through waypoints
        # Refresh command after 100 iterations, BlueSky cannot handle everything in one go
        i = 0
        j = 0
        self.scn_lim = 400
        for wplat, wplon, turn in zip(route_lats, route_lons, turns):
            # Check if this waypoint is a turn
            if turn:
                wptype = 'TURNSPD'
            else:
                wptype = 'FLYBY'
            if i > self.scn_lim:
                j += 1
                self.scen_text += f'\n00:00:00>ADDTDWAYPOINTS {trkid}'
                i = 0
            i += 1
            # Add the text for this waypoint. It doesn't matter if we always add a turn speed, as BlueSky will
            # ignore it if the wptype is set as FLYBY
            self.scen_text += f',{wplat},{wplon},{cruise_alt},{cruise_spd},{wptype},{turn_spd}'

        # Add delivery commands
        self.delivery_scen(trkid)
        # Add sortie commands
        self.sortie_scen(trkid)
        self.scen_text += '\n'
        # Check whether depot is an operation point. If so, delay the LNAV and VNAV
        # LNAV and VNAV will cause the truck to start moving before the operation has taken place
        D = "01" if 'ADDOPERATIONPOINTS TRUCK, ' + str(route_lats[0]) + '/' + str(route_lons[0]) \
            in self.scen_text else "00"
        # Enable vertical and horizontal navigation after first set of wps and operation points iteration
        self.scen_text += f'00:00:{D}>LNAV {trkid} ON\n'
        self.scen_text += f'00:00:{D}>VNAV {trkid} ON\n'
        # Turn trail on, tracing for testing
        self.scen_text += f'00:00:{D}>TRAIL ON'

        # Add a newline at the end of the addtdwaypoints command
        self.scen_text += '\n'

        # Delete truck at route end
        # It is considered it arrived if it is within 3 metres of the destination, converted to nautical miles.
        destination_tolerance = 3/1852 
        self.scen_text += f'00:{"{:02}".format((j * self.scn_lim + i)//200)}:00>{trkid} ATDIST {route_lats[-1]} {route_lons[-1]} {destination_tolerance} TRKDEL {trkid}\n'

        # Change directory to scenario folder
        try:
            os.chdir(os.getcwd() + '/scenario')
        except:
            raise Exception('Scenario folder not found')

        # Save the text in a scenario file
        with open(save_name, 'w') as f:
            f.write(self.scen_text)

    def delivery_scen(self, trkid):
        for node in self.delivery_nodes:
            self.scen_text += f"\n00:00:00>ADDOPERATIONPOINTS {trkid} {self.customers.loc[node]['Route_lat']}/" + \
                            f"{self.customers.loc[node]['Route_lon']}, DELIVERY, 5"

    def sortie_scen(self, trkid):
        """Function that writes the text of sorties on top of existing text. This should come after the waypoints have
        been added, otherwise the sorties will be added to the stack before the waypoints exist.
        args: type, description
        - trkid: str, identifyer of the truck that will perform the operation"""
        for node in self.cust_nodes:
            matching_trip = next((trip for trip in self.trips if trip[0] == node), None)
            if matching_trip:
                i = matching_trip[0]
                j = matching_trip[1]
                k = matching_trip[2]
                i_coords = f"{self.customers.loc[i]['Route_lat']}/{self.customers.loc[i]['Route_lon']}"
                j_lat = f"{self.customers.loc[j][' latDeg']}"
                j_lon = f"{self.customers.loc[j][' lonDeg']}"
                k_coords = f"{self.customers.loc[k]['Route_lat']}/{self.customers.loc[k]['Route_lon']}"
                self.scen_text += f"\n00:00:00>ADDOPERATIONPOINTS {trkid}, {i_coords}, SORTIE, 5, M600, "
                self.scen_text += f"{j_lat}, {j_lon}, {k_coords}, 100, 25"