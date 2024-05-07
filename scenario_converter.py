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

class mFSTSPRoute:
    def __init__(self, instance_path, sol_type = 'ALL', solutions_name = 'tbl_solutions'):
        self.route_waypoints = {}
        self.route_lats = {}
        self.route_lons = {}

        files = os.listdir(instance_path)
        solutions_files = [filename for filename in files if solutions_name in filename]
        self.data = {}
        solution_file = 'tbl_solutions_103_1_Heuristic.csv'
        if solution_file not in solutions_files:
            raise Exception(f'File {solution_file} not found')
            
        # extract all data from solution file
        data = pd.read_csv(instance_path + '/' + solution_file)
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
        self.customers = pd.read_csv(instance_path + '/tbl_locations.csv')

        # Find nearest city to the customers, used to identify the city where the scenario is going to be
        nearest_city = self.get_nearest_city((self.customers.iloc[0][' latDeg'], self.customers.iloc[0][' lonDeg']))

        if nearest_city == 'Buffalo, NY, USA':
            # get customer limits. polygon of customers. instead of city map
            #________FIX_______
            lims = (43.027642, 42.831743, -78.698845, -78.949956)
        else:
            raise Exception("This city is not supported yet, choose another type")
        
        # This is too small:
        # self.G = ox.graph_from_place(nearest_city, network_type='drive')
        # Adjusted box sizes to include the entire map
        self.G = ox.graph_from_bbox(bbox=lims, network_type='drive')

    def get_nearest_city(self, cust_coords):
        """Find nearest city. Either Seattle or Buffalo.
        arg: type, description
        cust_coords: tuple, coordinates of single customer in instance
        """
        dist = np.inf
        idx = np.inf
        cities = ['Seattle, Washington, USA', 'Buffalo, NY, USA']
        for index, city in enumerate(cities):
            city_centre = ox.geocode_to_gdf(city).centroid[0]
            lat, lon = city_centre.y, city_centre.x
            if dist > geopy.distance.geodesic(cust_coords, (lat, lon)):
                dist = geopy.distance.geodesic(cust_coords, (lat, lon))
                idx = index
        return cities[idx]

    def construct_truckroute(self):
        # Extract all truck route nodes
        truckactivities = self.data['solution'][self.data['solution']['vehicleType'] == 'Truck']
        # Should already be sorted, redundancy
        truckactivities = truckactivities.sort_values('startTime')
        # Get all traveling activities
        truckdriving = truckactivities[truckactivities['Description'].str.startswith(' Travel from node')]
        # Extract node sequence from traveling activities
        self.cust_nodes = truckdriving['Description'].str.extractall(r'(\d+)')
        self.cust_nodes = self.cust_nodes.astype(int)
        self.cust_nodes = self.cust_nodes[0].unique().tolist()
        # cust_nodes contains last node #nodes+1. The last one should always be the depot
        self.cust_nodes[-1] = 0
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

    def get_sorties(self):
        # Get all drone activities
        droneactivities = self.data['solution'][self.data['solution']['vehicleType'] == 'UAV']
        # Filter the sorties
        sorties = droneactivities[droneactivities['Description'].str.contains('Fly to UAV customer')]
        rendezvouss = droneactivities[droneactivities['activityType'] == (' UAV travels empty')]
        rendezvouss.loc[:, 'endNode'] = rendezvouss['endNode'].astype(int)
        self.cust_lim = max(rendezvouss['endNode'])
        # Convert rendezvouss to dictionary for quicker lookup
        rendezvouss_dict = {int(row['startNode']): int(row['endNode']) for index, row in rendezvouss.iterrows()}
        self.trips = []

        # Process sorties
        for index, sortie in sorties.iterrows():
            i = int(sortie['startNode'])
            j = int(sortie['endNode'])
            
            # Check if j is in rendezvouss_dict
            if j in rendezvouss_dict:
                if rendezvouss_dict[j] >= self.cust_lim:
                    # Set higher nodes to 0, this is the depot again
                    k = 0
                else:
                    k = rendezvouss_dict[j]
                self.trips.append((i, j, k))
            else:
                raise ValueError(f"No corresponding rendezvous found for endNode {j} starting from node {i}")
    
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
            # In general, we noticed that we don't need to slow down if the turn is smaller than 25 degrees. However, this will depend
            # on the cruise speed of the drones.
            if angle > 25:
                turns.append(True)
            else:
                turns.append(False)
            i += 1

        # Let the vehicle slow down for the depot
        turns.append(True)

        # Add some commands to pan to the correct location and zoom in.
        scen_text = f'00:00:00>PAN {route_lats[0]} {route_lons[0]}\n' # Pan to the origin
        scen_text += '00:00:00>ZOOM 50\n\n' # Zoom in

        # Get angle of first direction to ensure correct orientation
        achdg, _ = kwikqdrdist(route_lats[0], route_lons[0], route_lats[1], route_lons[1])

        # CRE command takes Aircraft ID, Aircraft Type, Lat, Lon, hdg, alt[ft], spd[kts]
        acid = 'TRUCK'
        actype = 'Truck'
        acalt = 0 # ft, ground altitude
        acspd = 5 if turns[1] else 25 #kts, set it as 5 if the first waypoint is a turn waypoint.
        scen_text += f'00:00:00>CRE {acid} {actype} {route_lats[0]} {route_lons[0]} {achdg} {acalt} {acspd}\n'

        # After creating it, we want to add all the waypoints. We can do that using the ADDTDWAYPOINTS command.
        # ADDTDWAYPOINTS can chain waypoint data in the following way:
        # ADDTDWAYPOINTS ACID LAT LON ALT SPD Turn? TurnSpeed
        # SPD here can be set as the cruise speed so the vehicle knows how fast to go
        cruise_spd = 25 #kts
        cruise_alt = acalt # Keep it constant throughout the flight
        # Turn speed of 5 kts usually works well
        turn_spd = 5 #kts

        # Initiate adddtwaypoints command
        scen_text += f'00:00:00>ADDTDWAYPOINTS {acid}' # First add the root of the command
        # Loop through waypoints
        # Refresh command after 100 iterations, BlueSky cannot handle everything in one go
        i = 0
        j = 0
        cust_i = 0
        self.scn_lim = 100
        for wplat, wplon, turn in zip(route_lats, route_lons, turns):
            # Check if this waypoint is a turn
            if turn:
                wptype = 'TURNSPD'
            else:
                wptype = 'FLYBY'
            if j == 0 and i > self.scn_lim:
                scen_text += '\n'
                # Enable vertical and horizontal navigation after first set of wps
                scen_text += f'00:00:00>LNAV {acid} ON\n'
                scen_text += f'00:00:00>VNAV {acid} ON\n'
                # Turn trail on, tracing for testing
                scen_text += '00:00:00>TRAIL ON'
            if i > self.scn_lim:
                added = cust_i
                while str(self.customers['Route_lat'].loc[self.cust_nodes[cust_i]]) in scen_text and \
                    str(self.customers['Route_lon'].loc[self.cust_nodes[cust_i]]) in scen_text and \
                    cust_i + 1 < len(self.cust_nodes):
                    cust_i += 1
                
                scen_text = self.sortie_scen(scen_text, acid, j, added, cust_i)

                j += 1
                scen_text += f'\n00:{"{:02}".format(j * self.scn_lim // 20)}:00>ADDTDWAYPOINTS {acid}'
                i = 0

            i += 1
            # Add the text for this waypoint. It doesn't matter if we always add a turn speed, as BlueSky will
            # ignore it if the wptype is set as FLYBY
            scen_text += f',{wplat},{wplon},{cruise_alt},{cruise_spd},{wptype},{turn_spd}'

        if j == 0:
            scen_text += '\n'
            # Enable vertical and horizontal navigation after first set of wps
            scen_text += f'00:00:00>LNAV {acid} ON\n'
            scen_text += f'00:00:00>VNAV {acid} ON\n'
            # Turn trail on, tracing for testing
            scen_text += '00:00:00>TRAIL ON'

        # Add a newline at the end of the addtdwaypoints command
        scen_text += '\n'

        # Delete AC at route end
        destination_tolerance = 3/1852 # we consider it arrived if it is within 3 metres of the destination, converted to nautical miles.
        scen_text += f'00:{"{:02}".format(j * self.scn_lim // 20)}:00>{acid} ATDIST {route_lats[-1]} {route_lons[-1]} {destination_tolerance} DEL {acid}\n'

        # # Change directory to scenario folder
        # try:
        #     os.chdir(root_path + '/scenario')
        # except:
        #     raise Exception('Scenario folder not found')

        # Save the text in a scenario file
        with open(save_name, 'w') as f:
            f.write(scen_text)

    def sortie_scen(self, scen_text, acid, J, added=None, cust_i=None):
        if not added is None and not cust_i is None:
            passed_custs = self.cust_nodes[added : cust_i]
        else:
            passed_custs = self.cust_nodes
        for node in passed_custs:
            matching_trip = next((trip for trip in self.trips if trip[0] == node), None)
            if matching_trip:
                i = matching_trip[0]
                j = matching_trip[1]
                k = matching_trip[2]
                i_coords = f"{self.customers.loc[i]['Route_lat']}/{self.customers.loc[i]['Route_lon']}"
                j_lat = f"{self.customers.loc[j][' latDeg']}"
                j_lon = f"{self.customers.loc[j][' lonDeg']}"
                k_coords = f"{self.customers.loc[k]['Route_lat']}/{self.customers.loc[k]['Route_lon']}"
                scen_text += f"\n00:{'{:02}'.format(J * self.scn_lim // 20)}:00>ADDOPERATIONPOINTS {acid}, {i_coords}, SORTIE, 5, M600, "
                scen_text += f"{j_lat}, {j_lon}, {k_coords}, 100, 25"

        return scen_text

def kwikqdrdist(lata, lona, latb, lonb):
    """Gives quick and dirty qdr[deg] and dist [m]
       from lat/lon. (note: does not work well close to poles)"""
    re      = 6371000.  # radius earth [m]
    dlat    = np.radians(latb - lata)
    dlon    = np.radians(((lonb - lona)+180)%360-180)
    cavelat = np.cos(np.radians(lata + latb) * 0.5)

    dangle  = np.sqrt(dlat * dlat + dlon * dlon * cavelat * cavelat)
    dist    = re * dangle

    qdr     = np.degrees(np.arctan2(dlon * cavelat, dlat)) % 360
    return qdr, dist

def plot_linestring(line, point=None, overlap=True):
    """Helper function, plots a Shapely LineString

    args: type, description:
        line: Shapely Linestring, sequence of coordinates of a geometry
        point: tuple, coordinates of an additional point to plot
        overlap: bool, plot on a new figure yes/no"""
    if not overlap:
        plt.figure()
    x, y = line.xy
    plt.plot(x, y, marker='o')  # Plot the line with markers at vertices
    plt.plot(x[-1],y[-1],'rs') 
    if not point is None:
        plt.plot(point[0], point[1], 'gs')
    plt.title('LineString Plot')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)  # Optional: adds a grid
    plt.show()

def reverse_linestring(line):
    """Helper function, reverses a Shapely LineString

    arg: type, description
    line: Shapely Linestring, line to reverse"""
    reversed_coords = LineString(list(line.coords)[::-1])
    return reversed_coords

def shift_circ_ls(line, adjustment=(1e-9, 1e-9)):
    """Helper function, adjusts a Shapely LineString's last coordinates, in case of circular loops
    arg: type, description
    line: Shapely Linestring, line to change last coordinates from
    adjustment: tuple, delta of change"""

    if line is None or line.is_empty:
        return line

    # Extract coordinates to a list
    coords = list(line.coords)

    # Modify the last coordinate
    last_coord = coords[-1]
    new_last_coord = (last_coord[0] + adjustment[0], last_coord[1] + adjustment[1])
    coords[-1] = new_last_coord

    # Create a new LineString with modified coordinates
    return LineString(coords)

routes = mFSTSPRoute("/Users/raven/Documents/TU/MSc/Thesis/Code/mFSTSP/Problems/20170606T123331779163",
                     "tbl_solutions_103_1_Heuristic.csv")
routes.get_sorties()
routes.construct_truckroute()
routes.construct_scenario("Buffalo.scn")
# routes = mFSTSPRoute("/Users/raven/Documents/TU/MSc/Thesis/Code/mFSTSP/Problems/20170606T113038113409",
#                      "tbl_solutions_103_1_Heuristic.csv")