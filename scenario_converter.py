import os
import geopy.distance
import pandas as pd
import geopandas as gpd
import osmnx as ox
import networkx as nx
import numpy as np
import taxicab as tc
from shapely.geometry import Point, LineString
from shapely.ops import linemerge

class mFSTSPRoute:
    def __init__(self, instance_path, sol_type = 'ALL', solutions_name = 'tbl_solutions'):
        files = os.listdir(instance_path)
        solutions_files = [filename for filename in files if solutions_name in filename]
        self.solutions = {}
        for solution_file in solutions_files:
            if (solution_file.split('_')[-1] != 'Heuristic.csv' and sol_type.upper() != 'HEURISTIC') \
                and sol_type.upper() != 'ALL':
                continue
            if not solution_file == 'tbl_solutions_103_1_Heuristic.csv':
                continue
            
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
            self.solutions[solution_file] = {
                'problemName': metadata['problemName'].iloc[0],
                'numUAVs': int(metadata['numUAVs'].iloc[0]),
                'requireTruckAtDepot': bool(metadata['requireTruckAtDepot'].iloc[0]),
                'requireDriver': bool(metadata['requireDriver'].iloc[0]),
                'solution': solution
                                            }
        
        # Load customer locations
        self.custnodes = pd.read_csv(instance_path + '/tbl_locations.csv')

        # Find nearest city to the customers, used to identify the city where the scenario is going to be
        nearest_city = self.get_nearest_city((self.custnodes.iloc[0][' latDeg'], self.custnodes.iloc[0][' lonDeg']))

        if nearest_city == 'Buffalo, NY, USA':
            lims = (43.027642, 42.831743, -78.698845, -78.949956)
        else:
            raise Exception("This city is not supported yet, choose another type")
        
        # This is too small:
        # self.G = ox.graph_from_place(nearest_city, network_type='drive')
        # Adjusted box sizes to include the entire map
        self.G = ox.graph_from_bbox(bbox=lims, network_type='drive')
        G_projected = ox.project_graph(self.G)

        # Extract projected coordinates and add them to the original DataFrame
        self.custnodes['proj_x'], self.custnodes['proj_y'] = self.project_customers(G_projected)

        # to_inspect = 19
        # nearest_node = ox.nearest_nodes(G_projected, 
        #                                   self.custnodes.loc[to_inspect]['proj_x'], 
        #                                   self.custnodes.loc[to_inspect]['proj_y'])


        # fig, ax = ox.plot_graph(self.G, node_size=30, node_color='#66ccff', node_zorder=3,
        #                     bgcolor='k', edge_linewidth=1.5, edge_color='#e2e2e2')
        # ax.scatter(self.custnodes.loc[to_inspect][' lonDeg'], self.custnodes.loc[to_inspect][' latDeg'], color='red', s=100, label='Additional Point', zorder=5)
        # ax.scatter(G_projected.nodes[nearest_node]['lon'], G_projected.nodes[nearest_node]['lat'], color='red', s=100, label='Additional Point', zorder=5)
        
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
    
    def project_customers(self, G_projected):
        """Projects the customer nodes onto a projected graph.
        arg: type, description
        G_projected: networkx multidirected projected graph"""
        # Create a GeoDataFrame from custnodes, so that these can be projected to projected graph
        cust_gdf = gpd.GeoDataFrame(
                        self.custnodes,
                        geometry=[Point(xy) for xy in zip(self.custnodes[' lonDeg'], self.custnodes[' latDeg'])],
                        crs='EPSG:4326'
                            )

        # Project the GeoDataFrame to match the graph's CRS
        cust_gdf_projected = cust_gdf.to_crs(G_projected.graph['crs'])

        return cust_gdf_projected.geometry.x, cust_gdf_projected.geometry.y

    def construct_route(self, file):
        # Extract all truck route nodes
        truckactivities = self.solutions[file]['solution'][self.solutions[file]['solution']['vehicleType'] == 'Truck']
        # Should already be sorted, redundancy
        truckactivities = truckactivities.sort_values('startTime')
        # Get all traveling activities
        truckdriving = truckactivities[truckactivities['Description'].str.startswith(' Travel from node')]
        # Extract node sequence from traveling activities
        cust_nodes = truckdriving['Description'].str.extractall(r'(\d+)')
        cust_nodes = cust_nodes.astype(int)
        cust_nodes = cust_nodes[0].unique().tolist()
        # cust_nodes contains last node #nodes+1. The last one should always be the depot
        cust_nodes[-1] = 0
        route = []

        # zip the customers to create pairs
        cust_pairs = zip(cust_nodes[:-1], cust_nodes[1:])

        for U, V in cust_pairs:
            # Now use the taxicab package to find the shortest path between 2 customer nodes.
            # Args come in this order: 0: distance, 1: nodes 2: unfinished linepart begin 3: unfinished linepart end
            # Look up location of customer nodes U and V in the custnode df
            routepart = tc.distance.shortest_path(self.G,   (self.custnodes.iloc[U][' latDeg'], 
                                                        self.custnodes.iloc[U][' lonDeg']), 
                                                        (self.custnodes.iloc[V][' latDeg'], 
                                                        self.custnodes.iloc[V][' lonDeg']))
            
            # Use the nodes to extract all edges u, v of graph G that the vehicle completely traverses
            routepart_edges = zip(routepart[1][:-1], routepart[1][1:])

            # routepart
            route.append(routepart[2])
            # For every pair of edges, append the route with the Shapely LineStrings
            for u, v in routepart_edges:
                # Some edges have this attribute embedded, when geometry is curved
                if 'geometry' in self.G.edges[(u, v, 0)]:
                    route.append(self.G.edges[(u, v, 0)]['geometry'])
                # Other edges don't have this attribute. These are straight lines between their two nodes.
                else:
                    # So, get a straight line between the nodes and append that line piece
                    route.append(LineString([(self.G.nodes[u]['x'], self.G.nodes[u]['y']), 
                                            (self.G.nodes[v]['x'], self.G.nodes[v]['y'])]))
            
            route.append(routepart[3])
        
        # These are a bunch of lines, these need to combine them into one multilinestring
        route_line = linemerge(route)

        # Now that we have the line, we can get the waypoints in LAT LON. 
        # The graph is in LON LAT, but BlueSky works with LAT LON.
        self.route_waypoints = list(zip(route_line.xy[1], route_line.xy[0]))
        self.route_lats = route_line.xy[1]
        self.route_lons = route_line.xy[0]

routes = mFSTSPRoute("/Users/raven/Documents/TU/MSc/Thesis/Code/mFSTSP/Problems/20170606T123331779163",
                     "tbl_solutions_103_1_Heuristic.csv")
routes.construct_route("tbl_solutions_103_1_Heuristic.csv")

# routes = mFSTSPRoute("/Users/raven/Documents/TU/MSc/Thesis/Code/mFSTSP/Problems/20170606T113038113409",
#                      "tbl_solutions_103_1_Heuristic.csv")