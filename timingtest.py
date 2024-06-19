import taxicab as tc 
import osmnx as ox
import os
from shapely import LineString
from shapely.ops import linemerge
from utils import reverse_linestring, simplify_graph, kwikqdrdist, mph2kts
from collections import Counter


def roadroute(G, A, B):
    """Compute the road route from point A to point B using the taxicab distance.
    
    Args:
        - A: tuple/ list of floats, the starting point (lat, lon)
        - B: tuple/ list of floats, the destination point (lat, lon)
    """
    route = []
    spdlims = []
    routepart = tc.distance.shortest_path(G, [A[0], A[1]], 
                                                [B[0], B[1]])

    # Use the nodes to extract all edges u, v of graph G that the vehicle completely traverses
    routepart_edges = zip(routepart[1][:-1], routepart[1][1:])

    # routepart at beginning
    route.append(routepart[2])

    # First time you have to get 2 speed limits, first wp spdlim does not matter, will be reached instantly
    spdlims.extend([30] * (len(routepart[2].coords)))

    try:
        # For every pair of edges, append the route with the Shapely LineStrings
        for u, v in routepart_edges:
            # Some edges have this attribute embedded, when geometry is curved
            if 'geometry' in G.edges[(u, v, 0)]:
                route.append(G.edges[(u, v, 0)]['geometry'])
                spdlims.extend([G.edges[(u, v, 0)]['maxspeed']] * (len(G.edges[(u, v, 0)]['geometry'].coords) - 1))
            # Other edges don't have this attribute. These are straight lines between their two nodes.
            else:
                # So, get a straight line between the nodes and append that line piece
                route.append(LineString([(G.nodes[u]['x'], G.nodes[u]['y']), 
                                        (G.nodes[v]['x'], G.nodes[v]['y'])]))
                spdlims.extend([G.edges[(u, v, 0)]['maxspeed']])
    except IndexError:
        pass
    
    try:
        # Additional check for first linepart directionality. Sometimes it might be facing the wrong way.
        # The end of the beginning (incomplete) linestring should match
        try:
            if not route[1].coords[0] == routepart[2].coords[-1]:
                # Check if flipped version does align
                if route[1].coords[0] == routepart[2].coords[0]:
                    route[0] = reverse_linestring(route[0])
                else:
                    raise Exception('Taxicab alignment Error: Coordinates of beginning LineString does not align')
        except IndexError:
            pass
    except AttributeError:
        pass

    try:
        # Check whether final incomplete linestring is in proper direction, similar check
        try:
            if not route[-1].coords[-1] == routepart[3].coords[0]:
                # Check if flipped version does align
                if route[-1].coords[-1] == routepart[3].coords[-1]:
                    route.append(reverse_linestring(routepart[3]))

                else:
                    raise Exception('Taxicab alignment Error: Coordinates of final LineString does not align')
            else:
                route.append(routepart[3])
            spdlims.extend([30] * (len(routepart[3].coords) - 1))
        except IndexError:
            pass
    except AttributeError or IndexError:
        pass

    return linemerge(route), spdlims


def construct_scenario(road_route, spdlims):
    """Construct the scenario text for the waypoints of the road route.
    
    Args:
        - road_route: LineString, the road route as a LineString
    """
    route_waypoints = list(zip(road_route.xy[1], road_route.xy[0]))
    route_lats = road_route.xy[1]
    route_lons = road_route.xy[0]

    scenario = "00:00:00>IMPL ACTIVEWAYPOINT TDActWp\n"
    scenario += "00:00:00>IMPL AUTOPILOT TDAutoPilot\n"
    scenario += "00:00:00>IMPL ROUTE TDRoute\n"

    trkid = 'TRUCK'
    actype = 'Truck'
    acalt = 0 # ft, ground altitude
    acspd = 0 # start with 0 speed from depot
    achdg, _ = kwikqdrdist(route_lats[0], route_lons[0], route_lats[1], route_lons[1])
    scenario += f'00:00:00>CRE {trkid} {actype} {route_lats[0]} {route_lons[0]} {achdg} {acalt} {acspd}\n'

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

    # ADDTDWAYPOINTS can chain waypoint data in the following way:
    # ADDTDWAYPOINTS ACID LAT LON ALT SPD Turn? TurnSpeed
    # SPD here can be set as the cruise speed so the vehicle knows how fast to go
    # cruise_spd = 25 #kts
    cruise_alt = 0 # Keep it constant throughout the flight
    # Turn speed of 5 kts usually works well
    turn_spd = 10 #kts
    sharpturn_spd = 5 #kts
    # Initiate adddtwaypoints command
    scenario += f'00:00:00>ADDTDWAYPOINTS TRUCK' # First add the root of the command
    # Loop through waypoints
    for wplat, wplon, turn, spdlim in zip(route_lats, route_lons, turns, spdlims):
        # Check if this waypoint is a turn
        if turn == 'turn' or turn == 'sharpturn':
            wptype = 'TURNSPD'
            wp_turnspd = turn_spd if turn == 'turn' else sharpturn_spd
        else:
            wptype = 'FLYBY'
            # Doesn't matter what we pick here, as long as it is assigned. 
            # Will be ignored
            wp_turnspd = turn_spd
        # Add the text for this waypoint. It doesn't matter if we always add a turn speed, as BlueSky will
        # ignore it if the wptype is set as FLYBY
        # we have to give a speed if we dont specify RTAs, so set the default to 25
        cruisespd = spdlim_ox2bs(spdlim)
        scenario += f',{wplat},{wplon},{cruise_alt},{cruisespd},{wptype},{wp_turnspd}'

    scenario += f'\n00:00:00>LNAV {trkid} ON\n'
    scenario += f'00:00:00>VNAV {trkid} ON\n'
    scenario += f'00:00:00>SPDAP {trkid} 5\n'

    return scenario

#(-78.73246705772989, 42.870505, -78.732116, 42.87057069148937)

def spdlim_ox2bs(spdlim):
    if type(spdlim) == str:
        try:
            spdlim = int(spdlim.strip('mph'))
        except ValueError:
            # ValueError occurs when there is a double entry for spd limit
            # Take the highest one
            spdlim = max([int(s.strip().replace(' mph', '')) for s in spdlim.split(',')])
    elif type(spdlim) == int or type(spdlim) == float:
        pass
    else:
        raise TypeError("Undefined type for speedlimit")

    return mph2kts(spdlim)

# Function to get the most common speed for each road type
def most_common_speed(speed_list):
    return Counter(speed_list).most_common(1)[0][0]


def add_missing_spd(G):
    highwayspds = {}
    all_speeds = []
    
    for edge in G.edges:
        highway_type = G.edges[edge].get('highway')
        if 'maxspeed' in G.edges[edge]:
            if highway_type not in highwayspds:
                highwayspds[highway_type] = []
            speed = G.edges[edge]['maxspeed']
            highwayspds[highway_type].append(speed)
            all_speeds.append(speed)
    
    most_common_speed_dict = {road_type: most_common_speed(speeds) for road_type, speeds in highwayspds.items()}
    general_most_common_speed = most_common_speed(all_speeds)
    
    for edge in G.edges:
        if 'maxspeed' not in G.edges[edge]:
            highway_type = G.edges[edge].get('highway')
            selected_spd = most_common_speed_dict.get(highway_type, general_most_common_speed)
            G.edges[edge]['maxspeed'] = selected_spd

    return G




lims = (43.03392544699964, 42.81679855300036, -78.67067714771929, -78.9389038522807)
G = ox.graph_from_bbox(bbox=lims, network_type='drive')
# Simplify the graph using osmnx
G = simplify_graph(G)
G = add_missing_spd(G)

road_route, spdlims = roadroute(G, (42.959024,	-78.719749), (42.958582	,-78.887663))
scenario = construct_scenario(road_route, spdlims)



# Change directory to scenario folder
try:
    os.chdir(os.getcwd() + '/scenario')
except:
    raise Exception('Scenario folder not found')

# Save the text in a scenario file
with open('timetest.scn', 'w') as f:
    f.write(scenario)
