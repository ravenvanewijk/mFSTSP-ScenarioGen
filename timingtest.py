import osmnx as ox
import os
import roadroute_lib as rr
from shapely.ops import linemerge
from utils import simplify_graph, kwikqdrdist
from graph_ops import add_missing_spd
from utils import spdlim_ox2bs

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


lims = (43.03392544699964, 42.81679855300036, -78.67067714771929, -78.9389038522807)
G = ox.graph_from_bbox(bbox=lims, network_type='drive')
# Simplify the graph using osmnx
G = simplify_graph(G)
G = add_missing_spd(G)

road_route, spdlims = rr.roadroute(G, (42.959024,	-78.719749), (42.958582	,-78.887663))
road_route_merged = linemerge(road_route)
scenario = construct_scenario(road_route, spdlims)



# Change directory to scenario folder
try:
    os.chdir(os.getcwd() + '/scenario')
except:
    raise Exception('Scenario folder not found')

# Save the text in a scenario file
with open('timetest.scn', 'w') as f:
    f.write(scenario)
