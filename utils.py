import matplotlib.pyplot as plt
import numpy as np
import math
from shapely.geometry import LineString
from shapely.ops import linemerge

city_coords = {
            "Seattle": (47.6062, -122.3321),
            "Buffalo": (42.8864, -78.8784)
            }

def get_map_lims(customer_locs, margin, unit='km'):
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

    # Extract latitudes and longitudes into separate lists
    latitudes = [loc[0] for loc in customer_locs]
    longitudes = [loc[1] for loc in customer_locs]

    # Find the maximum and minimum values
    latmax = max(latitudes)
    latmin = min(latitudes)
    lonmax = max(longitudes)
    lonmin = min(longitudes)

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

def find_nearest_city(location, city_coords):
    """Find the nearest city to a given location."""
    nearest_city = None
    min_distance = float('inf')

    for city, (city_lat, city_lon) in city_coords.items():
        _, distance = kwikqdrdist(location[0], location[1], city_lat, city_lon)
        if distance < min_distance:
            min_distance = distance
            nearest_city = city

    return nearest_city

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
    plt.grid(True)
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

def str_interpret(value):
    return value  # Ensure the value remains a string

def m2ft(m):
    """Converts distance in meters to feet"""
    return float(m) / 0.3048

def ms2kts(ms):
    """Converts speed in m/s to knots"""
    return float(ms) * 1.94384449 

def mph2kts(mph):
    """Converts speed in mph to knots"""
    return float(mph) * 0.868976242