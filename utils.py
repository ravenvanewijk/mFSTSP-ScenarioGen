import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString
from shapely.ops import linemerge
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError


class CityNotFoundError(Exception):
    """Custom exception for cases where the city cannot be found."""
    pass

def get_city_from_bbox(north, south, east, west):
    """
    Get the city name based on the bounding box limits.

    Parameters:
    north (float): Northern latitude of the bounding box.
    south (float): Southern latitude of the bounding box.
    east (float): Eastern longitude of the bounding box.
    west (float): Western longitude of the bounding box.

    Returns:
    str: The name of the city.

    Raises:
    Exception: If the city cannot be determined.
    """
    geolocator = Nominatim(user_agent="my_geopy_application")
    
    # Calculate the center point of the bounding box
    center_lat = (north + south) / 2
    center_lon = (east + west) / 2

    try:
        # Reverse geocode using the center point of the bounding box
        location = geolocator.reverse((center_lat, center_lon), exactly_one=True)
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        raise Exception(f"Geocoding service error: {e}")

    if location and location.raw:
        address = location.raw.get('address', {})
        city = address.get('city') or address.get('town') or address.get('village') or address.get('county')
        if city:
            return city
    
    raise CityNotFoundError("City could not be determined from the bounding box limits.")

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