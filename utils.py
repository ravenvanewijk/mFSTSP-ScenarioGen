import osmnx as ox
import networkx as nx
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString
from shapely.ops import linemerge


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

def simplify_graph(G, tol=0.0001, gpkg_file=False):
    """
    Simplify the geometries of the edges in the GeoDataFrame.
    
    Args:
    - G: GeoDataFrame containing the edges to be simplified
    - tol: float, the simplification tolerance
    
    Returns:
    - GeoDataFrame with simplified geometries (edges)
    """
    # Convert the graph to GeoDataFrames
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
    

    # Identify and handle list-type fields
    list_type_columns = [col for col in gdf_edges.columns if gdf_edges[col].apply(lambda x: isinstance(x, list)).any()]
    # Convert list-type columns to strings
    for col in list_type_columns:
        gdf_edges[col] = gdf_edges[col].apply(lambda x: ','.join(map(str, x)) if isinstance(x, list) else x)

    gdf_edges_simplified = gdf_edges.copy()
    simplified_geometries = gdf_edges_simplified['geometry'].apply(lambda geom: geom.simplify(tol, \
                                                                                                preserve_topology=True))
    gdf_edges_simplified['geometry'] = simplified_geometries
    
    if gpkg_file:
        # Save the edges and nodes to a gpkg file for closer inspection on the simplifcation.
        # This has already been performed for tol=0.0001 (default), which gives accurate results but also faster 
        # processing of scenarios and simulation
        gdf_edges.to_file("graph_comparison.gpkg", layer='original_edges', driver="GPKG")
        gdf_edges_simplified.to_file("graph_comparison.gpkg", layer=f'edges {tol}', driver="GPKG")
        gdf_nodes.to_file("graph_comparison.gpkg", layer='nodes', driver="GPKG")

    # Convert back to an OSMNX graph
    G_mod = ox.graph_from_gdfs(gdf_nodes, gdf_edges_simplified)

    return G_mod
