from queue import PriorityQueue
import numpy as np
from scipy.spatial import Voronoi
from bresenham import bresenham
import networkx as nx
from planning_utils import heuristic, create_grid, collinearity_prune
from udacidrone.frame_utils import global_to_local
import numpy.linalg as LA

'''
In path planning, Voronoi polygons help in determining the safest routes by maximizing 
the distance from obstacles. By placing points on obstacles and computing the Voronoi diagram, 
you create regions where paths along the edges are relatively safe from the obstacles.
'''
# To be able to make graph represent the obtimal path several steps are required:
# 1- We first should define  all free paths that the drone can flay through it this is done using:
# ---> Voronoi Diagram
# 2- Voronoi Diagram requires obestacles centers ----> obj_center() 
# 3- The actual free paths are the voronoi edges ----> find_open_edges_voronoi()  
# 4- Craeting graph from these edges ----> create_graph_from_edges()


def obj_center(data,north_offset,east_offset,saftey_distance,drone_alt):
    points = []
    
    for i in range(data.shape[0]):
        north,east,alt,d_north,d_east,d_alt = data[i,:]
        if alt + d_alt + saftey_distance > drone_alt:
            points.append([north-north_offset,east-east_offset])
    return points

def find_open_edges_voronoi(graph, grid):
    edges = []
    
    for v in graph.ridge_vertices:
        p1 = graph.vertices[v[0]]
        p2 = graph.vertices[v[1]]
        cells = list(bresenham(int(p1[0]),int(p1[1]),int(p2[0]),int(p2[1])))
        hit = False
        for c in cells:
            if np.amin(c) < 0 or c[0] >= grid.shape[0] or c[1] >= grid.shape[1]:
                hit = True
                break
            if grid[c[0],c[1]] == 1:
                hit = True
                break
        if not hit:
            p1 = (p1[0],p1[1])
            p2 = (p2[0],p2[1])
            edges.append((p1,p2))
    return edges     


def create_graph_from_edges(edges):
    G = nx.Graph()
    
    for e in edges:
        p1 = e[0]
        p2 = e[1]
        
        dist = np.linalg.norm(np.array(p1) - np.array(p2))
        G.add_edge( p1, p2, weight=dist)
    return G

def create_graph(data, drone_altitude, safety_distance):
    
    grid, north_offset, east_offset = create_grid(data,drone_altitude,safety_distance)
    
    centers = obj_center(data,north_offset,east_offset,drone_altitude, safety_distance)
    
    voronoi = Voronoi(centers)
    
    edges = find_open_edges_voronoi(voronoi,grid)
    
    graph = create_graph_from_edges(edges)
    
    return (graph,north_offset,east_offset)


def a_star(graph, start, goal):
    """Modified A* to work with NetworkX graphs."""

    path = []
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_cost = item[0]
        current_node = item[1]

        if current_node == goal:
            print('Found a path.')
            found = True
            break
        else:
            for next_node in graph[current_node]:
                cost = graph.edges[current_node, next_node]['weight']
                new_cost = current_cost + cost + heuristic(next_node, goal)

                if next_node not in visited:
                    visited.add(next_node)
                    queue.put((new_cost, next_node))

                    branch[next_node] = (new_cost, current_node)

    path = []
    path_cost = 0
    if found:

        # retrace steps
        path = []
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])

    return path[::-1], path_cost


def Closest_point(graph,point):
    
    current_point  = (point[0],point[1])
    closest_point  = None
    dis  = 100000
    
    for p in graph.nodes:
         d = np.linalg.norm(np.array(p) - np.array(current_point))
         if d < dis:
             closest_point = p
             dis = d
    return closest_point


def calculate_waypoints(global_start, global_goal, global_home, data, drone_altitude, safety_distance):
    
    """
    Calculates the waypoints for the trajectory from `global_start` to `global_goal`.
    Using `global_home` as home and colliders `data`.
    """
    # Calculate graph and offsets
    graph, north_offset, east_offset = create_graph(data, drone_altitude, safety_distance)

    map_offset = np.array([north_offset, east_offset, .0])

    # Convert start position from global to local.
    local_position = global_to_local(global_start, global_home) - map_offset

    # Find closest point to the graph for start
    graph_start = Closest_point(graph, local_position)

    # Convert goal postion from global to local
    local_goal = global_to_local(global_goal, global_home) - map_offset

    # Find closest point to the graph for goal
    graph_goal = Closest_point(graph, local_goal)

    # Find path
    path, _ = a_star(graph, graph_start, graph_goal)
    path.append(local_goal)

    # Prune path
    path = collinearity_prune(path, epsilon=3)

    # Calculate waypoints
    return [[int(p[0] + north_offset), int(p[1] + east_offset), drone_altitude, 0] for p in path]