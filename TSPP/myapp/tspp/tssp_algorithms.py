"""
===============================================================================
                           tssp_algorithm.py

Description:
    This script contains the implementation of the tssp_algorithm algorithm for
    solving the Travelling Salesman Path Problem. It works by generating a grid
    of points and finding the shortest path between them.

Usage:
    
TO BE COMPLETED

Author:
    Alexandre Benoit (amgb20@bath.ac.uk)
===============================================================================
"""

from tsp_solver.greedy import solve_tsp

# Nearest Neighbor
def nearest_neighbour(dist_matrix, start=0):
    n = len(dist_matrix)
    path = [start]
    unvisited = set(range(n))
    unvisited.remove(start)

    while unvisited:
        current = path[-1]
        nearest_point = min(
            unvisited, key=lambda point: dist_matrix[current][point])
        path.append(nearest_point)
        unvisited.remove(nearest_point)

    return path

# 2-opt


def two_opt(dist_matrix, start=0):
    n = len(dist_matrix)
    path = nearest_neighbour(dist_matrix, start=start)
    improvement = True

    while improvement:
        improvement = False
        for i in range(n - 2):
            for j in range(i + 2, n - 1):
                if dist_matrix[path[i]][path[i + 1]] + dist_matrix[path[j]][path[j + 1]] > dist_matrix[path[i]][path[j]] + dist_matrix[path[i + 1]][path[j + 1]]:
                    path[i + 1:j + 1] = reversed(path[i + 1:j + 1])
                    improvement = True

    return path

# Christofides


def christofides(dist_matrix, start=0):
    return solve_tsp(dist_matrix, endpoints=(start, None))