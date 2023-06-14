"""
===============================================================================
                           SimpleGrid.py

Description:
    This script generates the different location of the charging stations base 
    on the path taken by the robot.
    A max_distance is represented by the maximum distance the robot can travel
    according to its life time before needing to recharge.

Usage:
    
- charging_stations is an empty list storing indices of the waypoints where the robot will recharge.

- distance_travelled and set it to 0. Keeping track of the total distance the robot has traveled since its last recharge.

- For loop created to iterate through the indices in path (except the last one).

- out_of_charge_points calculates the total distance traveled by the robot and the indices of the waypoints where the robot will recharge.

a. Calculate the Euclidean distance (segment_distance) between the current waypoint and the next waypoint using numpy's linalg.norm function.

b. segment_distance is added to distance_travelled.

c. Check if distance_travelled is greater than max_distance. If it is, the robot needs to recharge. Append the index of the current waypoint to charging_stations, and reset distance_travelled to 0.

return the charging_stations list containing the indices of the charging stations.

- find_optimal_charging_station(path, coordinates, max_distance): This is the main function that finds the optimal charging station location. 
It takes a path, the coordinates of the points in the path, and the max_distance the vehicle can travel without needing a recharge. The function first identifies 
the out-of-charge points in the path by checking if the distance traveled exceeds the max_distance. Then, it checks all possible charging station locations by 
iterating through the coordinates. For each charging station location, it computes the total distance traveled using the total_distance() function. 
If the total distance is smaller than the current minimum distance and the charging station is not in the out_of_charge_points list, the function updates the 
minimum distance, best charging station, and best charging station routes. The function finally returns a dictionary of charging stations for each out-of-charge 
point and a list of best charging station routes.

Author:
    Alexandre Benoit (amgb20@bath.ac.uk)
===============================================================================
"""

import numpy as np
import itertools


def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2)) # np.linalg.norm() to calculate the Euclidean magnitude of the vector.


def total_distance(path, coordinates, charging_station, out_of_charge_points):
    distance_travelled = 0
    charging_station_routes = []
    for i in range(len(path) - 1):
        segment_distance = distance(
            coordinates[path[i]], coordinates[path[i + 1]])
        distance_travelled += segment_distance

        if path[i] in out_of_charge_points:
            distance_travelled += distance(
                coordinates[path[i]], coordinates[charging_station])
            charging_station_routes.append([path[i], charging_station])

    return distance_travelled, charging_station_routes


def find_optimal_charging_station(path, coordinates, max_distance):
    min_distance = float('inf')
    best_charging_station = None
    best_charging_station_routes = []

    # Find out-of-charge points
    out_of_charge_points = []
    distance_travelled = 0
    for i in range(len(path) - 1):
        segment_distance = distance(
            coordinates[path[i]], coordinates[path[i + 1]])
        distance_travelled += segment_distance
        if distance_travelled > max_distance:
            out_of_charge_points.append(path[i])
            distance_travelled = 0

    # Check all possible charging station locations
    charging_stations = {}
    for charging_station in range(len(coordinates)):
        total_dist, charging_station_routes = total_distance(
            path, coordinates, charging_station, out_of_charge_points)
        if total_dist < min_distance and charging_station not in out_of_charge_points:
            min_distance = total_dist
            best_charging_station = charging_station
            best_charging_station_routes = charging_station_routes
            for out_of_charge_point in out_of_charge_points:
                charging_stations[out_of_charge_point] = charging_station

    return charging_stations, best_charging_station_routes
