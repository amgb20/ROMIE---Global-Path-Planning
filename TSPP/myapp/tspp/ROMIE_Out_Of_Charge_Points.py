from geopy.distance import geodesic
import numpy as np

def compute_out_of_charge_points(route, distances, capacity):
    out_of_charge_points = []
    remaining_capacity = capacity
    last_station_location = route[0]  # fully charged at the beginning
    current_location = last_station_location

    for i in range(len(distances)):
        distance = distances[i]
        while remaining_capacity < distance:
            # a ratio is computed to determine the out-of-charge point and ratio is always < 1 as it enters the while loop
            ratio = remaining_capacity / distance
            # Compute the out_of_charge_point
            out_of_charge_point = {
                'lat': current_location['lat'] + ratio * (route[i+1]['lat'] - current_location['lat']),
                'lng': current_location['lng'] + ratio * (route[i+1]['lng'] - current_location['lng']),
            }
            out_of_charge_points.append(out_of_charge_point) # we need to append the out_of_charge_point to the list of out_of_charge_points

            remaining_capacity = capacity  # we assume the robot recharge here but will be handled differently in the code later

            # The robot will continue its path from the out_of_charge_point as we have not figured out of charge point midaway to charging station for charging staiton optmization reasons
            
            current_location = out_of_charge_point

            # Subtract the already covered distance from the total distance
            distance = distance - (ratio * distance)

        # Subtract the distance to the next waypoint from the remaining capacity
        remaining_capacity -= distance
        
        # Move to the next waypoint
        current_location = route[i+1]

    return out_of_charge_points



