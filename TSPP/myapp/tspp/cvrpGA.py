import random
import numpy
from geopy.distance import geodesic

# added lines
# We need to install deap 'pip install deap' DEAP (Distributed Evolutionary Algorithms in Python) which is an easy-to-use yet powerful library for creating genetic algorithms
# https://deap.readthedocs.io/en/master/
from deap import base, creator, tools, algorithms

# Create the data
N_NODES = 20  # Number of nodes
CAPACITY = 10  # Capacity of the vehicle
LOCATIONS = [random.sample(range(100), 2) for _ in range(N_NODES)]  # Randomly generate (x, y) coordinates for each node

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("indices", random.sample, range(N_NODES), N_NODES)

# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def calculate_distance_matrix(locations):
    # Create distance matrix
        distance_matrix = []
        for location_1 in locations:  # for each location in the list of locations
            row = []
            for location_2 in locations:  # for each location in the list of locations
                row.append(int(geodesic(
                    (location_1['lat'], location_1['lng']), (location_2['lat'], location_2['lng'])).meters))  # append the distance between the two locations
            distance_matrix.append(row)

def evalCVRP(individual, distance_matrix, demand_vector, capacity):
    total_distance = 0
    total_demand = 0
    vehicle_load = 0
    penalty = 10000  # The penalty for exceeding the capacity

    # Add the depot to the start and end of the route
    route = [0] + individual + [0]

    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i]][route[i+1]]
        vehicle_load += demand_vector[route[i]]

        if vehicle_load > capacity:
            total_distance += penalty
            vehicle_load = demand_vector[route[i]]

    return total_distance,

def cvrp(locations, demand_vector, capacity, n_individuals=100, n_generations=40):
    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register("indices", random.sample, range(1, len(locations)), len(locations) - 1)

    # Structure initializers
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Operator registering
    toolbox.register("evaluate", evalCVRP, distance_matrix=calculate_distance_matrix(locations), demand_vector=demand_vector, capacity=capacity)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=n_individuals)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=n_generations, stats=stats, halloffame=hof, verbose=True)

    best_route = hof[0]
    
    # Identify the charging stations and points where the drone runs out of charge
    charging_stations = []
    out_of_charge_points = []
    vehicle_load = 0

    for i in range(len(best_route) - 1):
        vehicle_load += demand_vector[best_route[i]]
        if vehicle_load > capacity:
            charging_stations.append(best_route[i])
            out_of_charge_points.append(best_route[i-1])
            vehicle_load = demand_vector[best_route[i]]

    return best_route, charging_stations, out_of_charge_points



