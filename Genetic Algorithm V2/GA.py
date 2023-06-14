# MODIFIED CODE FROM https://medium.com/@pasdan/genetic-algorithm-the-travelling-salesman-problem-via-python-deap-f238e0dd1a73
# understood through Udemy course: https://www.udemy.com/course/search-algorithms-in-artificial-intelligence-with-java/learn/lecture/12530754#overview

import numpy as np
import random
import csv
from deap import algorithms, base, creator, tools
import matplotlib.pyplot as plt
import time
import multiprocessing



# Load distance matrix
def load_distance_matrix():
    csv_file = 'distance_matrix_TSP50.csv'
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        data = list(csv_reader)
        dist_mat = np.array(data, dtype=int)
    return dist_mat

distances = load_distance_matrix()
NUMBER_OF_CITIES = distances.shape[0]

# Setup the Genetic Algorithm
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("indices", np.random.permutation, NUMBER_OF_CITIES)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Individual initialization method
def init_individual():
    indices = np.random.permutation(NUMBER_OF_CITIES-1) + 1  # This creates a permutation of the cities, excluding the first one
    return [0] + list(indices) 

toolbox.register("individual", tools.initIterate, creator.Individual, init_individual)  # Adjust this line

# When using multiprocessing with DEAP, the evaluation function needs to take a single argument, which is typically a tuple of arguments, rather than multiple arguments.

def EVALUATE(individual):
    summation = 0
    start = individual[0]
    for i in range(1, len(individual)):
        end = individual[i]
        summation += distances[start][end]
        start = end
    return summation,


pool = multiprocessing.Pool() # add multiprocessing


toolbox.register("map", pool.map)
toolbox.register("evaluate", EVALUATE)

toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1) # indpb is the probability of mutating each individual
toolbox.register("select", tools.selTournament, tournsize=10) # tournsize is the number of individuals participating in each tournament
# toolbox.register("select", tools.selRoulette) # tournsize is the number of individuals participating in each tournament

# Run and Examine the results
def main():
    try:
        pop = toolbox.population(n=5000) # change the population number when increasing the distance matrix
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("Avg", np.mean)
        stats.register("Std", np.std)
        stats.register("Min", np.min)
        stats.register("Max", np.max)

        # typical value for cxpb parameter 0.5-0.7
        # cxpb is the probability with which two individuals are crossed
        # mutpb is the probability for mutating an individual


        # pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.01, ngen=100, 
        #                                 stats=stats, halloffame=hof, verbose=True)
        # calculate the fitness for the initial population
        fitnesses = list(pool.map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for g in range(100):  # number of generations
            offspring = toolbox.select(pop, len(pop))
            offspring = algorithms.varAnd(offspring, toolbox, cxpb=0.7, mutpb=0.01)

            # calculate fitness for the offspring
            fitnesses = list(pool.map(toolbox.evaluate, offspring))
            for ind, fit in zip(offspring, fitnesses):
                ind.fitness.values = fit

            pop = offspring
            hof.update(pop)
            record = stats.compile(pop)
            print(record)

        gen = log.select("gen")
        min_fitness_values = log.select("Min")
    
    finally:
        pool.close()

    return pop, log, hof, gen, min_fitness_values

if __name__ == "__main__":
    start_time = time.time()
    pop, log, hof, gen, min_fitness_values = main()
    end_time = time.time()
    total_time = end_time - start_time

    print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))
    print("Total time taken to find the minimum distance: %s seconds" % total_time)

    # Plot
    plt.figure()
    plt.plot(gen, min_fitness_values)
    plt.scatter(gen[min_fitness_values.index(min(min_fitness_values))], min(min_fitness_values), color='red')  # highlight the minimum point
    plt.xlabel("Generation")
    plt.ylabel("Minimum Distance")
    plt.show()
