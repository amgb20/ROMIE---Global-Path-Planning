"""
===============================================================================
                           SimpleGrid.py

Description:
    This script generates the different TSPP algorithms plot by calling run and
    measures the time complexity of each method calling run_experiments_and_save
    _plot and display grid points vs elapsed time.

Usage:
    
TO BE COMPLETED

Author:
    Alexandre Benoit (amgb20@bath.ac.uk)
===============================================================================
"""

import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt_instance
import matplotlib.pyplot as plt_cpu
import time
import os
import csv
import base64
import psutil  # To get the number of cores

from io import BytesIO
from scipy.optimize import curve_fit
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform
from tsp_solver.greedy import solve_tsp
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from . import tssp_algorithms
from . import chargingStation
from multiprocessing import Pool, cpu_count

# Function that create a timer to detect if the algorithm is taking too long


def start_clock():
    start_time = time.perf_counter()  # perf_counter() is the most precise clock
    return start_time


def end_clock(start_time):
    finish_time = time.perf_counter()
    return finish_time

# Function that returns the cpu usage


def cpu_start():
    # start_time = psutil.cpu_percent(interval=None)
    start_time = time.process_time()
    time.sleep(0.1)
    return start_time


def cpu_end():
    finish_time = time.process_time()
    # finish_time = psutil.cpu_percent(interval=None)
    return finish_time

# get ram usage


def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss


def find_optimal_start(dist_matrix, tsp_algorithm):
    min_cost = float('inf')
    optimal_start = 0

    for start in range(len(dist_matrix)):
        path = tsp_algorithm(dist_matrix, start)
        cost = sum(dist_matrix[path[i]][path[i + 1]]
                   for i in range(len(path) - 1))

        if cost < min_cost:
            min_cost = cost
            optimal_start = start

    return optimal_start

# Function that plots the path by calling the tspp algorithm

# --- max_distance is the maximum distance the robot can travel before needing to recharge -- #


def run(length, width, tspp_algorithm, max_distance):

    # generate a grid of points
    x = np.linspace(0, length - 1, length)
    y = np.linspace(0, width - 1, width)
    grid_points = np.array(list(itertools.product(x, y)))  # 2D array of points

    # compute distance matrix
    dist_matrix = squareform(pdist(grid_points))

    # Convert tsp_algorithm to a string
    algorithms = {
        'nn': tssp_algorithms.nearest_neighbour,
        'two-opt': tssp_algorithms.two_opt,
        'christofides': tssp_algorithms.christofides
    }

    # find optimal starting position
    optimal_start = find_optimal_start(dist_matrix, algorithms[tspp_algorithm])

    # solve, compare and compute elapsed time and CPU time and RAM usage

    # Start CPU usage monitoring
    start_cpu_usage = psutil.cpu_percent(interval=None, percpu=True)

    # measure memory usage before running the algorithm
    start_memory = get_memory_usage()

    start = start_clock()
    start_CPU = cpu_start()
    path = algorithms[tspp_algorithm](dist_matrix, optimal_start)
    end_CPU = cpu_end()
    elapsed_time = end_clock(start) - start

    cpu_percentages = ((end_CPU - start_CPU)/elapsed_time)*100

    # Measure memory usage after running the algorithm
    end_memory = get_memory_usage()

    # Calculate the memory usage (difference)
    memory_usage = end_memory - start_memory

    charging_stations, charging_station_routes = chargingStation.find_optimal_charging_station(
        path, grid_points, max_distance)

    # compute cost
    cost = sum(dist_matrix[path[i]][path[i + 1]]
               for i in range(len(path) - 1)) + 1

    # Create a plot filename
    plot_filename = f"{tspp_algorithm}_{length}x{width}.png"

    # Plot and save the image
    plot = plot_path(path, f"{tspp_algorithm} (Optimal start: {optimal_start})", color='blue',
                     filename=f"{tspp_algorithm}_path.csv", coordinates=grid_points, charging_stations=charging_stations, charging_station_routes=charging_station_routes)

    image_base64 = plot_to_base64_image(plot)

    return path, cost, elapsed_time, image_base64, cpu_percentages, memory_usage, charging_stations


def run_experiments_and_save_plot(number_of_points, tspp_algorithm):
    grid_sizes = range(1, number_of_points + 1)
    elapsed_times = []
    cpu_usages = []

    for size in grid_sizes:
        _, _, elapsed_time, _, cpu_usage, memory_usage, _ = run(
            size, size, tspp_algorithm, max_distance=1)
        elapsed_times.append(elapsed_time)
        cpu_usages.append(cpu_usage)  # store the CPU usage

    plot_C = plot_complexity(grid_sizes, elapsed_times,
                             f"{tspp_algorithm}: Time complexity")

    # Save the results to a CSV file
    with open('results.csv', 'w', newline='') as csvfile:
        fieldnames = ['grid_size', 'elapsed_time', 'cpu_usage', 'memory_usage']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(grid_sizes)):
            writer.writerow(
                {'grid_size': grid_sizes[i], 'elapsed_time': elapsed_times[i], 'cpu_usage': cpu_usages[i], 'memory_usage': memory_usage})

    return plot_C

# Plot time complexity


def plot_complexity(grid_sizes, elapsed_times, title):

    plot_C = plt.figure()

    # plot the data from the computed experimental grid_sizes and elapsed_times
    exp_data, = plt.plot(grid_sizes, elapsed_times,
                         marker='o', label='Experimental')

    # Plot the fitted polynomial
    if title == "nn: Time complexity":
        # Fit a quadratic polynomial to the data
        z = np.polyfit(grid_sizes, elapsed_times, 2)
        p = np.poly1d(z)
        # Create equations for the legend
        equation1 = f'y = {z[0]:.2e}x^2 + {z[1]:.2e}x + {z[2]:.2e}'
    elif title == "two-opt: Time complexity":
        # Fit a quadratic polynomial to the data
        z = np.polyfit(grid_sizes, elapsed_times, 2)
        p = np.poly1d(z)
        # Create equations for the legend
        equation1 = f'y = {z[0]:.2e}x^2 + {z[1]:.2e}x + {z[2]:.2e}'
    elif title == "christofides: Time complexity":
        # Fit a cubic polynomial to the data
        z = np.polyfit(grid_sizes, elapsed_times, 3)
        p = np.poly1d(z)
        # Create equations for the legend
        equation1 = f'y = {z[0]:.2e}x^3 + {z[1]:.2e}x^2\n + {z[2]:.2e}x + {z[3]:.2e}'

    # Plot the fitted curve for experimental data
    exp_curve, = plt.plot(grid_sizes, p(grid_sizes),
                          'g--', label="Theoretical")

    plt.xlabel('Grid size (NxN)')
    plt.ylabel('Elapsed time (seconds)')
    plt.title(title)
    plt.grid(True)

    # Add padding to prevent the y-axis label from being cropped
    plt.tight_layout()

    # Create a custom legend
    legend = plt.legend(
        [exp_data, exp_curve],
        ['Experimental', f'Theoretical: {equation1}'],
        loc='best',
        title='Legend'
    )

    # Save the results to a CSV file
    with open('results.csv', 'w', newline='') as csvfile:
        fieldnames = ['grid_size', 'elapsed_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(grid_sizes)):
            writer.writerow(
                {'grid_size': grid_sizes[i], 'elapsed_time': elapsed_times[i]})

    return plt  # return the plt instance


def plot_all_complexity(number_of_points):
    grid_sizes = range(1, number_of_points + 1)
    elapsed_times_nn = []
    elapsed_times_two_opt = []
    elapsed_times_christofides = []

    for size in grid_sizes:
        _, _, elapsed_time_nn, _ = run(size, size, "nn")
        _, _, elapsed_time_two_opt, _ = run(size, size, "two-opt")
        _, _, elapsed_time_christofides, _ = run(size, size, "christofides")
        elapsed_times_nn.append(elapsed_time_nn)
        elapsed_times_two_opt.append(elapsed_time_two_opt)
        elapsed_times_christofides.append(elapsed_time_christofides)

    plot_C = plt.figure()

    # plot the data from the computed experimental grid_sizes and elapsed_times for each algorithm
    plt.plot(grid_sizes, elapsed_times_nn,
             marker='o', label='Nearest Neighbor')
    plt.plot(grid_sizes, elapsed_times_two_opt, marker='s', label='Two-opt')
    plt.plot(grid_sizes, elapsed_times_christofides,
             marker='^', label='Christofides')

    plt.xlabel('Grid size (NxN)')
    plt.ylabel('Elapsed time (seconds)')
    plt.title("All Algorithms: Experimental Time Complexity")
    plt.grid(True)

    # Add padding to prevent the y-axis label from being cropped
    plt.tight_layout()

    # Create a legend
    plt.legend(loc='best', title='Legend')

    # Save the combined plot to a file
    plt.savefig('all_complexity.png')

    return plt

# Plot paths


def plot_path(path, title, color='blue', filename=None, coordinates=None, charging_stations=None, charging_station_routes=None):
    plot = plt_instance.figure()
    plt_instance.scatter(coordinates[:, 0], coordinates[:, 1],
                         c='red', label='Waypoints')
    
    need_to_charge_label_added = False
    charging_trajectory_label_added = False
    
    # place the charging stations
    if charging_stations is not None:
        for out_of_charge_point, charging_station in charging_stations.items():
            if not need_to_charge_label_added:
                plt_instance.scatter(coordinates[out_of_charge_point, 0], coordinates[out_of_charge_point, 1],
                                     c='green', marker='*', label='Need to charge...', s=100)
                need_to_charge_label_added = True
            else:
                plt_instance.scatter(coordinates[out_of_charge_point, 0], coordinates[out_of_charge_point, 1],
                                     c='green', marker='*', s=100)
            # plot charging station routes with thin purple lines
            if not charging_trajectory_label_added:
                plt_instance.plot([coordinates[out_of_charge_point, 0], coordinates[charging_station, 0]],
                                  [coordinates[out_of_charge_point, 1], coordinates[charging_station, 1]],
                                  c='purple', linestyle='dashed', linewidth=0.5, label='Charging Trajectory')
                charging_trajectory_label_added = True
            else:
                plt_instance.plot([coordinates[out_of_charge_point, 0], coordinates[charging_station, 0]],
                                  [coordinates[out_of_charge_point, 1], coordinates[charging_station, 1]],
                                  c='purple', linestyle='dashed', linewidth=0.5)

    # # place the charging stations
    # if charging_stations is not None:
    #     plt_instance.scatter(coordinates[charging_stations, 0], coordinates[charging_stations, 1],
    #                          c='green', marker='*', label='Need to charge...', s=100)

    # place the robot at the optimal start
    plt_instance.scatter(coordinates[0, 0], coordinates[0, 1],
                         c='yellow', marker='s', label='Robot')
    plt_instance.plot(coordinates[path, 0], coordinates[path, 1], c=color)

    # # plot charging station routes with thin purple lines
    # if charging_station_routes is not None:
    #     for route in charging_station_routes:
    #         plt_instance.plot(coordinates[route, 0], coordinates[route, 1],
    #                           c='purple', linestyle='dashed', linewidth=0.5, label='Charging Trajectory')

    plt_instance.title(title)
    plt_instance.xlabel("X")
    plt_instance.ylabel("Y")
    plt_instance.legend(loc='upper right')

    # Number the points according to the order of the path
    for i, point in enumerate(path):
        plt_instance.annotate(
            str(i), (coordinates[point, 0], coordinates[point, 1]), fontsize=8, ha='right')

    # # Add custom legend to avoid duplicate labels
    # handles, labels = plt_instance

    if filename:
        # save the 2D coordinates of the path to a csv file
        np.savetxt(filename, coordinates, delimiter=",")

    return plot


def plot_to_base64_image(image=None):
    buf = BytesIO()
    image.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return image_base64


def path_coordinates_to_csv_string(path, coordinates):
    ordered_path_coordinates = coordinates[path, :]
    csv_data = "\n".join(
        [f"{coord[0]},{coord[1]}" for coord in ordered_path_coordinates])
    return csv_data
