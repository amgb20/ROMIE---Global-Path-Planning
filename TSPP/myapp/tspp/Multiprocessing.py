# def process_grid_size(args):
#     size, tspp_algorithm = args
#     _, _, elapsed_time, _ = run(size, size, tspp_algorithm)
#     return elapsed_time

# # function trying to use multiprocessing to run the experiments in parallel
# def run_parallel_experiments(number_of_points, tspp_algorithm):
#     grid_sizes = range(1, number_of_points + 1)
#     elapsed_times = []

#     # Set the number of processes to the minimum between the number of available CPU cores and a certain limit (e.g., 4).
#     num_processes = min(cpu_count(), 4)

#     with Pool(processes=num_processes) as pool:
#         results = pool.map(process_grid_size, [
#                            (size, tspp_algorithm) for size in grid_sizes])
#         elapsed_times.extend(results)

#     plot_para = plot_complexity(grid_sizes, elapsed_times,
#                                 f"{tspp_algorithm}: Time complexity")

#     # Save the results to a CSV file
#     with open('para.csv', 'w', newline='') as csvfile:
#         fieldnames = ['grid_size', 'elapsed_time']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         for i in range(len(grid_sizes)):
#             writer.writerow(
#                 {'grid_size': grid_sizes[i], 'elapsed_time': elapsed_times[i]})

#     return plot_para