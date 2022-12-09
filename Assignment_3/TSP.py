import tsplib95
import numpy as np
import random
import time

def load_TSP(file_path: str):
    """_This function takes the file path of TSP problem set as
        input, load the file then take the x and y coordinates of
        cities as output_

    Args:
        file_path (_str_): _the file path of TSP problem set_

    Returns:
        _numpy.array_: _Euclidean coordinates of cities_
    """
    
    # Load the TSP problem set
    problem = tsplib95.load(file_path)
    # Count the number of cities
    n_cities = len(problem.node_coords)
    
    # Initialize the x and y coordinates
    x_coordinates = np.zeros(n_cities)
    y_coordinates = np.zeros(n_cities)

    # Load the x and y coordinates for each city as output
    for i in range(n_cities):
        x_coordinates[i] = list(problem.node_coords.items())[i][1][0]
        y_coordinates[i] = list(problem.node_coords.items())[i][1][1]
        
    return x_coordinates, y_coordinates


def distance_matrix(x_coordinates, y_coordinates):
    """_This function takes the x coordinates and y coordinates
        arrays as input, then compute the distance matrix as output_

    Args:
        x_coordinates (_numpy.array_): _Array containing the x_coordinates
        for all cities_
        y_coordinates (_numpy.array_): _Array containing the y_coordinates
        for all cities_

    Returns:
        _numpy.ndarray_: _The distance matrix for all cities_
    """

    # initialize the distance matrix
    dists_matrix = np.zeros((len(x_coordinates), len(x_coordinates)))

    for i in range(len(x_coordinates)):
        # compute (x1-x2)^2
        dists_matrix[i] = (x_coordinates - x_coordinates[i])**2
        # compute (y1-y2)^2
        dists_matrix[i] += (y_coordinates - y_coordinates[i])**2
    
    # compute sqrt((x1-x2)^2 + (y1-y2)^2) (Euclidean distance)
    return np.sqrt(dists_matrix)


def objective_function(solution, distance_matrix) -> float:
    """_This function takes current solution (the city visiting
        route) and the distance matrix as input, then compute
        the total traveling distance for current solution as output_

    Args:
        solution (_numpy.array_): _The city visiting route
        distance_matrix (_numpy.ndarray_): _The distance matrix_

    Returns:
        _float_: _the total traveling distance for current solution_
    """

    # Create the route tuple of adjcent cities ([1, 2], [2, 3], ...)
    route = np.array((solution[:-1], solution[1:])).T

    # Compute and return the total distance
    return sum(distance_matrix[route[:, 0], route[:, 1]])


def Temperature(n, a, b):
    T = a / np.log(n + b)
    return T

def fast_Temp(t, n):
    T = t / float(n + 1)
    return T


def two_opt_swap(route, v1, v2):
    
    return np.concatenate((route[0:v1], route[v2:-len(route) + v1-1:-1],route[v2 + 1:len(route)]))


def main(a, b, num_ite):
    # Load the TSP settings
    node_x, node_y = load_TSP('TSP-Configurations/a280.tsp')
    
    # Compute the distance matrix
    dists_matrix = distance_matrix(node_x, node_y)
    
    # Initialize the route
    solution = np.arange(0, len(node_x))
    # Randomly permute the route
    rng = np.random.default_rng()
    solution = rng.permuted(solution)
    # The salesman must return to the start city
    best_solution = np.append(solution, solution[0])
    
    # Start simulation
    counter = 0
    cost_record = []
    # Compute the initial cost (total distance)
    best_cost = objective_function(best_solution, dists_matrix)
    T0 = 10000
    # 2-opt swap
    while True:
        for v1 in range(1, len(best_solution)-2): # From each city except the first and last,
            for v2 in range(v1+1, len(best_solution)-1): # to each of the cities following,
                # Generate new solution by 2-opt
                new_solution = two_opt_swap(best_solution, v1, v2)
                # Calculate new cost
                new_cost = objective_function(new_solution, dists_matrix)
                cost_diff = new_cost - best_cost
                
                # Start simulated annealing
                # Compute current temperature
                T = Temperature(counter, a, b)
                # T = fast_Temp(T0, counter)
                # Acceptance probability
                P = min(np.exp(-cost_diff/T), 1)
                # Accept or not?
                if random.random() < P:
                    best_solution = new_solution
                    best_cost = new_cost
                    
                # increase counter
                counter += 1
                cost_record.append(best_cost)
                
                if counter == num_ite:
                    # Break out
                    return np.append(solution, solution[0]), best_solution, cost_record


if __name__ == '__main__':
    
    start_time = time.time()
    
    a = 1
    b = 20
    
    initial_sol, best_sol, cost_record = main(a, b, 200000)
    
    np.save('initial_sol.npy', initial_sol)
    np.save('best_sol.npy', best_sol)
    np.save('costs.npy', cost_record)
    
    print('----- %s seconds -----' % (time.time() - start_time))
            