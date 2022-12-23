import tsplib95
import numpy as np
import random
import time
from multiprocessing.pool import Pool

def load_TSP(file_path: str) -> np.ndarray:
    """This function takes the file path of TSP problem set as
        input, load the file then take the x and y coordinates of
        cities as output

    Args:
        file_path (str): the file path of TSP problem set

    Returns:
        numpy.array: Euclidean coordinates of cities
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


def distance_matrix(x_coordinates: np.ndarray, y_coordinates: np.ndarray) -> np.ndarray:
    """This function takes the x coordinates and y coordinates
       arrays as input, then compute the distance matrix as output

    Args:
        x_coordinates (numpy.array): Array containing the x_coordinates
        for all cities
        y_coordinates (numpy.array): Array containing the y_coordinates
        for all cities

    Returns:
        numpy.ndarray: The distance matrix for all cities
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


def objective_function(solution: list, distance_matrix: np.ndarray) -> float:
    """This function takes current solution (the city visiting
       route) and the distance matrix as input, then compute
       the total traveling distance for current solution as output

    Args:
        solution (numpy.array): The city visiting route
        distance_matrix (numpy.ndarray): The distance matrix

    Returns:
        float: the total traveling distance for current solution
    """

    # Create the route tuple of adjcent cities ([1, 2], [2, 3], ...)
    route = np.array((solution[:-1], solution[1:])).T

    # Compute and return the total distance
    return sum(distance_matrix[route[:, 0], route[:, 1]])


def log_Temp(n: int, a: float, b: float) -> float:
    """Calculate temperature based on T = a/log(n+b)

    Args:
        n (int): the iteration number
        a (float): hyperparameter
        b (float): hyperparameter

    Returns:
        float: current step temperature
    """
    T = a / np.log(n + b)
    return T

def fast_Temp(t: float, n: int) -> float:
    """Calculate temperature based on fast annealing
       T = T / (n+1)

    Args:
        t (float): last step temperature
        n (int): the iteration number

    Returns:
        float: current step temperature
    """
    T = t / float(n + 1)
    return T

def exp_Temp(t0: float, alpha: float, n: float) -> float:
    """Calculate temperature based on exponential annealing
       T = T0 *alpha^n

    Args:
        t0 (float): initial temperature
        alpha (float): hyperparameter
        n (int): the iteration number

    Returns:
        float: current step temperature
    """
    return t0*(alpha**n)


def adaptive_Temp(tk: float, fm: float, fk: float) -> float:
    """Calculate temperature based on adaptive annealing
       T = (1 + (fk-fm)/fk)*Tk

    Args:
        tk (float): Temperature value, here we use the logarithm temp
        fm (float): the best cost value so far
        fk (float): the cost value at current step

    Returns:
        float: current step temperature
    """
    mu = 1 + (fk-fm)/fk
    return mu*tk


def two_opt_swap(route: list, v1: int, v2: int) -> np.ndarray:
    """Apply the 2-opt swapping method to the route list,
    from route[v1] to route[v2]. Details can be found
    at https://en.wikipedia.org/wiki/2-opt

    Args:
        route (list): Current traveling route
        v1 (int): the start position of 2-opt strategy
        v2 (int): the end position of 2-opt strategy

    Returns:
        np.ndarray: new traveling route
    """
    return np.concatenate((route[0:v1], route[v2:-len(route) + v1-1:-1],route[v2 + 1:len(route)]))


def main(num_ite: int, TSP_COORDINATES: np.ndarray, annealing_type: str) -> list:
    """Implement the Simulated Annealing

    Args:
        num_ite (int): number of iterations
        TSP_COORDINATES (numpy.ndarray): the X and Y coordinates of cities
        annealing_type (str): the annealing type used

    Returns:
        list: [best solution, lowest cost, record of temperature]
    """
    # Load the TSP settings
    node_x, node_y = TSP_COORDINATES[0], TSP_COORDINATES[1]
    
    # Compute the distance matrix
    dists_matrix = distance_matrix(node_x, node_y)
    
    # Initialize the route
    solution = np.arange(0, len(node_x))
    # Randomly permute the route
    rng = np.random.default_rng()
    solution = rng.permuted(solution)
    # The salesman must return to the start city
    best_solution = np.append(solution, solution[0])
    
    ################################## Start simulation ##################################
    
    # Compute the initial cost (total distance)
    best_cost = objective_function(best_solution, dists_matrix)
    T0 = 1000
    T = T0
    T_record = np.zeros(num_ite)
    # 2-opt swap
    for i in range(num_ite):
    # Loop until max_iteration number has reached
        # Generate RV (2-opt potisions)
        v1 = random.randint(0, len(best_solution)-1)
        v2 = random.randint(v1, len(best_solution)-1)
        # Generate new solution by 2-opt
        new_solution = two_opt_swap(best_solution, v1, v2)
        # Calculate new cost
        new_cost = objective_function(new_solution, dists_matrix)
        cost_diff = new_cost - best_cost
        
        T_record[i] = T
        
        # Start simulated annealing
        # Compute current temperature
        if annealing_type != 'greedy':
            
            # Acceptance probability
            P = np.exp(-cost_diff/T)
            # Accept or not?
            if random.random() < P:
                best_solution = new_solution
                best_cost = new_cost
            
            if annealing_type == 'log':
                T = log_Temp(i+1, a=41.5, b=1.1)
            elif annealing_type == 'fast':
                T = fast_Temp(T0, i+1)
            elif annealing_type == 'exp':
                T = exp_Temp(T0, 0.85, i+1)
            elif annealing_type == 'adaptive':
                Tk = log_Temp(i+1, a=41.5, b=1.1)
                T = adaptive_Temp(Tk, best_cost, new_cost)
                
        else:
            if cost_diff < 0:
                best_solution = new_solution
                best_cost = new_cost
            
    return [best_solution, best_cost, T_record]


if __name__ == '__main__':
    start_time = time.time()
    
    EIL51_COORDINATES = load_TSP('TSP-Configurations/eil51.tsp')
    A280_COORDINATES = load_TSP('TSP-Configurations/a280.tsp')
    PCB442_COORDINATES = load_TSP('TSP-Configurations/pcb442.tsp')
    
    ite_list = np.linspace(1, 500000, 101, dtype=int)
    
    repetition = 10
    
    ############################# Test scaling on problem size #############################
    
    eil51_args = zip(ite_list, [EIL51_COORDINATES] * len(ite_list), ['log']*len(ite_list))
    a280_args = zip(ite_list, [A280_COORDINATES] * len(ite_list), ['log']*len(ite_list))
    pcb442_args = zip(ite_list, [PCB442_COORDINATES] * len(ite_list), ['log']*len(ite_list))
    
    with Pool() as pool:
        data_eil51 = pool.starmap(main, eil51_args)
        data_a280 = pool.starmap(main, a280_args)
        data_pcb442 = pool.starmap(main, pcb442_args)
    
    np.save('data_eil51.npy', data_eil51, allow_pickle=True)
    np.save('data_a280.npy', data_a280, allow_pickle=True)
    np.save('data_pcb442.npy', data_pcb442, allow_pickle=True)
    
    
    ############################# Cooling scheduling experiments #############################

    # log_args = zip(ite_list, [A280_COORDINATES] * len(ite_list), ['log']*len(ite_list))
    # fast_args = zip(ite_list, [A280_COORDINATES] * len(ite_list), ['fast']*len(ite_list))
    # exp_args = zip(ite_list, [A280_COORDINATES] * len(ite_list), ['exp']*len(ite_list))
    # adaptive_args = zip(ite_list, [A280_COORDINATES] * len(ite_list), ['adaptive']*len(ite_list))
    
    # greedy_args = zip(ite_list, [A280_COORDINATES] * len(ite_list), ['greedy']*len(ite_list))
    
    # data_log, data_fast, data_exp, data_adaptive, data_greedy = [], [], [], [], []
    
    # with Pool(12) as pool:
    #     for i in range(1):
    #         data_log.append(pool.starmap(main, log_args))
            # data_fast.append(pool.starmap(main, fast_args))
            # data_exp.append(pool.starmap(main, exp_args))
            # data_adaptive.append(pool.starmap(main, adaptive_args))
            
            # data_greedy.append(pool.starmap(main, greedy_args))
    
    # np.save('Cooling_scheduling_results/data_log.npy', data_log, allow_pickle=True)
    # np.save('Cooling_scheduling_results/data_fast.npy', data_fast, allow_pickle=True)
    # np.save('Cooling_scheduling_results/data_exp.npy', data_exp, allow_pickle=True)
    # np.save('Cooling_scheduling_results/data_adaptive.npy', data_adaptive, allow_pickle=True)
    
    # np.save('Cooling_scheduling_results/data_greedy.npy', data_greedy, allow_pickle=True)
    
    # def main(num_ite, TSP_COORDINATES, annealing_type, seed=42):
    

    print('----- %s seconds -----' % (time.time() - start_time))
            