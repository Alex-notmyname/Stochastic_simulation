import simpy
import multiprocessing as mp
import random
import numpy as np
import time

RANDOM_SEED = 42
MU = 1
RHO = 0.9
LAMBDA = RHO * MU
CUSTOMERS = 50000
SERVICE_TIME = 1

RHO_list = [0.85, 0.9, 0.95, 0.99]

class Customer():
    """_Start one simpy simulation_
    """

    def __init__(self, env, customers, arrive_rate, mu, servers, \
                 kendall_notation='M/M/n', SJF=False):
        """_Instantiate the Customer class_

        Args:
            env (_simpy.Environment()_): _simpy.Environment()_
            customers (_int_): _Number of customers in total_
            arrive_rate (_float_): _Arrival rate of new customer (lambda)_
            mu (_float_): _service rate (mu)_
            servers (_simpy.Resource()_): _simpy.Resource()_
            SJF (bool, optional): _Shortest Job First Policy?_. Defaults to False.
        """
        self.env = env
        # Start the run process everytime an instance is created.
        self.action = env.process(self.source())
        # Number of customers in total
        self.customers = customers
        # Arriving rate of new customers
        self.Lambda = arrive_rate
        # Server mu
        self.mu = mu
        # simpy.Resource()
        self.servers = servers
        # Monitoring the waiting time for each customer
        self.waiting_time = np.zeros(customers)
        # The experiment mode
        self.kendall_notation = kendall_notation
        # shortest job first scheduling?
        self.SJF = SJF

    def customer(self, env, name, servers, mu):
        """_Generate an instance of a customer and
            add it to the queue_

        Args:
            env _simpy.Environment()_): _simpy.Environment()_
            name (_int_): _the sequence of customer_
            servers (_simpy.Resource()_): _simpy.Resource()_
            mu (_float_): _service rate (mu)_

        Yields:
            _simpy.Environment().timeout()_: _Occupy the server
            until the service is completed_
        """
        arrive = env.now
        # print('%7.4f %s: Here I am' % (arrive, name))

        # Complete Markovian system
        if self.kendall_notation == 'M/M/n':  
            tib = random.expovariate(mu) # Service time (mu)
        # Markovian arrival rate, Deterministic service time
        elif self.kendall_notation == 'M/D/n':
            tib = 1/mu
        # Hyperexponential service time distribution
        elif self.kendall_notation == 'M/H/n':
            if random.random() <= 0.75:
                tib = random.expovariate(1.0)
            else:
                tib = random.expovariate(1.0 / 5.0)
            
        if self.SJF == True:
            with servers.request(priority=tib) as req:
                yield req

                wait = env.now - arrive
                # Store the waiting time of customer_i
                self.waiting_time[name] = wait

                # print('%7.4f %s: Waited %6.3f' % (env.now, name, wait))

                yield env.timeout(tib)
                # print('%7.4f %s: Finished' % (env.now, name))

        else:
            with servers.request() as req:
                yield req

                wait = env.now - arrive
                # Store the waiting time of customer_i
                self.waiting_time[name] = wait

                # print('%7.4f %s: Waited %6.3f' % (env.now, name, wait))

                yield env.timeout(tib)
                # print('%7.4f %s: Finished' % (env.now, name))

    def source(self):
        """_Add new customers to the system until no customers left_

        Yields:
            _simpy.Environment().timeout()_: _Occupy the queue
            until new customer join the queue_
        """

        for i in range(self.customers):
            c = self.customer(self.env, i, self.servers, self.mu)
            self.env.process(c)
            # New customer join in line
            t = random.expovariate(self.Lambda)
            yield self.env.timeout(t)


def Q2_main():
    """_Experiments for the question 2_

    Returns:
        _np.ndarray_: _A np.ndarray contains waiting time for each customer_
    """
    n_servers = [1, 2, 4]
    W_rho = []

    for i in range(len(n_servers)):
        W = np.concatenate([np.zeros(CUSTOMERS, dtype=np.float64)[np.newaxis, :]] * len(RHO_list))
        for j, rho in enumerate(RHO_list):
            LAMBDA = rho * MU
            # Initialize environment
            env = simpy.Environment()
            # Initialize servers
            server = simpy.Resource(env, capacity=n_servers[i])
            # Instantiate the customer class
            customer = Customer(env, CUSTOMERS, LAMBDA*n_servers[i], MU, server)
            # Run simulation
            env.run()

            W[j, :] = customer.waiting_time
            # print(customer.waiting_time)

        W_rho.append(W)
    
    return W_rho


def Q3_main():
    """_Experiments for the question 3_

    Returns:
        _np.ndarray_: _A np.ndarray contains waiting time for each customer_
    """
    W = np.concatenate([np.zeros(CUSTOMERS, dtype=np.float64)[np.newaxis, :]] * len(RHO_list))

    for i, rho in enumerate(RHO_list):
        LAMBDA = rho * MU
        # Initialize environment
        env = simpy.Environment()
        # Initialize servers
        server = simpy.PriorityResource(env, capacity=1)
        # Instantiate the customer class
        customer = Customer(env, CUSTOMERS, LAMBDA*1, MU, server, SJF=True)
        # Run simulation
        env.run()

        W[i, :] = customer.waiting_time
        # print(customer.waiting_time)
    
    return W


def Q4_main():
    """_Experiments for the question 3_

    Returns:
        _np.ndarray_: _A np.ndarray contains waiting time for each customer_
    """
    n_servers = [1, 2, 4]

    W = np.concatenate([np.zeros(CUSTOMERS, dtype=np.float64)[np.newaxis, :]] * 6)

    # M/D/n Experiment
    for i in range(len(n_servers)):
        # Initialize environment
        env = simpy.Environment()
        # Initialize servers
        server = simpy.Resource(env, capacity=n_servers[i])
        # Instantiate the customer class
        customer = Customer(env, CUSTOMERS, LAMBDA*n_servers[i], MU, server,\
                            kendall_notation='M/D/n')
        # Run simulation
        env.run()

        W[i, :] = customer.waiting_time
        # print(customer.waiting_time)

    # Hyperexponential service time distribution
    for i in range(len(n_servers)):
        # Initialize environment
        env = simpy.Environment()
        # Initialize servers
        server = simpy.Resource(env, capacity=n_servers[i])
        # Instantiate the customer class
        customer = Customer(env, CUSTOMERS, LAMBDA*n_servers[i], MU, server,\
                            kendall_notation='M/H/n')
        # Run simulation
        env.run()

        W[i+3, :] = customer.waiting_time
        # print(customer.waiting_time)

    
    return W


if __name__ == '__main__':
    # count time
    start_time = time.time()

    pool = mp.Pool()

    # repeat experiments for 100 times (stochasticity)
    repetition = 100
    n_servers = [1, 2, 4]
    
    # Q2 Experiment
    W2 = pool.starmap(Q2_main, [() for _ in range(repetition)])
    W2 = np.array(W2)
    for i in range(3):
        waiting_time = W2[:, i]
        np.save('data/Q2/M_M_%s' % n_servers[i], waiting_time)

    # Q3 Experiment
    W3 = pool.starmap(Q3_main, [() for _ in range(repetition)])
    np.save('data/Q3/M_M_1_priority', W3)

    # Q4 Experiment
    W4 = pool.starmap(Q4_main, [() for _ in range(repetition)])
    W4 = np.array(W4)
    for i in range(6):
        waiting_time = W4[:, i, :]
        if i <= 2:
            np.save('data/Q4/M_D_%s' % n_servers[i], waiting_time)
        else:
            np.save('data/Q4/M_H_%s' % n_servers[i-3], waiting_time)

    print('----- %s seconds -----' % (time.time() - start_time))