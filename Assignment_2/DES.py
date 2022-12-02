import simpy
import multiprocessing as mp
import random
import numpy as np
import time

RANDOM_SEED = 42
MU = 1
RHO = 0.9
LAMBDA = RHO * MU
CUSTOMERS = 10000
SERVICE_TIME = 1

class Customer():

    def __init__(self, env, customers, arrive_rate, mu, servers, \
                 kendall_notation='M/M/n', SJF=False) -> None:
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

        arrive = env.now
        # print('%7.4f %s: Here I am' % (arrive, name))

        # Complete Markovian system
        if self.kendall_notation == 'M/M/n':  
            tib = random.expovariate(mu) # Service time (mu)
        # Markovian arrival rate, Deterministic service time
        elif self.kendall_notation == 'M/D/n':
            tib = SERVICE_TIME
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

        for i in range(self.customers):
            c = self.customer(self.env, i, self.servers, self.mu)
            self.env.process(c)
            # New customer join in line
            t = random.expovariate(self.Lambda)
            yield self.env.timeout(t)


def Q2_main():

    # random.seed(RANDOM_SEED)

    n_servers = [1, 2, 4]

    W = np.concatenate([np.zeros(CUSTOMERS, dtype=np.float64)[np.newaxis, :]] * len(n_servers))

    for i in range(len(n_servers)):
        # Initialize environment
        env = simpy.Environment()
        # Initialize servers
        server = simpy.Resource(env, capacity=n_servers[i])
        # Instantiate the customer class
        customer = Customer(env, CUSTOMERS, LAMBDA*n_servers[i], MU, server)
        # Run simulation
        env.run()

        W[i, :] = customer.waiting_time
        # print(customer.waiting_time)
    
    return W


def Q3_main():

    W = np.zeros(CUSTOMERS, dtype=np.float64)

    # Initialize environment
    env = simpy.Environment()
    # Initialize servers
    server = simpy.PriorityResource(env, capacity=1)
    # Instantiate the customer class
    customer = Customer(env, CUSTOMERS, LAMBDA*1, MU, server, SJF=True)
    # Run simulation
    env.run()

    W = customer.waiting_time
    # print(customer.waiting_time)
    
    return W


def Q4_main():

    # random.seed(RANDOM_SEED)

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
    W2 = pool.starmap(Q2_main, [() for _ in range(repetition)])
    W3 = pool.starmap(Q3_main, [() for _ in range(repetition)])
    W4 = pool.starmap(Q4_main, [() for _ in range(repetition)])
    W2 = np.array(W2)
    W4 = np.array(W4)

    np.save('data/Q3/M_M_1_priority', W3)

    n_servers = [1, 2, 4]
    for i in range(3):
        waiting_time = W2[:, i, :]
        np.save('data/Q2/M_M_%s' % n_servers[i], waiting_time)

    for i in range(6):
        waiting_time = W4[:, i, :]
        if i <= 2:
            np.save('data/Q4/M_D_%s' % n_servers[i], waiting_time)
        else:
            np.save('data/Q4/M_H_%s' % n_servers[i-3], waiting_time)

    print('----- %s seconds -----' % (time.time() - start_time))