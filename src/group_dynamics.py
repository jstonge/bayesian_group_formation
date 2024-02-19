import numpy as np
from scipy.stats import poisson, geom


def sample_next_group(g, X):
    """
    g: group membership
    X: transition matrix
    """
    num_groups = X.shape[0]
    return np.random.choice(np.arange(num_groups), p=X[g,:])


def sample_random_markov_transition_matrix(num_groups):
    
    X = np.random.rand(num_groups+1,num_groups+1)
    
    # make the last group a sink, so that individuals leave the system and don't return
    X[-1,:] = 0
    X[-1,-1] = 1

    # normalize transition matrix
    X = X / X.sum(axis=1)[:,None]
    return X


def sample_prior_prob_b(num_groups,num_nodes, T=100):

    """
    Simulate the dyanmics of a group of individuals.

    Let b[i,t] be group membership of individual i at time t, with the -1 
    label signifying that the individual is not in the system.
    Allows for individuals to join and leave the system.

    n1 2 2 2 3 3 3 3 4 4 4 -1 -1
    n2 2 2 2 2 2 3 3 3 3 4 -1 -1
    n3 -1 -1 1 1 1 1 1 1 1 1 1 1 
    n4 1 1 1 1 1 1 1 1 1 1 1 1 1


    t = 1,2,3; groups = [1,2]
    t = 4; groups = [1,2,3]
    t = 5; [1,3]
    t = 6; [1,3,4]

       g1   g2  g3  -1
    g1 0.9  0.0 0.0 0.1
    g2 0.1  0.6  0.1 0.1
    g3 0.05 0.05 0.85 0.05
    -1 0.0  0.0  0.0 1.0

    allows for prestige dynamics
    could be a function of time

    memoryless transitions: most transitions have a characteristic length

    -1 means retired from science
    -2 means hasn't yet joined science
    """

    # markov transition matrix from group to group
    X = sample_random_markov_transition_matrix(num_groups)

    entrance_times = [np.random.choice(np.arange(T)) for _ in range(num_nodes)]

    b = np.zeros((num_nodes,T),dtype=int) - 2

    for i in range(num_nodes):
        # choose group membership at entrance time
        b[i,entrance_times[i]] = np.random.randint(0,num_groups)
        t = entrance_times[i]

        # choose group membership for the rest of the time
        while t < T:

            # choose time until next transition
            dT = poisson.rvs(3)
            for _ in range(dT):
                t += 1
                if t >= T:
                    break
                # choose group membership at time t
                b[i,t] = b[i,t-1]

            # choose next group
            if t < T:
                b[i,t] = sample_next_group(b[i,t-1], X)

    # replaces all instances of num_groups with the value -1
    b[b==num_groups] = -1
    return b


def get_population_time_series(b):

    # boolean array indicating whether or not the individual is in the system
    x = b >= 0
    return x.sum(axis=0)

if __name__ == "__main__":
    b = sample_prior_prob_b(3,40,T=100)
    # print(b)
    #print(get_population_time_series(b))