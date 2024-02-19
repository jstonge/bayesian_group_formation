"""
- Assume a social graph G underlies the collaboration that leads to the 
  hypergraph of academic papers H. The generative process is as follows:
    1. Social ties in science
        - Assume G is divided into B groups via SBM.
        - Random effects influencing individual propensite to initiate and receive connections
        - Being part of the same groups increase probably of a social ties 
        - Social ties increase the probability of collaboration on a paper
        - **Dyad-level effects**:
            - Some people are more likely to 'nominate' others to be on a paper
            - Reciprocity can be at work
        - **Individual features**:
            - Prestige
            - Productivity
            - Etc...
        - One hypothesis is that those features are correlated.
    2. Academic paper production
        - Instead of an observation model, we have hypergraph H
        - y[i,j] -> generate a tensor of observable variables x[i,j,q,t]
        
"""
import numpy as np
import pandas as pd
from scipy.stats import bernoulli, poisson
from numpy.linalg import cholesky
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Union

def inv_logit(x):
    return np.exp(x) / (1 + np.exp(x))

def rmvnorm(n, mean, cov):
    """Generate random multivariate normal vectors."""
    L = cholesky(cov)
    z = np.random.normal(size=(n, cov.shape[0]))
    return mean + np.dot(z, L.T)

def simulate_sbm_plus_srm_network(
    N_id: int,
    B: List[np.ndarray],
    V: int,
    groups: pd.DataFrame,
    sr_mu: List[float],
    dr_mu: List[float],
    sr_sigma: List[float],
    dr_sigma: float,
    sr_rho: float,
    dr_rho: float,
    mode: str,
    individual_predictors: Union[np.ndarray, None] = None,
    dyadic_predictors: Union[np.ndarray, None] = None,
    individual_effects: Union[np.ndarray, None] = None,
    dyadic_effects: Union[np.ndarray, None] = None
) -> Dict[str, Any]:
    """
    params:
    =======
      N_id: Number of individuals
      B: List of matrices that hold intercept and offset terms. Log-odds. The first matrix should be  1 x 1 with the value being the intercept term.
      V: Number of blocking variables in B.
      groups: Dataframe of the block IDs of each individual for each variable in B.
      sr_mu: Mean A vector for sender and receivier random effects. In most cases, this should be c(0,0).
      dr_mu: Mean A vector for dyadic random effects. In most cases, this should be c(0,0).
      sr_sigma: A standard deviation vector for sender and receivier random effects. The first element controls node-level variation in out-degree, the second in in-degree.
      dr_sigma: Standard deviation for dyadic random effects.
      sr_rho: Correlation of sender-receiver effects (i.e., generalized reciprocity).
      dr_rho: Correlation of dyad effects (i.e., dyadic reciprocity).
      mode: Outcome mode: can be "bernoulli" or "poisson".
      individual_predictors: An N_id by N_individual_parameters matrix of covariates.
      dyadic_predictors: An N_id by N_id by N_dyadic_parameters array of covariates.
      individual_effects: A 2 by N_individual_parameters matrix of slopes. The first row gives effects of focal characteristics (on out-degree). 
      dyadic_effects: An N_dyadic_parameters vector of slopes.
    """
    # B=B_list
    # sr_mu=[0,0]
    # dr_mu=[0,0]
    # sr_sigma=[0.3, 1.5]
    # dr_sigma=1
    # sr_rho=0.6 
    # dr_rho=0.7 
    # mode="bernoulli"
    # dyadic_predictors=None

    # Create correlation matrices (aka matrixes)
    Rho_sr = np.array([[1, sr_rho], [sr_rho, 1]])
    Rho_dr = np.array([[1, dr_rho], [dr_rho, 1]])

    # Varying effects on individuals
    sr_sigma_matrix = np.diag(sr_sigma)
    dr_sigma_matrix = np.diag([dr_sigma, dr_sigma])

    sr = np.zeros((N_id, 2))
    for i in range(N_id):
        sr[i, :] = rmvnorm(1, sr_mu, sr_sigma_matrix.dot(Rho_sr).dot(sr_sigma_matrix))[0]
        if individual_predictors is not None:
            sr[i, 0] += np.sum(individual_effects[0, :] * individual_predictors[i, :])
            sr[i, 1] += np.sum(individual_effects[1, :] * individual_predictors[i, :])

    y_true = np.zeros((N_id, N_id))
    p = np.zeros((N_id, N_id))
    for i in range(N_id-1):
        for j in range(i+1, N_id):
            
            # Dyadic effects
            dr_scrap = rmvnorm(1, dr_mu, dr_sigma_matrix.dot(Rho_dr).dot(dr_sigma_matrix))[0]
            if dyadic_predictors is not None:
                dr_scrap[0] += np.sum(dyadic_effects * dyadic_predictors[i, j, :])
                dr_scrap[1] += np.sum(dyadic_effects * dyadic_predictors[j, i, :])

            B_i_j = B_j_i = np.zeros(V)
            for v in range(V):
                # v=0
                B_i_j[v] = B[v][groups.iloc[i, v]-1, groups.iloc[j, v]-1]
                B_j_i[v] = B[v][groups.iloc[j, v]-1, groups.iloc[i, v]-1]

            dr_ij = dr_scrap[0] + np.sum(B_i_j)
            dr_ji = dr_scrap[1] + np.sum(B_j_i)

            # Simulate outcomes
            if mode == "bernoulli":
                p[i, j] = inv_logit(sr[i, 0] + sr[j, 1] + dr_ij)
                y_true[i, j] = bernoulli.rvs(p[i, j])
                p[j, i] = inv_logit(sr[j, 0] + sr[i, 1] + dr_ji)
                y_true[j, i] = bernoulli.rvs(p[j, i])

            elif mode == "poisson":
                p[i, j] = np.exp(sr[i, 0] + sr[j, 1] + dr_ij)
                y_true[i, j] = poisson.rvs(p[i, j])
                p[j, i] = np.exp(sr[j, 0] + sr[i, 1] + dr_ji)
                y_true[j, i] = poisson.rvs(p[j, i])

    # Set diagonal elements to 0, as per original R code
    np.fill_diagonal(y_true, 0)
    np.fill_diagonal(p, 0)

    
    return {
        'network': y_true,
        'tie_strength': p,
        'group_ids': groups,
        'individual_predictors': individual_predictors,
        'dyadic_predictors': dyadic_predictors,
        'sr': sr,
        # 'dr': dr, # 'dr' was not explicitly calculated as a return object in the provided code but could be included if needed
        # 'samps': samps, # 'samps' used in binomial mode needs to be defined or passed if you want to return it
    }


# VISUALIZE TRUE NETWORK

# Initialization parameters
V = 1  # One blocking variable
G = 3  # Three categories in this blocking variable
N_id = 30  # Number of individuals

# Simulate group assignments
np.random.seed(0)  # For reproducibility
clique = np.random.choice(range(1, G + 1), N_id, replace=True)

# Block matrix B initialization
B = np.full((G, G), -7)
np.fill_diagonal(B, -1.5)

# Convert B into the expected format for the simulation function
B_list = [B]

# Groups dataframe equivalent in Python
groups = pd.DataFrame({'clique': clique})

# Individual predictors and effects initialization
individual_predictors = np.random.normal(0, 1, size=(N_id, 1))
individual_effects = np.array([[1.7], [0.3]])

# Simulate social graph using SRM network model
A = simulate_sbm_plus_srm_network(
    N_id=N_id, B=B_list, V=V, 
    groups=groups, sr_mu=[0,0], dr_mu=[0,0],
    sr_sigma=[0.3, 1.5], dr_sigma=1, sr_rho=0.6, 
    dr_rho=0.7, mode="bernoulli",
    individual_predictors=individual_predictors, 
    individual_effects=individual_effects
)

def plot_G(A):
    # Creating the graph with networkx
    G = nx.from_numpy_matrix(A['network'])

    # Assigning colors based on group IDs, for networkx plotting
    color_map = ['turquoise' if clique[i] == 1 else 'red' if clique[i] == 2 else 'gold' for i in range(N_id)]
    degrees = dict(G.degree())
    scaled_degrees = {node: degree*20 for node, degree in degrees.items()}
    node_sizes = [scaled_degrees[node] for node in G.nodes()]

    # Plotting
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos=nx.kamada_kawai_layout(G),
            node_color=color_map,
            edgecolors='black', 
            node_size=node_sizes,
            arrowstyle='->', 
            arrowsize=12)
    plt.show()

plot_G(A)

# now that we have a social network, how do people 
# choose a set of collaborators on a paper?
y = A['network']
G = nx.from_numpy_matrix(y)

# I THINK THIS WILL GO HERE TOO
# simulate productivity covariate
# W <- rnorm(N) # standardized relative wealth in community
# if (missing(bWG))
#     bWG <- rep(0,M) # effect of productivity
# if (missing(bWR))
#     bWR <- rep(0,M) # effect of productivity

hyperedges = []
for i in G.nodes():
    # How many papers this person will have depend on indidividual features
    poisson()
    instigator = np.random.choice(N_id) # should depend on productivity, age, etc.




## HYPERGRAPH OF ACADEMIC PAPERS


for node, gid in zip(G.nodes, A['group_ids']['clique']):
    G.nodes[node].update({'group': gid})



