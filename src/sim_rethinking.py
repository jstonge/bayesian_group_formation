"""
Generative process for producing papers, given a social network and group memberships.
1. Simulate a simple graph with group memberships.
 - All groups have the same size and status
 - Ties between individuals depend on SBM parameters.
2. From the social networks, simulate the publication process.
 - The output is {individual: [collab_set_1, collab_set_2, ...]}
 - Number of individuals in collab depends on {absence, weak, strong} norms,
   which define lambda in Poisson distribution {1,3,5}.
 - More likely to publish with people within group, but can publish with anyone.

TODO:
 - Add heterogeneity in social networks:
    - Some people more likely to have social ties because of their
      - Status (PI, PostDoc, PhD) 
      - Author age (older researchers should have more connections), 
      - Perhaps just as a general tendency (some people want more friends).
    - We could probably make it higher-order; some synergies with triangles.
 - Add heterogeneity in publication process:
   - Affiliation (people more likely to reach out to individuals from prestigious institutions)
   - Correlation between groups; some groups like to work together;
     - Especially if current PI belonged to previous groups...
   - Toolset; some people are more likely to be invited if their toolset is useful.
   - Reciprocity; I invite you on my papers if you invite me on your papers.
"""
import numpy as np
import pandas as pd
from scipy.stats import bernoulli, poisson
from numpy.linalg import cholesky
import networkx as nx
import matplotlib.pyplot as plt


def inv_logit(x):
    return np.exp(x) / (1 + np.exp(x))

def rmvnorm(n, mean, cov):
    """Generate random multivariate normal vectors."""
    L = cholesky(cov)
    z = np.random.normal(size=(n, cov.shape[0]))
    return mean + np.dot(z, L.T)

# Initialization parameters
V = 1  # One blocking variable
G = 3  # Three categories in this blocking variable
N_id = 30  # Number of individuals

# Simulate group assignments
groups = np.random.choice(range(1, G + 1), N_id, replace=True)

# Block matrix B initialization
B = np.full((G, G), -7)
np.fill_diagonal(B, -1.5)

# Individual predictors and effects initialization
individual_predictors = np.random.normal(0, 1, size=(N_id, 1))
individual_effects = np.array([[1.7], [0.3]])

# params from STRAND; 
# In their case, it was how much correlation there was between, say, someone
# who send a lot of gift with receiver sending back gifts (aka reciprocity).
# Can we say something similar?
# What is the correlation between someone who always participate to your
# paper and you'll reciprocate... maybe? 
sr_mu=[0,0] 
dr_mu=[0,0]
sr_sigma=[0.3, 1.5] 
dr_sigma=1 # Standard deviation for dyadic random effects.
sr_rho=0.6 # Correlation of sender-receiver effects (i.e., generalized reciprocity).
dr_rho=0.7 # Correlation of dyad effects (i.e., dyadic reciprocity).

# Create correlation matrices 
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
        
        # if dyadic_predictors is not None:
        #     dr_scrap[0] += np.sum(dyadic_effects * dyadic_predictors[i, j, :])
        #     dr_scrap[1] += np.sum(dyadic_effects * dyadic_predictors[j, i, :])
        
        B_i_j = B[groups[i]-1, groups[j]-1]
        
        # dyadic random effects i -> j.
        dr_ij = dr_scrap[0] + np.sum(B_i_j)
        
        # BERNOULLI MODEL; either you are friends or not.
        # SR (sender and receivier random effects) + receivier + sender random effects
        p[i, j] = inv_logit(sr[i, 0] + sr[j, 1] + dr_ij)
        
        # symetric social ties
        y_true[i, j] = y_true[j, i] = bernoulli.rvs(p[i, j])
        p[j, i] = p[i, j]

# draw the underlying social network
nx.draw(nx.from_numpy_array(y_true), with_labels=True)
        
num_papers = 100
all_papers = []

for paper in num_papers:
    
    instigator = np.random.choice(range(N_id)) # should be dependent on productivity

    collaborator_set = ()

    nb_collaborators = np.random.poisson(4) # should be dependent on group norms

    