"""
measure: publication data
goal: infer blocks in social graph from publication data
"""
import numpy as np
import scipy.stats as stats
import scipy.special as sp
import networkx as nx
import matplotlib.pyplot as plt

def inv_logit(x):
    return np.exp(x) / (1 + np.exp(x))

def plot_G(y):
    G = nx.from_numpy_array(y)

    color_map = ['turquoise' if groups[i] == 1 else 'red' if groups[i] == 2 else 'gold' for i in range(30)]
    degrees = dict(G.degree())
    scaled_degrees = {node: degree*20 for node, degree in degrees.items()}
    node_sizes = [scaled_degrees[node] for node in G.nodes()]
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos=nx.kamada_kawai_layout(G),
            node_color=color_map,
            edgecolors='black', 
            node_size=node_sizes,
            arrowstyle='->', 
            arrowsize=12)
    plt.show()

N_id=30
N_groups=3
gprobs=[3, 3, 3]
observed_groups=True
in_block, out_block =0.9, 0.01
indeg_mu, outdeg_mu= 0, 0.
indeg_sigma, outdeg_sigma = 1, 1
alpha=[0, -2, -4]
beta=[2, 1, 4]
theta=0.5
prob_obs_g=0.8
N_pubs=100
pg_prior=[6, 6, 6]
    
# Sample people into groups
groups = np.random.choice(np.arange(1, N_groups + 1), size=N_id, replace=True, p=gprobs/np.sum(gprobs))

# Define interaction matrix across groups
B = np.zeros((N_groups, N_groups)) + out_block
np.fill_diagonal(B, in_block)

# Varying effects on individuals
v = np.random.multivariate_normal([outdeg_mu, indeg_mu], np.diag([outdeg_sigma**2, indeg_sigma**2]), N_id)

# Simulate ties; simple SBM stuff. 
# TODO:
# in-deg and out-deg could depend on individual features such as academic age.
y_true = np.zeros((N_id, N_id))
for i in range(N_id):
    for j in range(N_id):
        if i != j:
            pij = inv_logit(sp.logit(B[groups[i] - 1, groups[j] - 1]) + v[i, 0] + v[j, 1])
            y_true[i, j] = stats.bernoulli.rvs(pij)

# plot_G(y_true)

# Now simulate publication process
lambda_ = np.log([0.1,1.5]) # rates of recruitement for y=0; y=1
all_collabs = {}

# individual features
P=np.random.exponential(1, N_id) # relative prestige in community
bPR = 0.3 # effect of prestige on recruitment - prestigious folks can more easily recruit
bRP = -1 # effect of prestige on recruitment - prestige get more easily recruited

# dyadic features?
# something about reciprocity; i invite you  and you invite me...
y_true = y_true.astype(int)

for k in range(N_pubs):
    
    # instigator; prob of choosing should be dependent on productivity
    i = np.random.choice(range(N_id))
    collab_set = set()
    
    # should be dependent on group norms
    # number_collaborators = np.random.poisson(4) 
    
    # now we look at the recruiment process; we go through all individuals
    # looking at the probability of being recruited by i.
    for j in range(N_id):
        if i != j:
            # Can we do similar than STRAND eq.6
            #    φ[i, j] = B[b(i),b( j)] + λ[i] + π[ j] + δ[i, j] + . . .
            # B:    is the SBM intercept matrix 
            # λ[i]: vector of individual-specific sender/nominator effects governing out-degree
            # π[j]: vector of individual-specific receiver/target effects governing in-degree
            # δ[i, j]: matrix of dyadic effects governing dyadic reciprocity
                
            pij = inv_logit(lambda_[y_true[i, j]] + bPR*P[i] )
            prob_recruit_ij = stats.bernoulli.rvs(pij)

            if np.random.uniform() < prob_recruit_ij:
                # i successfully recruited j
                collab_set.add(j)

            # TODO: pji
    
    if all_collabs.get(i) is None:
        all_collabs[i] = [collab_set]
    else:
        all_collabs[i] += [collab_set]


def flatten(x):
    return [item for sublist in x for item in sublist]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.hist([len(_) for _ in flatten([v for k,v in all_collabs.items()])])
ax1.set_xlabel("collaboration group size")

ax2.hist([len(v) for k,v in all_collabs.items()])
ax2.set_xlabel("number of papers published")
