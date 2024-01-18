
import xgi
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as math
from collections import Counter




def leave_group(bk, tau, beta, N):
    """
    Prob node $i$ leave its own group of size $k$ at time $t$:
    
    $$p_{k}(\tau) = \frac{b_k}{\tau^{\beta}/N}$$
    
    where $b_k$ is a constant, $\beta$ is a real valued exponent that
    modulates the impact of residence time and $N$ is the number of agents.
    Each agent only belongs to one hyperedge $k$ at a time.
    """
    return bk / (tau**beta / N)

def change_group(H, w):
    """
    Prob to join $\omega$ proportional to fraction $X_{i,\omega}$ of agents in $\omega$ at time
    $t$ have already interacted with $i$ in the past:
    
    $$X_{i,\omega} = \frac{1 + [ \omega \cap_{t'=1}^{t} \sigma_{i}^{t'} ] }{ 1 + |\omega|}$$

    """
    pass

# The algorithm (here agent-first view). At each timestep:
# 1. select an agent randomly from N
# 2. if agent $i \in V$, currently a member of $\sigma_{i}^{t}$, two potential actions:
#   2.1. stay in the same group, depending on time spent there and group size
#   2.2. leave it for a different one:
#     2.2.1. choose based on acquaintance made until that time




def gen_from_zeroinflated():
    """
    hyperedge of k=0 are really groups of 1. 
    """
    N = 1000
    x_1 = np.random.normal(5,1,size=N)
    x_2 = np.random.normal(5,1,size=N)
    x = pd.DataFrame([x_1,x_2]).T
    poisson_part = np.zeros(N)
    zi_part = np.zeros(N)
    for i, item in enumerate(x_1):
        poisson_part[i] = np.random.poisson(math.exp(0.4*x_1[i]-0.1*x_2[i])) #needed to initialize the test object. Note the poisson parameter is of the form e^(Bx), ln(lambda) = Bx
    for i, item in enumerate(x_1):
        zi_part[i] = np.random.logistic(0.3*x_1[i]-0.2*x_2[i]) > 0 #needed to initialize the test object.

    return poisson_part * zi_part

def gen_histplot():
    dist_coauthors = gen_from_zeroinflated()
    dist_coauthors_count = Counter(dist_coauthors)
    dist_coauthors_count = dict(sorted(dist_coauthors_count.items(), key=lambda pair: pair[0]))
    x = [_+1 for _ in dist_coauthors_count.keys()]
    y = [_ for _ in dist_coauthors_count.values()]

    plt.bar(x, y)
    plt.xlabel("Number of coauthors")
    plt.ylabel("Frequency")
    plt.suptitle("Distribution of number of coauthors")

gen_histplot()


n = 1000
m = 500

min_edge_size = 1
max_edge_size = 40

# hyperedge dict
hyperedge_dict = {}
for paper_event in range(500):
    plt.hist([np.random.poisson(10) for i in range(1000)])
    nb_coauthors = random.choice(range(min_edge_size, max_edge_size + 1))
    

# hyperedge_dict = {
#     i: random.sample(range(n), random.choice(range(min_edge_size, max_edge_size + 1)))
#     for i in range(m)
# }

H = xgi.Hypergraph(hyperedge_dict)
print(f"The hypergraph has {H.num_nodes} nodes and {H.num_edges} edges")

plt.hist(H.nodes.degree.aslist())
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.suptitle("Degree distribution with n=1000, m=500,\nmin_edge_size=1, max_edge_size=40")

# H.nodes.filterby("degree", 3) 