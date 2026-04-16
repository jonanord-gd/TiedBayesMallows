from TiedBayesMallows.model.distance import total_distance_fast
from TiedBayesMallows.model.priors import log_Z_star_from_sizes
from TiedBayesMallows.model.blocks import blocks_to_block_index
from math import comb, factorial
import random, time
from typing import List, Tuple, Optional, Callable
import numpy as np


def sample_categorical(w: List[float], rng: random.Random) -> int:
    tot_w = sum(w)
    probs = [x / tot_w for x in w]
    u = rng.random()
    s = 0.0
    for k, p in enumerate(probs):
        s += p
        if u <= s:
            return k
    return len(probs) - 1

def generate_candidates(blocks, block_idx, elem_idx):
    """
    Generate 2K candidates by removing element l from blocks[block_idx][elem_idx].
    
    Returns a list of 2K candidate block configurations.
    """
    K = len(blocks)
    l = blocks[block_idx][elem_idx]
    
    # Remove element l from its block
    base = []
    for i, block in enumerate(blocks):
        if i == block_idx:
            new_block = block[:elem_idx] + block[elem_idx+1:]
            if new_block:  # only keep non-empty blocks
                base.append(new_block)
        else:
            base.append(list(block))
    
    candidates = []
    
    # First K+1 candidates: insert [l] as a new block between/around existing blocks
    for pos in range(len(base) + 1):
        candidate = base[:pos] + [[l]] + base[pos:]
        candidates.append(candidate)
    
    # Last K-1 candidates: add l to one of the other blocks
    for i, block in enumerate(base):
        if i == (block_idx if blocks[block_idx] != [l] else -1):
            continue  # skip the block it came from
        candidate = [list(b) for b in base]
        candidate[i] = candidate[i] + [l]
        candidates.append(candidate)
    
    return candidates

def fast_gibbs(rankings: List[List[int]],
    blocks: List[List[List[int]]],
    block_index: List[int],
    theta: float,
    gamma: float,
    delta: float,
    z: List[int],
    rng: random.Random = None,
    tie_penalty: float = 0.5):

    if rng is None:
        rng = random.Random()

    C = len(blocks) # number of clusters
    N = len(rankings) # number of assessors
    n = len(rankings[0]) # number of items
    
    U_all = np.zeros((N, comb(n, 2)))
    for i in range(N):
        U_all[i] = np.array([1 if rankings[i][k] > rankings[i][l] else 0 for k in range(n) for l in range(k+1, n)]) # number of times i is perferred to j
    
    H = np.zeros((C, comb(n, 2)))
    # NOTE: K varies per cluster; arrays below are allocated per-cluster inside the loop
    cluster_distances = np.zeros(C)
    
    # Initate likelihoods (per-cluster, allocated below)
    likelihoods_create_permutation = [None] * C
    likelihoods_add_permutation = [None] * C
    likelihoods_all = [None] * C

    for c in range(C):
        cluster_rankings = [rankings[i] for i in range(N) if z[i] == c]
        blocks_c = blocks[c]
        K = len(blocks_c) # number of blocks in cluster c
        
        # Update the consensus ranking for cluster c 
        H[c] = np.sum([U_all[i] for i in range(N) if z[i] == c], axis=0) # total number of preferences of i over j in cluster c
        Z_c = log_Z_star_from_sizes([len(b) for b in blocks_c], theta) # log Z* of cluster c from blocks
        
        py_num = sum(np.log(gamma + i*delta) for i in range(1, K)) # log numerator of PY table term
        py_dom = sum(np.log(gamma + i) for i in range(1, n)) # log denominator
        py_blocks = sum(sum(np.log(1 - delta + j) for j in range(1, s)) for s in [len(block) for block in blocks_c]) # log probability of block sizes
        py_prod = py_num + py_blocks - py_dom # log probability
        
        prop_order = -np.log(factorial(K))
        
        S = total_distance_fast(cluster_rankings, blocks_c, tie_penalty=tie_penalty)
        cluster_distances[c] = S
        
        lw_base = prop_order + py_prod + Z_c + (-theta * S)
        likelihoods_create_permutation[c] = np.full(K + 1, lw_base)
        likelihoods_add_permutation[c] = np.full(K - 1, lw_base)
        likelihoods_all[c] = np.zeros(2 * K)
        

        
    
    for iteration in range(1000): # number of iterations
        z = compute_cluster_assignments(rankings, blocks, theta, gamma, delta, tie_penalty) # Not a real function yet
        
        for c in range(C):
            cluster_rankings = [rankings[i] for i in range(N) if z[i] == c]
            blocks_c = blocks[c]
            K = len(blocks_c) # number of blocks in cluster c
            cluster_distance_base = cluster_distances[c]
            
            block_sizes = [len(block) for block in blocks_c]
            block_index_arr = np.array(block_index)  # convert to numpy for np.where
            if cluster_rankings:
                
                N_c = len(cluster_rankings) # number of assessors in cluster c
                # Update the consensus ranking for cluster c 
                H[c] = np.sum([U_all[i] for i in range(N) if z[i] == c], axis=0) # total number of preferences of i over j in cluster c

                H_c_minus = H[c]
                H_c_plus = N_c - H[c]
                H_c_zero = np.full_like(H[c], tie_penalty * N_c)
                H_c = np.array([H_c_plus, H_c_zero, H_c_minus])
                # Update blocks 
                l = rng.randint(0, n-1) # randomly item to update
                block_l = block_index[l] # block index of item l
                block = blocks_c[block_l] # block of cluster c that contains item l
                s_orig = len(block)
                elem_idx = block.index(l) # index of item l in its block
                
                candidates = generate_candidates(blocks_c, block_l, elem_idx) # generate 2K candidates by removing item l from its block and either creating a new block or adding it to another block
                
                if s_orig == 1: # Singelton block case
                    py_ratio = np.array([(s - delta)/(gamma + (K-1)*delta) for s in block_sizes])  # ratio of adding item l to each block
                    Z_ratio = np.array([np.exp(-theta/2*(-s))*(1/(s+1))*(1-np.exp(-theta*(s+1)))/(1-np.exp(-theta)) for s in block_sizes]) # ratio of the partition function when adding item l to each block
                    order_ratio = np.array([K for _ in range(K)]) # ratio of the number of permutations when adding item l to each block
                
                    likelihoods_add_permutation[c] = likelihoods_add_permutation[c] * py_ratio * Z_ratio * order_ratio # update likelihoods of adding item l to each block
                    # create likelihood stays the same since the permutation structure remains unchanged
                else:
                    py_dom = s_orig - delta - 1
                    py_num_create = gamma + K*delta 
                    py_num_add = np.array([s - delta for s in block_sizes])
                    
                    Z_dom = np.exp(-theta/2*(s_orig+1))*(s_orig+1)*(1-np.exp(-theta*s_orig))
                    Z_num_create = np.exp(-theta/2)*(1-np.exp(-theta)) 
                    Z_num_add = np.array([np.exp(-theta/2*(s))*(s)*(1-np.exp(-theta*(s+1)))/(1-np.exp(-theta*s)) for s in block_sizes])
                    
                    order_ratio = 1/(K+1)
                    
                    likelihoods_create_permutation[c] = likelihoods_create_permutation[c] * (py_num_create/py_dom) * (Z_num_create/Z_dom) * order_ratio # update likelihood of creating a new block with item l
                    likelihoods_add_permutation[c] = likelihoods_add_permutation[c] * (py_num_add/py_dom) * (Z_num_add/Z_dom) * order_ratio # update likelihood of adding item l to each block
            
                # Add distance contribution
                block_l = block_index[l]
                
                block_l_indexs = np.where(block_index_arr == block_l)[0]
                H_c_blocks = H_c[:, block_l_indexs]
                
                cumulative =  H_c_blocks[2, :].sum() - H_c_blocks[1, :].sum() # cost of moving item l to a new block (creating a new block with item l)
                
                move_cost_create = np.repeat(cluster_distance_base,K+1) # first column: add cost, second column: create cost
                move_cost_add = np.repeat(cluster_distance_base,K-1)
                
                move_cost_create[block_l + 1] = cumulative 
                # walk right: 
                for k in range(block_l+1,K):
                    block_k_indexs = np.where(block_index_arr == k)[0]
                    H_c_blocks_k = H_c[:, block_k_indexs]

                    cumulative += H_c_blocks_k[2, :].sum() - H_c_blocks_k[0, :].sum()
                    move_cost_create[k+1] += cumulative # cost of creating a block below block k 
                    
                    add_delta = H_c_blocks_k[1, :].sum() - H_c_blocks_k[2, :].sum()
                    move_cost_add[k] += cumulative + add_delta # cost of moving item l to block k 
                    
                # walk left
                cumulative = H_c_blocks[0, :].sum() - H_c_blocks[1, :].sum()
                move_cost_create[block_l] = cumulative # cost of creating a block above block l
                for k in range(block_l-1, -1, -1):
                    block_k_indexs = np.where(block_index_arr == k)[0]
                    H_c_blocks_k = H_c[:, block_k_indexs]
                
                    cumulative += H_c_blocks_k[0, :].sum() - H_c_blocks_k[2, :].sum()
                    move_cost_create[k] = cumulative # cost of creating a block above block k
                    
                    add_delta = H_c_blocks_k[1, :].sum() - H_c_blocks_k[0, :].sum()
                    move_cost_add[k-1] += cumulative + add_delta # cost of moving item l to block k
                
                likelihoods_all[c][0:K+1] = likelihoods_create_permutation[c] * np.exp(-theta * move_cost_create)
                likelihoods_all[c][K+1:] = likelihoods_add_permutation[c] * np.exp(-theta * move_cost_add)
                
                # still need to add all components of the likelihood 
                # Also need to sample gibbs relative to likelihood
                
                cand_ind = sample_categorical(likelihoods_all[c], rng)
                if cand_ind < K+1: # create new block
                    new_blocks = candidates[cand_ind]
                    blocks[c] = new_blocks
                    block_index = blocks_to_block_index(blocks[c], n) # update block index
                else: # add to existing block
                    new_blocks = candidates[cand_ind]
                    blocks[c] = new_blocks
                    block_index = blocks_to_block_index(blocks[c], n) # update block index
            else:
                continue # No assessors in this cluster
                
            
    
    return
