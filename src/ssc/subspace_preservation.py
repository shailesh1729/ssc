import numpy as np
import pandas as pd
import time

from jax import random
import jax.numpy as jnp

import cr.sparse as crs
import cr.sparse.data as crdata
import cr.sparse.la.subspaces
import cr.sparse.cluster as cluster
import cr.sparse.cluster.ssc as ssc
import cr.sparse.cluster.spectral as spectral

from typing import NamedTuple


class Row(NamedTuple):
    r : int
    "Experiment number"
    t : int
    "Trial number"
    M : int
    "Ambient space dimension"
    D : int
    "Dimension of individual subspaces"
    K : int
    "Number of subspaces"
    S : int
    "Number of points per cluster/subspace"
    num_points : int
    "Total number of points"
    num_missed : int
    "Number of points which were misclustered"
    error: float
    "Clustering error"
    acc: float
    "Clustering accuracy"
    spr_error: float
    "Subspace preservation error"
    spr_perc : float
    "Percentage of subspace preserving representations"
    runtime: float
    """Time taken for whole computation"""
    rep_time: float
    """Time taken for building representations"""
    cluster_time: float
    """Time taken for spectral clustering"""

    def __str__(self):
        s = []
        for x in [
            f"Exp {self.r} trial : {self.t}",
            f"M {self.M} D : {self.D}, K : {self.K}, S : {self.S}",
            f"num_points {self.num_points} num_missed : {self.num_missed}",
            f"error {self.error:.2f} acc : {self.acc:.2f}",
            f"spr_error {self.spr_error:.2f} spr_perc : {self.spr_perc:.1f}%%",
            f"run_time : {self.runtime:.2f} sec",
            f"rep_time {self.rep_time:.2f} sec cluster_time : {self.cluster_time:.2f} sec",
            ]:
            s.append(x.rstrip())
        return ' '.join(s)

def bench_subspace_preservation(key, solver, experiment_name):
    """An extensive benchmark for testing subspace preservation of an SSC algorithm
    """
    destination = f'{experiment_name}.csv'
    df = pd.DataFrame(columns=Row._fields)
    num_points_per_cluster_list = jnp.array([30, 55, 98, 177, 320, 577, 1041, 1880, 3396, 6132, 11075, 20000])
    #num_points_per_cluster_list = jnp.array([30, 55, 98])
    # signal density levels
    rhos = jnp.round(5**jnp.arange(1, 5, 0.4))
    # Number of experiments
    R = len(num_points_per_cluster_list)
    # Number of trials per configuration
    num_trials = 50
    print(f"Number of experiments: {R}, trials: {num_trials}")
    keys = random.split(key, R)
    # Ambient space dimension
    M = 9
    # Number of subspaces
    K = 5
    # common dimension for each subspace
    D = 6

    # # Some limits
    # R = 2
    # num_trials = 2

    for r in range(R):
        rho = rhos[r]
        rho_id = r 
        rkey = keys[r]
        tkeys = random.split(rkey, num_trials)
        S = num_points_per_cluster_list[r]
        for tt in range(num_trials):
            tkey = tkeys[tt]
            skeys = random.split(tkey, 5)
            bases = crdata.random_subspaces_jit(skeys[0], M, D, K)
            cluster_sizes = jnp.ones(K, dtype=int) * S
            X = crdata.uniform_points_on_subspaces_jit(skeys[1], bases, int(S))
            true_labels = jnp.repeat(jnp.arange(K), S)
            start_time = time.perf_counter()
            # Build representation of each point in terms of other points 
            Z, I, R = solver(X, D)
            # Combine values and indices to form full representation
            Z_sp = ssc.sparse_to_bcoo(Z, I)
            # Build the affinity matrix
            affinity = ssc.rep_to_affinity(Z_sp)
            rep_time = time.perf_counter()
            # Perform the spectral clustering on the affinity matrix
            res = spectral.normalized_symmetric_sparse_fast_k_jit(keys[2], affinity, K)
            end_time = time.perf_counter()
            runtime=end_time-start_time # in seconds
            # Predicted cluster labels
            pred_labels = res.assignment
            clust_err = cluster.clustering_error_k(true_labels, pred_labels, K)
            spr_stats = ssc.sparse_subspace_preservation_stats_jit(Z, I, true_labels)
            # summarized information
            row = Row(r=r, t=tt, M=M, D=D, K=K, S=S, 
                num_points=len(true_labels),
                num_missed=clust_err.num_missed,
                error=clust_err.error,
                acc=1-clust_err.error,
                spr_error=spr_stats.spr_error,
                spr_perc=spr_stats.spr_perc,
                runtime=runtime,
                rep_time=rep_time-start_time,
                cluster_time=end_time-rep_time)
            print(row)
            df.loc[len(df)] = row
        # save results after each experiment
        df.to_csv(destination, index=False)
    # save results after all experiments
    df.to_csv(destination, index=False)
            





