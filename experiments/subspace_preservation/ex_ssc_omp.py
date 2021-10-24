# Configure JAX for 64-bit computing
from jax.config import config
config.update("jax_enable_x64", True)

from jax import random

from ssc.subspace_preservation import bench_subspace_preservation
import cr.sparse.cluster.ssc as ssc

key = random.PRNGKey(0)
solver = lambda X, K: ssc.batch_build_representation_omp_jit(X, K, 800)
bench_subspace_preservation(key, solver, 
    'ssc-omp-subspace-preservation-random-subspaces')
