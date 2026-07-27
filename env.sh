# Source this before doing ANYTHING in this project.

source "/home/ubuntu/diffgemma_fa/.venv/bin/activate"

# The claude CLI usually lives here and a non-interactive shell won't find it.
export PATH="$HOME/.local/bin:$HOME/bin:$PATH"

# GPU isolation: only this device is visible to JAX.
export CUDA_VISIBLE_DEVICES=0

# DEDICATED BOX: preallocation ON. Faster, no fragmentation, and the only
# configuration in which SPEC section 7.3's overhead numbers are publishable.
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=.90
export DGFA_DEDICATED=1

# Private compilation cache.
export JAX_COMPILATION_CACHE_DIR="/home/ubuntu/diffgemma_fa/.jax_cache"

export HF_HOME="/home/ubuntu/diffgemma_fa/.hf"
export PROJECT_DIR="/home/ubuntu/diffgemma_fa"

# Highest phase this hardware can complete (see SPEC section 8). 3 means the
# model does not fit here: stop after the JAX kernel microbenchmarks.
export DGFA_MAX_PHASE=6
