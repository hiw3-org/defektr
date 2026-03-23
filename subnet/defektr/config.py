# Defektr subnet — permanent constants
# Per-challenge parameters (model constraints, scoring weights, dataset category, etc.)
# live in the challenge spec JSON published by the validator each round.

# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------

NETUID = 3

# How often set_weights() is called (standard Bittensor tempo, in blocks).
EPOCH_TEMPO = 100  # ≈ 20 min

# How long miners have to train and submit a model for a challenge (in blocks).
CHALLENGE_INTERVAL = 50_000  # ≈ 7 days

# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------

CACHE_DIR = "model_cache/"

# Maximum ONNX session load time before the model is rejected.
MAX_LOAD_TIME_S = 30.0

# ---------------------------------------------------------------------------
# Benchmark sampling
# ---------------------------------------------------------------------------

# Number of images drawn per evaluation round.
BENCHMARK_N = 100

# ---------------------------------------------------------------------------
# Anti-gaming
# ---------------------------------------------------------------------------

# Cosine similarity above this threshold → outputs treated as a copy.
COPY_DETECT_THRESHOLD = 0.95
