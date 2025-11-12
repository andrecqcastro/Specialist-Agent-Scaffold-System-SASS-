import os
import datetime


# --- Model Configuration ---
DEFAULT_TASK_MODEL = "gpt-4o-mini"
DEFAULT_META_MODEL = "gpt-4o" 

# --- Evolution Configuration ---
DEFAULT_MAX_FAILURES_PER_CHILD = 5  # x=5 errors to find
DEFAULT_NUM_PARENTS_TO_SELECT_K = 1 # k parents selected per iteration
DEFAULT_ITERATIONS_T = 10           # T iterations of DGM

# BASE_LAMBDA_PRODUCT = DEFAULT_ITERATIONS_T * (some_base_lambda)
# We use this to calculate lambda based on user-provided iteration count
# e.g., 10 iterations * 30 lambda = 300.
# If user provides 5 iterations, lambda becomes 300 / 5 = 60 (more exploitation)
# If user provides 20 iterations, lambda becomes 300 / 20 = 15 (more exploration)
BASE_LAMBDA_PRODUCT = 300.0
DEFAULT_SIGMOID_ALPHA0 = 0.5        # α0 (alpha_0)

# --- Project & Dataset Configuration ---
DEFAULT_DATASET = "openai/gsm8k"
STARTING_AGENT_FILENAME = "math_agent_v0.py"


# This global dictionary will be updated by logging_utils.py
token_usage_stats = {"total_prompt_tokens": 0, "total_completion_tokens": 0, "total_cost_usd": 0.0}

# Cost per 1 million tokens (as of late 2024)
MODEL_COSTS = {
    "gpt-4o": {"prompt": 5.0 / 1_000_000, "completion": 15.0 / 1_000_000},
    "gpt-4o-mini": {"prompt": 0.15 / 1_000_000, "completion": 0.6 / 1_000_000},
    "gpt-4-turbo": {"prompt": 10.0 / 1_000_000, "completion": 30.0 / 1_000_000},
    # Add other models as needed
}

# This will be set by setup_logging() in main.py
LOG_FOLDER = ""