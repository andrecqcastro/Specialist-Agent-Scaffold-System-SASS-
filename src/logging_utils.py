import os
import sys
import json
import datetime
from openai import OpenAI
from dotenv import load_dotenv
from src import config


# --- OpenAI Client ---
try:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(api_key=api_key)
except Exception as e:
    print(f"Error initializing OpenAI client: {e}", file=sys.stderr)
    sys.exit(1)


def setup_logging(log_dir: str):
    """Creates the unique log directory for this run."""
    config.LOG_FOLDER = log_dir
    if not os.path.exists(config.LOG_FOLDER):
        os.makedirs(config.LOG_FOLDER)
    print(f"Logs will be saved in: {config.LOG_FOLDER}")

def log_event(filename, data):
    """Logs a dictionary or string to a file in the run's log directory."""
    if not config.LOG_FOLDER:
        print(f"Warning: LOG_FOLDER not set. Cannot log: {filename}", file=sys.stderr)
        return
        
    log_path = os.path.join(config.LOG_FOLDER, filename)
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            if isinstance(data, dict) or isinstance(data, list):
                f.write(json.dumps(data, indent=2) + "\n")
            else:
                f.write(str(data) + "\n")
    except Exception as e:
        print(f"Warning: Failed to write log to {filename}. Error: {e}", file=sys.stderr)


def update_token_stats(response, model_name):
    """Updates the global token and cost counters."""
    if not response or not response.usage:
        return
    
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    
    config.token_usage_stats["total_prompt_tokens"] += prompt_tokens
    config.token_usage_stats["total_completion_tokens"] += completion_tokens
    
    costs = config.MODEL_COSTS.get(model_name, {"prompt": 0, "completion": 0})
    cost = (prompt_tokens * costs["prompt"]) + (completion_tokens * costs["completion"])
    config.token_usage_stats["total_cost_usd"] += cost

    log_event("token_usage.log", 
              f"Model: {model_name}, Prompt: {prompt_tokens}, Completion: {completion_tokens}, Cost: ${cost:.6f}")

def get_final_token_summary() -> str:
    """Returns a formatted string of the final token usage and cost."""
    summary = (
        "\n--- Meta-Model API Usage (Evolution) ---\n"
        f"Total Prompt Tokens:   {config.token_usage_stats['total_prompt_tokens']}\n"
        f"Total Completion Tokens: {config.token_usage_stats['total_completion_tokens']}\n"
        f"Estimated Total Cost:  ${config.token_usage_stats['total_cost_usd']:.6f}"
    )
    return summary