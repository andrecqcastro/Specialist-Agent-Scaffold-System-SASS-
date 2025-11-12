import os
import re
import sys
import importlib.util
import traceback
from src.logging_utils import log_event

def clean_generated_code(code_string: str) -> str:
    """Robustly extracts Python code from LLM-generated markdown."""
    if code_string is None:
        return ""
    
    # Pattern 1: ```python ... ```
    match = re.search(r"```python\s*([\s\S]+?)\s*```", code_string, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Pattern 2: ``` ... ``` (no language specified)
    match = re.search(r"```\s*([\s\S]+?)\s*```", code_string, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Pattern 3: No backticks, assume full string is code
    if "```" not in code_string:
        return code_string.strip()
        
    return code_string.strip() # Fallback

def save_agent_code(folder: str, filename: str, code: str):
    """Saves agent code to a file."""
    if not os.path.exists(folder):
        os.makedirs(folder)
    filepath = os.path.join(folder, filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)
    except IOError as e:
        print(f"Error saving agent code to {filepath}: {e}")

def extract_final_answer(text: str) -> str:
    """
    Extracts the final numeric answer from gsm8k-formatted text.
    Normalizes the answer by removing commas and trailing periods.
    """
    text = str(text)
    
    # Main pattern: #### 123
    matches = re.findall(r"####\s*([0-9,.]+)", text)
    if matches:
        number_str = matches[-1].replace(",", "").strip()
        # NORMALIZATION: Remove trailing period (e.g., "3." -> "3")
        # This will not affect valid decimals (e.g., "1.5")
        normalized_str = number_str.rstrip(".")
        return normalized_str

    # Fallback: If '####' is missing, get the last number in the string
    fallback_matches = re.findall(r"([0-9,.]+)", text)
    if fallback_matches:
        number_str = fallback_matches[-1].replace(",", "").strip()
        normalized_str = number_str.rstrip(".")
        return normalized_str
        
    return "" # No answer found

def load_agent_from_file(filepath: str, module_name: str):
    """
    Dynamically loads a Python module from a file path.
    Returns the loaded module or None on failure.
    """
    try:
        abs_filepath = os.path.abspath(filepath)
        spec = importlib.util.spec_from_file_location(module_name, abs_filepath)
        if spec is None:
            print(f"Error: Could not create module spec from {filepath}")
            return None
            
        agent_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = agent_module
        
        # Temporarily add module's directory to path for relative imports
        module_dir = os.path.dirname(abs_filepath)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
            
        spec.loader.exec_module(agent_module)
        
        # Clean up sys.path
        if sys.path[0] == module_dir:
            sys.path.pop(0)
            
        # Clean up sys.modules to allow reloading next time
        if module_name in sys.modules:
            del sys.modules[module_name] 
            
        return agent_module
        
    except Exception as e:
        print(f"--- FATAL ERROR LOADING AGENT: {filepath} ---")
        print(traceback.format_exc())
        log_event("fatal_errors.log", f"Failed to load agent {filepath}:\n{traceback.format_exc()}")
        return None

def validate_agent_model_usage(code_string: str, allowed_model: str) -> bool:
    """
    Checks if the agent code tries to use a different LLM model.
    """
    # This regex looks for model strings like "gpt-4", "claude-3", etc.
    model_pattern = re.compile(r"""
        model\s*=\s*['"](
            gpt-[\w.-]+ |
            claude-[\w.-]+ |
            gemini-[\w.-]+
        )['"]
    """, re.IGNORECASE | re.VERBOSE)
    
    found_models = model_pattern.findall(code_string)
    
    if not found_models:
        return True # No model definitions found, pass.

    for model in found_models:
        if model.strip() != allowed_model.strip():
            print(f"  [Validation] FAILED: Agent code specifies invalid model '{model}'. Allowed: '{allowed_model}'.")
            log_event("invalid_agents.log", 
                      f"Agent validation failed: Found model '{model}', expected '{allowed_model}'.")
            return False
            
    return True