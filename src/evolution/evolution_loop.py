import os
import math
import numpy as np
from src.logging_utils import log_event
from src.agent_utils import load_agent_from_file, validate_agent_model_usage, save_agent_code
from src.agents.developer import call_developer_agent
from src.evolution.evaluation import evaluate_agent_on_dataset, find_failures_on_train

def select_parents(archive: list, k: int, sigmoid_lambda: float, sigmoid_alpha0: float) -> list:
    """
    Selects `k` parents from the archive using the DGM formula.
    """
    # Filter for agents that are not "perfect" (score < 1.0)
    eligible_set = [agent for agent in archive if agent['score'] < 1.0]

    if not eligible_set:
        if not archive:
            return [] # Archive is empty
        print("  [Selection] All agents are perfect. Selecting randomly from archive.")
        eligible_set = archive

    weights, agents_in_set = [], []
    for agent in eligible_set:
        alpha_i = agent['score']
        n_i = agent['children_count']
        
        # s_i: Score-based fitness (Sigmoid function)
        s_i = 1 / (1 + math.exp(-sigmoid_lambda * (alpha_i - sigmoid_alpha0)))
        
        # h_i: Novelty/History-based fitness (Discourage child-heavy parents)
        h_i = 1 / (1 + n_i)
        
        # Final weight
        w_i = s_i * h_i
        
        weights.append(w_i)
        agents_in_set.append(agent)

    total_weight = np.sum(weights)
    if total_weight == 0:
        # All agents have zero weight, select uniformly
        probabilities = np.ones(len(agents_in_set)) / len(agents_in_set)
    else:
        probabilities = np.array(weights) / total_weight

    selected_parents = np.random.choice(
        a=agents_in_set, 
        size=k, 
        replace=True, # DGM uses sampling with replacement
        p=probabilities
    )
    return list(selected_parents)


def run_dgm_loop(dataset, 
                 start_agent_path: str,
                 agent_folder: str,
                 num_iterations: int,
                 task_model: str,
                 meta_model: str,
                 dgm_params: dict) -> dict | None:
    """
    Executes the full DGM (Deep Genetic Manager) evolution loop.
    """
    print("=" * 52)
    print("STARTING DGM (Deep Genetic Manager) EVOLUTION LOOP")
    print("=" * 52)

    train_data = dataset['train']
    validation_data = dataset['validation']
    
    agent_version_counter = 0
    archive = [] # Archive A

    # 1. Initialize A <- {g0}
    print(f"--- Initializing Base Agent v0 ---")
    module_name_v0 = f"math_agent_v{agent_version_counter}"
    agent_module_v0 = load_agent_from_file(start_agent_path, module_name_v0)
    
    if agent_module_v0 is None or not hasattr(agent_module_v0, 'run_agent'):
        print("FATAL ERROR: Base agent v0 could not be loaded. Aborting.")
        return None

    initial_score = evaluate_agent_on_dataset(agent_module_v0, validation_data)
    
    g0 = {
        'id': f"v{agent_version_counter}",
        'path': start_agent_path,
        'score': initial_score,
        'children_count': 0,
        'parent_id': None
    }
    archive.append(g0)
    log_event("archive.log", g0)
    print(f"Base agent v0 initialized. Validation Score: {initial_score:.4f}")

    # 2. Start DGM loop
    for t in range(num_iterations):
        print(f"\n--- DGM Iteration {t+1} / {num_iterations} ---")
        
        # P <- SelectParents(A)
        parents = select_parents(
            archive, 
            dgm_params['num_parents_to_select_k'],
            dgm_params['sigmoid_lambda'],
            dgm_params['sigmoid_alpha0']
        )
        
        if not parents:
            print("No parents could be selected. Ending evolution.")
            break
            
        parent_ids = [p['id'] for p in parents]
        print(f"  [Selection] Selected parents: {parent_ids}")
        log_event("evolution.log", f"Iteration {t+1}: Selected parents {parent_ids}")

        # foreach p ∈ P
        for parent_agent in parents:
            print(f"\n  --- Processing Parent: {parent_agent['id']} (Score: {parent_agent['score']:.4f}) ---")
            
            parent_module = load_agent_from_file(parent_agent['path'], parent_agent['id'])
            if parent_module is None:
                print(f"    ERROR: Could not load parent module {parent_agent['id']}. Skipping.")
                continue
            
            failures, successes = find_failures_on_train(
                parent_module, 
                train_data, 
                dgm_params['max_failures_per_child']
            )

            if not failures:
                print(f"    Parent {parent_agent['id']} had no failures on training set. Will not generate child.")
                continue
            
            success_examples_to_send = successes[:dgm_params['num_successes_to_send']]
            print(f"    Sending {len(failures)} failures and {len(success_examples_to_send)} successes to Developer.")
            
            try:
                with open(parent_agent['path'], 'r', encoding='utf-8') as f:
                    parent_code = f.read()
            except FileNotFoundError:
                print(f"    ERROR: Could not read parent code {parent_agent['id']}. Skipping.")
                continue

            # c <- p.modify(p)
            new_code = call_developer_agent(
                previous_code=parent_code,
                success_examples=success_examples_to_send,
                failed_examples=failures,
                current_version_id=parent_agent['id'],
                task_model=task_model,
                meta_model=meta_model
            )

            if not new_code:
                print("    Developer Agent failed to produce code. Child discarded.")
                log_event("invalid_agents.log", f"Child of {parent_agent['id']} failed generation (no code).")
                continue

            agent_version_counter += 1
            child_id = f"v{agent_version_counter}"
            
            # --- VALIDATION 1: Duplicate Check ---
            if new_code.strip() == parent_code.strip():
                print(f"    Child {child_id} is identical to parent. Discarding.")
                log_event("invalid_agents.log", f"Child {child_id} was identical to parent {parent_agent['id']}.")
                continue
                
            # --- VALIDATION 2: Model Regex Guard ---
            if not validate_agent_model_usage(new_code, task_model):
                print(f"    Child {child_id} failed model validation. Discarding.")
                log_event("invalid_agents.log", f"Child {child_id} (parent {parent_agent['id']}) failed model validation.")
                continue

            child_filename = f"math_agent_{child_id}.py"
            child_path = os.path.join(agent_folder, child_filename)
            save_agent_code(agent_folder, child_filename, new_code)
            
            # Log the generation data
            log_event(f"child_{child_id}_generation_data.json", {
                "child_id": child_id,
                "parent_id": parent_agent['id'],
                "failures_sent": failures,
                "successes_sent": success_examples_to_send
            })
            
            child_module = load_agent_from_file(child_path, child_id)
            
            # if c.is_valid()
            if child_module is not None and hasattr(child_module, 'run_agent'):
                print(f"    Child {child_id} is FUNCTIONAL.")
                parent_agent['children_count'] += 1
                
                # s <- evaluate(c, B)
                child_score = evaluate_agent_on_dataset(child_module, validation_data)
                
                # A <- A ∪ {(c, s)}
                child_agent_data = {
                    'id': child_id,
                    'path': child_path,
                    'score': child_score,
                    'children_count': 0,
                    'parent_id': parent_agent['id']
                }
                archive.append(child_agent_data)
                log_event("archive.log", child_agent_data) # Log child to main archive log
            else:
                print(f"    Child {child_id} is NOT functional (failed to load). Discarding.")
                log_event("invalid_agents.log", f"Child {child_id} (parent {parent_agent['id']}) was NOT functional.")
                try:
                    os.remove(child_path) # Clean up invalid agent
                except OSError:
                    pass

    print("\n" + "=" * 52)
    print("DGM EVOLUTION LOOP COMPLETE.")
    print("=" * 52)

    if not archive:
        print("No agents survived in the archive.")
        return None

    sorted_archive = sorted(archive, key=lambda x: x['score'], reverse=True)
    log_event("archive_final.json", sorted_archive)
    
    print("\nFinal Agent Archive (sorted by validation score):")
    for i, agent in enumerate(sorted_archive):
        print(f"  {i+1}. ID: {agent['id']} (Parent: {agent['parent_id']}) | Score: {agent['score']:.4f} | Children: {agent['children_count']}")
    
    return sorted_archive[0] # Return the best agent