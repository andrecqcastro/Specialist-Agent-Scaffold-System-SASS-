python
import numpy as np
import traceback
from src.agent_utils import extract_final_answer

def evaluate_agent_on_dataset(agent_module, dataset):
    """
    Evaluates an agent on a dataset (validation or test) and returns its accuracy score.
    Returns 0.0 if the agent is not functional.
    """
    if agent_module is None or not hasattr(agent_module, 'run_agent'):
        print(f"  [Evaluation] Agent module is not functional. Score: 0.0")
        return 0.0

    agent_runner = agent_module.run_agent
    success_count = 0
    total_count = len(dataset)
    if total_count == 0:
        return 0.0

    print(f"  [Evaluation] Running evaluation on {total_count} samples...")
    for sample in dataset:
        try:
            question = sample['question']
            correct_answer_raw = sample['answer']
            correct_answer_final = extract_final_answer(correct_answer_raw)

            agent_answer_raw = agent_runner(question)
            agent_answer_final = extract_final_answer(agent_answer_raw)

            if agent_answer_final == correct_answer_final and correct_answer_final != "":
                success_count += 1
        except Exception as e:
            # This should not happen if run_agent() catches its own errors
            print(f"  [Evaluation] Critical runtime error during evaluation: {e}")
            pass
    
    accuracy = success_count / total_count
    print(f"  [Evaluation] Complete. Score: {accuracy:.4f} ({success_count}/{total_count})")
    return accuracy

def find_failures_on_train(agent_module, train_dataset, max_failures: int):
    """
    Runs the agent on the training set until `max_failures` errors are found.
    """
    failures = []
    successes = []
    
    if agent_module is None or not hasattr(agent_module, 'run_agent'):
        print(f"  [Train] Agent module is not functional. Cannot find failures.")
        return [], []

    agent_runner = agent_module.run_agent

    print(f"  [Train] Searching for up to {max_failures} failures in training set...")
    # Shuffle dataset for random sampling of failures
    for sample in train_dataset.shuffle(seed=np.random.randint(10000)):
        if len(failures) >= max_failures:
            print(f"  [Train] Failure limit of {max_failures} reached.")
            break

        question = sample['question']
        correct_answer_raw = sample['answer']
        correct_answer_final = extract_final_answer(correct_answer_raw)

        try:
            agent_answer_raw = agent_runner(question)
            agent_answer_final = extract_final_answer(agent_answer_raw)

            if agent_answer_final == correct_answer_final and correct_answer_final != "":
                successes.append({
                    "question": question,
                    "correct_answer": correct_answer_raw,
                    "agent_output": agent_answer_raw
                })
            else:
                failures.append({
                    "question": question,
                    "correct_answer": correct_answer_raw,
                    "agent_output": f"Wrong Answer: {agent_answer_raw}"
                })
        except Exception as e:
            # This should be caught by run_agent, but as a fallback
            failures.append({
                "question": question,
                "correct_answer": correct_answer_raw,
                "agent_output": f"Runtime Error: {traceback.format_exc()}"
            })
    
    print(f"  [Train] Search complete. Found {len(failures)} failures, {len(successes)} successes.")
    return failures, successes