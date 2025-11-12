from src.logging_utils import client, update_token_stats, log_event
from src.agent_utils import clean_generated_code

def call_developer_agent(previous_code: str,
                         success_examples: list,
                         failed_examples: list,
                         current_version_id: str,
                         task_model: str,
                         meta_model: str) -> str | None:
    """
    Uses the META_MODEL to analyze and improve an existing agent's code.
    """

    failures_report = "\n".join([
        f"- Q: {f['question']}\n  A (Correct): {f['correct_answer']}\n  A (Agent): {f['agent_output']}\n"
        for f in failed_examples
    ])
    
    if not success_examples:
        success_report = "No successful examples were provided for reference."
    else:
        success_report = "\n".join([
            f"- Q: {s['question']}\n  A (Correct): {s['correct_answer']}\n  A (Agent): {s['agent_output']}\n"
            for s in success_examples
        ])

    system_prompt = f"""
You are an elite Python programmer specializing in LangGraph agents.
Your task is to analyze, debug, and **SUBSTANTIALLY IMPROVE** an existing agent's code (version {current_version_id}).

**CRITICAL GOAL: YOU MUST PROPOSE CHANGES.**
Do not just return the same code. Your purpose is to *evolve* the agent.
If the agent is failing, fix the bug.
If the agent is succeeding, try to make it more robust, efficient, or improve its system prompt for clarity.

A common failure is a bad System Prompt or a loop. **You MUST update the agent's System Prompt** to explicitly tell it:
a.  To think step-by-step.
b.  To use its tools for all calculations.
c.  **CRITICAL**: When the tool returns a result (ToolMessage), the agent **MUST** use that result to formulate its final answer. It should **NOT** call the tool again if the answer is already available.
d.  To **end its FINAL response with the format: `#### <number>`**.
e.  The agent's internal LLM *MUST* be `model="{task_model}"`.

**CRITICAL INSTRUCTIONS:**
1.  **YOU MUST MODIFY THE CODE.** Returning the original code is a failure and will be discarded.
2.  **Your ENTIRE response MUST be the raw, complete Python code.** Do not add explanations or markdown.
3.  **DO NOT wrap the code in ```python ... ```.**
4.  The script *MUST* preserve the function: `def run_agent(question: str) -> str:`
5.  The script *MUST* preserve the `if __name__ == "__main__":` block.
"""

    user_prompt = f"""
### AGENT CODE ({current_version_id}) ###
{previous_code}

ANALYSIS REPORT
This agent produced the following failures on the training set:

Failed Examples: {failures_report}

Successful Examples (for reference): {success_report}

YOUR TASK
Generate the complete, raw Python code for the next version that fixes these issues. Remember: YOU MUST MODIFY THE CODE and preserve the run_agent function and if __name__ == '__main__' block. Ensure the agent's internal model is set to model="{task_model}". """

print(f"--- Calling 'Developer Agent' ({meta_model}) to evolve from {current_version_id}... ---")
try:
    response = client.chat.completions.create(
        model=meta_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
    )
    update_token_stats(response, meta_model) # Log tokens
    generated_code = response.choices[0].message.content
    return clean_generated_code(generated_code)
    
except Exception as e:
    print(f"Error calling OpenAI API (Developer Agent): {e}")
    log_event("fatal_errors.log", f"Developer Agent API call for {current_version_id} failed: {e}")
    return None