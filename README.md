![Logo do Projeto](logo/SASS.jpg)


# Specialist Agent Scaffold-System (SASS) #

This repository provides a scaffolding system for creating, evaluating, and evolving **Specialist Agents**. The methodology is inspired by Darwinian principles of selection and adaptation, where a "Developer" LLM guides the evolution.

The system uses a `meta_model` (like GPT-4o) to act as a "Developer" that iteratively debugs and improves a `task_model` (a smaller agent, like GPT-4o-mini) based on its performance on a given dataset.

## Requirements

* Python 3.10+
* An OpenAI API key set as an environment variable (`OPENAI_API_KEY`)

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/andrecqcastro/Specialist-Agent-Scaffold-System-SASS-.git
    ```

2.  **Navigate to the directory:**
    ```bash
    cd Specialist-Agent-Scaffold-System-SASS-
    ```

3.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Set up your environment:**
    Create a `.env` file in the root directory and add your API key:
    ```
    OPENAI_API_KEY="sk-..."
    ```

##  Methodology

This system evolves agents using a "Developer" LLM to improve a "Task" LLM. The process works as follows:

1.  **Initialization (Creator):** The process begins with a "Creator" agent (using the `meta_model`) generating an initial "v0" specialist agent from a base template. This v0 agent is designed to use the specified `task_model` (e.g., `gpt-4o-mini`).

2.  **Initial Evaluation:** The v0 agent is evaluated on a **validation dataset** to get its initial fitness score. It is then added to the **Archive** (a list of all agent versions).

3.  **Evolution Loop (DGM):** The system iterates for a set number of generations (`--iterations`). In each iteration:
    * **a. Parent Selection (Selection):** One or more parents are selected from the Archive. Selection is probabilistic, favoring agents that have both a **high score** (fitness) and a **low `children_count`** (novelty). This mimics natural selection, where fitter individuals are more likely to reproduce, while also encouraging diversity.
    * **b. Failure Analysis (Environmental Pressure):** The chosen parent agent is run on the **training dataset** to identify a specific number (`--max_failures`) of questions it gets wrong. These failures represent the environmental pressures challenging the agent's survival.
    * **c. Evolution (Guided Mutation):** A "Developer" agent (using the `meta_model`) is given the parent's code, the list of failures, and a few success examples. Its task is to analyze the failures and rewrite the parent's code to fix the bugs and improve the logic. This acts as an intelligent, guided mutation rather than a random one.
    * **d. Child Validation (Fitness Test):** The newly generated "child" agent's code is saved, validated (e.g., ensuring it's not identical to the parent), and then evaluated on the **validation dataset** to get its own fitness score.
    * **e. Archive Update (Inheritance):** The new child (with its score and parent ID) is added to the Archive, inheriting its "genes" (code) from the parent and becoming part of the next generation's gene pool.

4.  **Final Selection:** After all iterations are complete, the single best-performing agent (based on the highest validation score) is selected from the Archive.

5.  **Test Set Evaluation:** This "champion" agent is run *one time* on the **test dataset** (which it has never seen before) to provide a final, unbiased measure of its performance.

##  Running

The main script `main.py` accepts command-line arguments to configure the evolution run.

**Example (using defaults):**

```bash
python main.py