# Specialist Agent Scaffold-System (SASS) #

This repository provides a scaffolding system for creating, evaluating, and evolving **Specialist Agents** using a genetic algorithm inspired by Deep Genetic Manager (DGM).

The system uses a `meta_model` (like GPT-4o) to act as a "Developer" that iteratively debugs and improves a `task_model` (a smaller agent, like GPT-4o-mini) based on its performance on a given dataset.

## Requirements

* Python 3.10+
* An OpenAI API key set as an environment variable (`OPENAI_API_KEY`)

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/andrecqcastro/Specialist-Agent-Scaffold-System-SASS-.git
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment:**
    Create a `.env` file in the root directory and add your API key:
    ```
    OPENAI_API_KEY="sk-..."
    ```

##  Running

The main script `main.py` accepts command-line arguments to configure the evolution run.

**Example (using defaults):**

```bash
python main.py