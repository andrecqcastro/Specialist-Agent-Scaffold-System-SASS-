from datasets import load_dataset, DatasetDict

def get_prepared_dataset(dataset_name: str) -> DatasetDict | None:
    """
    Loads, shuffles, and splits the specified dataset.
    Currently hardcoded for gsm8k structure.
    """
    print(f"--- Loading and Preparing Dataset: {dataset_name} ---")
    
    if dataset_name != "openai/gsm8k":
        print(f"Warning: Dataset '{dataset_name}' is not 'openai/gsm8k'.")
        print("This script is optimized for gsm8k partitioning. Using default splits.")
        # Fallback for other datasets - may not work if splits are named differently
        try:
            return load_dataset(dataset_name)
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            return None

    try:
        # Specific partitioning logic for gsm8k
        dataset_original = load_dataset(dataset_name, "main")
        train_full_shuffled = dataset_original['train'].shuffle(seed=42)
        test_full_shuffled = dataset_original['test'].shuffle(seed=42)

        # Using small, fixed subsets for rapid testing
        train_subset = train_full_shuffled.select(range(185))
        validation_subset = train_full_shuffled.select(range(185, 200))
        test_subset = test_full_shuffled.select(range(50))

        partitioned_dataset = DatasetDict({
            "train": train_subset,
            "validation": validation_subset,
            "test": test_subset
        })
        
        print("Dataset successfully re-partitioned:")
        print(partitioned_dataset)
        return partitioned_dataset
        
    except Exception as e:
        print(f"Error loading or partitioning dataset {dataset_name}: {e}")
        return None