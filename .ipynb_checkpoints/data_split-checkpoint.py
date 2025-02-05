import random

# Configuration
INPUT_FILE = "data/Indian_Tax_Law.txt"
TRAIN_FILE = "data/Indian_Tax_Law_train.txt"
VAL_FILE = "data/Indian_Tax_Law_val.txt"
VAL_SPLIT = 0.2  # 10% for validation
SEED = 42  # For reproducibility

def split_file(input_file, train_file, val_file, val_split=0.1, seed=42):
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Shuffle the lines
    random.shuffle(lines)
    
    # Calculate split index
    split_idx = int(len(lines) * (1 - val_split))
    
    # Write to train file
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(lines[:split_idx])
    
    # Write to validation file
    with open(val_file, 'w', encoding='utf-8') as f:
        f.writelines(lines[split_idx:])
    
    print(f"Split complete: {len(lines[:split_idx])} lines -> {train_file}")
    print(f"               {len(lines[split_idx:])} lines -> {val_file}")

if __name__ == "__main__":
    split_file(INPUT_FILE, TRAIN_FILE, VAL_FILE, VAL_SPLIT, SEED)