import os
import random
import glob

def create_splits():
    # Target 80/10/10
    seed = 42
    random.seed(seed)
    
    # We will look at GT folder
    # In Colab it's /content/drive/MyDrive/Shingan_AI/train/GT, locally d:\Shangan\datasets\train\GT
    local_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'train', 'GT'))
    if not os.path.exists(local_dir):
        print(f"Directory {local_dir} not found. Ensure you have data.")
        # Alternatively, we could just write the split script to be run on Colab
    
    files = sorted(os.listdir(local_dir))
    files = [f for f in files if f.endswith('.npy')]
    
    if not files:
        print("No files found.")
        return
        
    random.shuffle(files)
    
    n = len(files)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    
    train_files = files[:n_train]
    val_files = files[n_train:n_train+n_val]
    test_files = files[n_train+n_val:]
    
    split_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'splits'))
    os.makedirs(split_dir, exist_ok=True)
    
    with open(os.path.join(split_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_files))
        
    with open(os.path.join(split_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_files))
        
    with open(os.path.join(split_dir, 'test.txt'), 'w') as f:
        f.write('\n'.join(test_files))
        
    print(f"Splits created in {split_dir}:")
    print(f"Train: {len(train_files)}")
    print(f"Val: {len(val_files)}")
    print(f"Test: {len(test_files)}")

if __name__ == "__main__":
    create_splits()
