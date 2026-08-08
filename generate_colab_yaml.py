import yaml

train_dir = "/content/drive/MyDrive/Shingan_AI/train"

with open("configs/experiment_004.yaml", "r") as f:
    config = yaml.safe_load(f)

config["training"]["epochs"] = 50
config["data"]["train_dir"] = train_dir
config["data"]["val_dir"] = train_dir
config["data"]["test_dir"] = train_dir
config["tracking"]["save_dir"] = "/content/drive/MyDrive/Shingan_AI/experiments"
config["tracking"]["experiment_name"] = "experiment_004_full"

with open("configs/experiment_004_colab.yaml", "w") as f:
    yaml.dump(config, f, sort_keys=False)

print("Created experiment_004_colab.yaml")
