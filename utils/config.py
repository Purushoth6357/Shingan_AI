import os
import yaml

class Config:
    def __init__(self, config_dict):
        for k, v in config_dict.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)
                
    def __repr__(self):
        return str(self.__dict__)

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        
    return Config(config_dict)

if __name__ == "__main__":
    # Test loading
    cfg = load_config("../configs/default.yaml")
    print(f"Loaded config: {cfg.experiment_name}")
    print(f"Batch size: {cfg.data.batch_size}")
