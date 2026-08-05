import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config, load_config

def test_config_initialization():
    test_dict = {
        "training": {
            "early_stopping": {
                "enabled": True,
                "patience": 5
            }
        }
    }
    
    cfg = Config(test_dict)
    
    # Assert nested objects are accessible via attributes
    assert hasattr(cfg, 'training')
    assert hasattr(cfg.training, 'early_stopping')
    assert getattr(cfg.training.early_stopping, 'enabled') is True
    assert getattr(cfg.training.early_stopping, 'patience') == 5
    
    # Test safe fallback
    early_stop_cfg = getattr(cfg.training, 'early_stopping', None)
    assert early_stop_cfg is not None
    assert getattr(early_stop_cfg, 'enabled', False) is True

def test_config_missing_attributes():
    test_dict = {
        "training": {}
    }
    
    cfg = Config(test_dict)
    
    # Should safely return default when using getattr
    early_stop_cfg = getattr(cfg.training, 'early_stopping', None)
    assert early_stop_cfg is None
    
    # Should raise AttributeError if accessed directly without getattr (expected behavior)
    with pytest.raises(AttributeError):
        _ = cfg.training.early_stopping
