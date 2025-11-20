"""
Configuration loading utilities.
"""

import yaml


def load_config(config_path):
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary with configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_model_config(config):
    """Get model configuration"""
    return config.get('model', {})


def get_data_config(config):
    """Get data configuration"""
    return config.get('data', {})


def get_training_config(config):
    """Get training configuration"""
    return config.get('training', {})


def get_callbacks_config(config):
    """Get callbacks configuration"""
    return config.get('callbacks', {})


def get_inference_config(config):
    """Get inference configuration"""
    return config.get('inference', {})


def get_paths_config(config):
    """Get paths configuration"""
    return config.get('paths', {})
