"""
Monkey patch module to apply custom task configurations to lm_eval.
"""
import os
import shutil
import site
import logging

logger = logging.getLogger(__name__)

def apply_custom_task_configs():
    """Apply custom task configurations by copying our YAML files over the original ones"""
    
    # Get the site-packages directory
    site_packages = site.getsitepackages()[0]
    lm_eval_tasks_dir = os.path.join(site_packages, "lm_eval", "tasks")
    
    # Get the directory where this script is located (eval_config)
    config_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Copy our custom YAML files to replace the originals
    custom_configs = [
        {
            "source": os.path.join(config_dir, "hendrycks_math.yaml"),
            "target": os.path.join(lm_eval_tasks_dir, "hendrycks_math", "hendrycks_math.yaml")
        },
        {
            "source": os.path.join(config_dir, "hendrycks_math_algebra.yaml"), 
            "target": os.path.join(lm_eval_tasks_dir, "hendrycks_math", "hendrycks_math_algebra.yaml")
        },
        {
            "source": os.path.join(config_dir, "gpqa_main_generative_n_shot_yaml"),
            "target": os.path.join(lm_eval_tasks_dir, "gpqa", "generative", "_gpqa_main_generative_n_shot_yaml")
        }
    ]
    
    for config in custom_configs:
        if os.path.exists(config["source"]):
            try:
                shutil.copy2(config["source"], config["target"])
                logger.info(f"Applied custom config: {os.path.basename(config['source'])}")
            except Exception as e:
                logger.warning(f"Failed to apply {os.path.basename(config['source'])}: {e}")
        else:
            logger.warning(f"Custom config not found: {config['source']}")
