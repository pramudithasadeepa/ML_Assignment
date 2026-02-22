import os
import logging
from datetime import datetime

# Setup logging
def setup_logging(log_level=logging.INFO):
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_filename = datetime.now().strftime("ml_project_%Y%m%d_%H%M%S.log")
    log_path = os.path.join(log_dir, log_filename)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("ML_Project")

logger = setup_logging()

# Paths configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")

def ensure_dirs():
    dirs = [MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            logger.info(f"Created directory: {d}")

if __name__ == "__main__":
    ensure_dirs()
    logger.info("Project directories ensured.")
