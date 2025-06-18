import os
from datetime import datetime
import logging

# 1. Create log filename
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# 2. Logs folder path
logs_dir = os.path.join(os.getcwd(), "logs")
print("Logs directory:", logs_dir)

# 3. Make sure folder exists
os.makedirs(logs_dir, exist_ok=True)

# 4. Full path to log file
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)
print("Log file path:", LOG_FILE_PATH)

# 5. Set up logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
)
