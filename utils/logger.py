import logging
import os
from datetime import datetime

LOGFILE = f"{datetime.now().strftime("%d_%m_%y_%H_%M_%S")}"
logfile_dir = os.path.join(os.getcwd(), "logs", LOGFILE)

os.makedirs(logfile_dir, exist_ok=True)

curr_logfile = os.path.join(logfile_dir, LOGFILE)

logging.basicConfig(
    filename=curr_logfile,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
