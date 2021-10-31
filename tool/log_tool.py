import logging
import time


def log_tool_init(model_start_time,
                  level=logging.INFO,
                  console_level=logging.INFO,
                  no_console=False):

    # clear handlers
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.root.handlers = []

    # define log dir path & log file path

    log_path = "log_{}".format(model_start_time)

    # make log handler
    logging.root.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logfile = logging.FileHandler(log_path)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)

    if not no_console:
        log_console = logging.StreamHandler()
        log_console.setLevel(console_level)
        log_console.setFormatter(formatter)
        logging.root.addHandler(log_console)
