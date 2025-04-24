import logging

def setup_logger(filename):
    logging.basicConfig(
        filename=filename,
        # INFO as minimum log level
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='a'
    )
    logging.info("="*100)
    print("="*100)

def log(message):
    logging.info(message)
    print(message)