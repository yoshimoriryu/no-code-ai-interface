import logging

def setup_logging():
    # Define the logging format
    logging_format = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"

    # Set up basic configuration
    logging.basicConfig(
        level=logging.INFO,  # You can change this to DEBUG for more detailed logs
        format=logging_format
    )
    # logger = logging.getLogger("uvicorn.error")
    logger = logging.getLogger(__name__)

    return logger