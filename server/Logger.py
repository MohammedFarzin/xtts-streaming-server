import logging

class Log:
    def __init__(self, path):
        self.path = path

    def initialize_logger_handler(self):
        logger = logging.getLogger(__name__)

        # Set the minimum logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        logger.setLevel(logging.DEBUG)

        # Create handlers (console and file)
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(self.path)

        # Set the level for handlers
        console_handler.setLevel(logging.WARNING)
        file_handler.setLevel(logging.DEBUG)

        # Create formatters and add them to the handlers
        console_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        console_handler.setFormatter(console_format)
        file_handler.setFormatter(file_format)

        # Add handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger