import logging


class Logger:
    def __init__(self, file):
        self.logger = logging.getLogger()

        if len(self.logger.handlers) == 0:
            file_handler = logging.FileHandler(file)

            self.logger.addHandler(file_handler)
            self.logger.setLevel(logging.INFO)

    def info(self, value):
        self.logger.info(f"{value}")

    def error(self, value):
        self.logger.error(f"{value}")
