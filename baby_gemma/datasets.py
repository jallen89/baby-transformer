import os
import urllib.request
import logging
from abc import ABC, abstractmethod


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class BaseTextDataset:

    @abstractmethod
    def load(self) -> str:
        """Download (if necessary) and return the full text corpus."""
        pass

class TinyShakeSpeareDataset:

    def __init__(self, 
                 filename='tinyshakespeare.txt'):
        self.filename = filename
        self.url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        logger.debug("Initialized TinyShakeSpeareDataset with filename=%s", filename)


    def load(self) -> str:
        file_path = self.filename
        if not os.path.exists(file_path):
            logger.info("Downloading TinyShakespeare dataset from %s to %s", self.url, file_path)
            urllib.request.urlretrieve(self.url, file_path)
        else:
            logger.info("Using existing dataset file at %s", file_path)

        logger.debug("Reading dataset file from %s", file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        logger.info("Dataset length: %d characters", len(text))
        logger.info("Sample text:\n%s", text[:150])
        return text