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


class WikiText103Dataset(BaseTextDataset):
    """
        Dataset loader for the raw WikiText-103 corpus.
    Assumes archive is already unpacked into `data_dir`.
        """

    SPLIT_FILES = {
        "train": "wiki.train.raw",
        "valid": "wiki.valid.raw",
        "test": "wiki.test.raw",
    }

    def __init__(self, data_dir: str = "data/wikitext-103-raw", split: str = "train"):
        if split not in self.SPLIT_FILES:
            raise ValueError(
                f"Unknown split '{split}'. Expected one of {list(self.SPLIT_FILES.keys())}"
            )

        self.data_dir = data_dir
        self.split = split
        self.filename = self.SPLIT_FILES[split]
        self.file_path = os.path.join(self.data_dir, self.filename)

        logger.debug(
            "Initialized WikiText103Dataset with data_dir=%s, split=%s, file_path=%s",
            data_dir,
            split,
            self.file_path,
        )

    def load(self) -> str:
        if not os.path.exists(self.file_path):
            logger.error("Dataset file not found at %s", self.file_path)
            raise FileNotFoundError(
                f"Expected WikiText-103 split file at '{self.file_path}'. "
                "Ensure wikitext-103-raw-v1.zip has been extracted into that directory."
            )

        logger.info("Reading WikiText-103 (%s split) from %s", self.split, self.file_path)
        with open(self.file_path, "r", encoding="utf-8") as f:
            text = f.read()

        logger.info("WikiText-103 (%s) length: %d characters", self.split, len(text))
        logger.info("Sample text:\n%s", text[:150])
        return text