import yaml

from yaml import Loader
from typing import Optional


class Config:
    def __init__(self, dictionary):
        self.dict = dictionary

        if not dictionary:
            return

        def _traverse(key, elem):
            if isinstance(elem, dict):
                return key, Config(elem)
            else:
                return key, elem

        obj = dict(_traverse(k, v) for k, v in dictionary.items())
        self.__dict__.update(obj)

    def to_dict(self):
        return self.dict

    @staticmethod
    def load(path: Optional[str]):
        if not path:
            return Config(None)
        else:
            with open(path, mode='r') as file:
                config = yaml.load(file, Loader=Loader)
                return Config(config)
