import random
from typing import Optional

import yaml
from yaml import Loader


class Config:
    def __init__(self, dictionary):
        self.dict = dictionary

        def _traverse(key, elem):
            if isinstance(elem, dict):
                return key, Config(elem)
            else:
                return key, elem

        obj = dict(_traverse(k, v) for k, v in dictionary.items())
        self.__dict__.update(obj)

    def to_dict(self):
        return self.dict


def load_config_dict(path: Optional[str]):
    if not path:
        return None
    else:
        with open(path, mode='r') as file:
            config = yaml.load(file, Loader=Loader)
            config.setdefault('seed', random.randint(0, 100))
            return config
