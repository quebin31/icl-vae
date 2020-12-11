import random
from typing import Optional

import ruamel.yaml


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
        yaml = ruamel.yaml.YAML(typ="safe")
        with open(path, mode='r') as file:
            config = yaml.load(file)
            config.setdefault('seed', 'random')
            config.setdefault('rho', 0.5)

            if config['seed'] == 'random':
                config['seed'] = random.randint(0, 1000)

        with open(path, mode='w') as file:
            yaml.default_flow_style = False
            yaml.dump(config, file)

        return config
