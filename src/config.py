import yaml


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

    @staticmethod
    def load(path: str):
        with open(path, mode='r') as file:
            contents = file.read()
            config = yaml.load(contents)
            return Config(config)
