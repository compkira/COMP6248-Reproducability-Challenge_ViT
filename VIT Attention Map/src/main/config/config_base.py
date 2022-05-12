class ConfigBase(object):
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def parse(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        print(f"User {self.__class__.__name__}:")
        for attr in self._getattr():
            print("%-16s| %s" % (attr, getattr(self, attr)))
        print('')

    def to_dict(self):
        cfg = {}
        for attr in self._getattr():
            cfg[attr] = getattr(self, attr)
        return cfg

    def _getattr(self):
            return list(filter(lambda l: not l.startswith('_'), list(self.__class__.__dict__.keys())))
