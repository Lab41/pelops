import sys

class _Const:
    """ Create a constant class
    """
    class ConstError(TypeError): pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError("Cannot rebind constant {}".format(name))
        self. __dict__[name] = value

    def __delattr__(self, name):
        if name in self.__dict__:
            raise self.ConstError("Cannot unbind constant {}".format(name))
        raise NameError(name)

sys.modules[__name__] = _Const()