
import copy


def nested_dotdict(nested_dict):
    if not isinstance(nested_dict, dict):
        return nested_dict
    new_dict = {k: nested_dotdict(nested_dict[k]) for k in nested_dict}
    return DotDict(new_dict)

class DotDict(dict):
    """ Dictionary that allows dot notation access (nested not supported). """
    # __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    def __getattr__(self, item):
        try:
            return super().__getitem__(item)
        except KeyError:
            raise AttributeError(item)
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result