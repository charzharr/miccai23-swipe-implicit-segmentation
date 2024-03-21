
import torch
from lib.utils.parse import parse_bool


class ValueSampler:
    """ Object that uniformly samples from dicrete or continues sets. 
    > Discrete means it uniformly samples from a set collection of values.
    > Continues means it uiformly samples from a range of 2 values. 
    """
    
    def __init__(self, is_discrete, values):
        """
        Args:
            is_discrete (bool): are set of values discrete or continuous
            values: if discrete, then a sequence of values to sample from,
                else assumes that it describes an inclusive range.
        """
        self.is_discrete = parse_bool(is_discrete, 'is_discrete')
        self.values = ValueSampler.parse_values(is_discrete, values)
    
    def sample(self):
        return ValueSampler.sample_value(self.is_discrete, self.values, 
                                         _check_input=False)
        
    @staticmethod
    def sample_value(is_discrete, values, _check_input=True):
        if _check_input:
            is_discrete = parse_bool(is_discrete, 'is_discrete')
            values = ValueSampler.parse_values(is_discrete, values)
        if is_discrete:
            return values[rand_int(0, len(values) - 1)]
        else:
            return rand_float(values[0], values[1])
            
    @staticmethod
    def parse_values(is_discrete, values):
        if is_discrete:
            assert len(values) > 0, 'Values must be a non-empty collection.'
            values = list(values)  # use list over set bc can over-rep val
        else:
            assert len(values) == 2, 'Value range must be a seq of len 2.'
            values = sorted(list(values))
        return values
    
    def __repr__(self):
        string = (f'[ValueSampler] - Discrete: {self.is_discrete}, '
                  f'Vals: {self.values}')
        return string


def rand_int(low_incl, high_incl):
    """ Sample a random integer via torch inclusively between low & high. """
    return torch.randint(int(low_incl), int(high_incl + 1), (1,)).item()


def rand_float(low_incl, high_incl):
    """ Sample a random float via torch inclusively between low & high. 
        Note that the high_incl is only approximately inclusive. 
    """
    return low_incl + (high_incl - low_incl) * torch.rand(1).item()

