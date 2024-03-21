""" Module src/lib/utils/parse.py (Author: Charley Zhang, 2021)

Parse parameters and display useful error messages.
As the saying goes, assume you have a bunch of monkeys using your code.
"""

import string
import numbers



### ------ #      Individual Numbers      # ----- ###

def parse_bool(param, name): 
    """ In python booleans are implemented as ints (so this is in numbers). """
    try:
        param = bool(param)
    except:
        raise ValueError(f'"{name}" must be a boolean or bool convertible.')
    return param


def parse_int(param, name):
    try:
        param = int(param)
    except:
        raise ValueError(f'"{name}" must be an integer or int convertible')
    return param


def parse_nonnegative_int(param, name):
    param = parse_int(param, name)
    
    msg = (f'"{name}" must be 0 or a natural number. '
            f'You gave: {param} (after auto int conversion)')
    assert param >= 0, msg
    
    return param


def parse_positive_int(param, name):
    param = parse_int(param, name)
    
    msg = (f'"{name}" must be a natural number. '
            f'You gave: {param} (after auto int conversion)')
    assert param > 0, msg
    
    return param


def parse_float(param, name):
    try:
        param = float(param)
    except:
        raise ValueError(f'"{name}" must be a float or float-convertible')
    return param


def parse_probability(param, name):
    param = parse_float(param, name)
    
    msg = f'Probability must be a number in [0, 1], not {param}'
    if not (0 <= param <= 1):
        raise ValueError(msg)
    return param


### ------ #      Number Ranges      # ----- ###

def parse_range(nums_range, name, out_min=None, out_max=None, out_type=None):
    r"""
    Args:
        nums_range: single number or tuple of 2 numbers
            If single positive number, -n, n will be returned.
        name: Name of the parameter for an informative error message.
        out_min: Minimal value that range can take,
            default is None, i.e. there is no minimal value.
        out_max: Maximal value that range can take,
            default is None, i.e. there is no maximal value.
        type_constraint: Precise type output range must take.
    Returns:
        A tuple of two numbers (min_range, max_range).
    """
    if isinstance(nums_range, numbers.Number):  # single number given
        if nums_range < 0:
            raise ValueError(
                f'If {name} is a single number,'
                f' it must be positive, not {nums_range}')
        if out_min is not None and nums_range < out_min:
            raise ValueError(
                f'If {name} is a single number, it must be greater'
                f' than {out_min}, not {nums_range}'
            )
        if out_max is not None and nums_range > out_max:
            raise ValueError(
                f'If {name} is a single number, it must be smaller'
                f' than {out_max}, not {nums_range}'
            )
        if out_type is not None:
            if not isinstance(nums_range, out_type):
                raise ValueError(
                    f'If {name} is a single number, it must be of'
                    f' type {out_type}, not {type(nums_range)}'
                )
        return (-nums_range, nums_range)

    try:
        min_value, max_value = nums_range
    except (TypeError, ValueError):
        raise ValueError(
            f'If {name} is not a single number, it must be'
            f' a sequence of len 2, not {nums_range}'
        )

    min_is_number = isinstance(min_value, numbers.Number)
    max_is_number = isinstance(max_value, numbers.Number)
    if not min_is_number or not max_is_number:
        message = (
            f'{name} values must be numbers, not {nums_range}')
        raise ValueError(message)

    if min_value > max_value:
        raise ValueError(
            f'If {name} is a sequence, the second value must be'
            f' equal or greater than the first, but it is {nums_range}')

    if out_min is not None and min_value < out_min:
        raise ValueError(
            f'If {name} is a sequence, the first value must be greater'
            f' than {out_min}, but it is {min_value}'
        )

    if out_max is not None and max_value > out_max:
        raise ValueError(
            f'If {name} is a sequence, the second value must be smaller'
            f' than {out_max}, but it is {max_value}'
        )

    if out_type is not None:
        min_type_ok = isinstance(min_value, out_type)
        max_type_ok = isinstance(max_value, out_type)
        if not min_type_ok or not max_type_ok:
            raise ValueError(
                f'If "{name}" is a sequence, its values must be of'
                f' type "{out_type}", not "{type(nums_range)}"'
            )
    return nums_range