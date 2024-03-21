

import random


# ------ Dataset Instance Manipulation ----- #

def split(indices, train=0.6, val=0.2, test=0.2):
    indices = set(indices)
    train = random.sample(indices, k=math.ceil(0.6 * len(indices)))
    val_test_indices = indices - set(train)
    test = random.sample(val_test_indices, k=math.ceil(0.2 * len(indices)))
    val = list(val_test_indices - set(test))
    return sorted(train), sorted(val), sorted(test)


