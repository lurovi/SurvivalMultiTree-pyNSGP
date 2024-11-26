import numpy as np
import random


def choice(vector):
    return vector[int(random.random() * len(vector))]
