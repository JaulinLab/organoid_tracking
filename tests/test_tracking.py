import pytest
import numpy as np
import trackpy as tp
import random

import organoid_tracking

@pytest.fixture
def test_make_organoid_masks(x,y,r,L):

    X,Y = np.ogrid[:L, :L]

    assert isinstance(x, list)
    assert isinstance(y, list)
    assert len(x) == len(y)

    test_image = np.zeros((L, L))

    for xn, yn, n in zip(x,y,range(len(x))):

        distance = np.sqrt((X-xn)**2 + (Y-yn)**2)
        test_image[distance<r] = n

    return test_image.astype(int)

@pytest.fixture
def test_image():

    # number of organoids
    n = random.sample([1,2,3,4], 1)[0]
    L = 30

    # organoid positions
    r = 5
    x = random.sample(range(0, L), n)
    y = random.sample(range(0, L), n)
    
    test_image = test_make_organoid_masks(x,y,r,L)

    return test_image.astype(int)

@pytest.fixture
def test_image_sequence():
    n = 5
    return [test_image() for  i in range(n)]


def test_segmentation(test_image):

    num_organoids = len(np.unique(test_image)) - 1
    mask_frame = organoid_tracking.get_mask_properties(test_image)
    num_detected_organoids = len(mask_frame)

    assert num_organoids == num_detected_organoids
    return





