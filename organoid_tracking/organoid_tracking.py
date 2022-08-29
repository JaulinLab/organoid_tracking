import numpy as np
import trackpy as tp
import pandas as pd
import skimage.measure


def get_mask_properties(label_image, properties=("centroid", "perimeter", "area")):

    mask_properties = skimage.measure.regionprops_table(
        label_image, properties=properties
    )

    mask_properties = pd.DataFrame(mask_properties).rename(
        {"centroid-0": "x", "centroid-1": "y"}, axis="columns"
    )

    return mask_properties
