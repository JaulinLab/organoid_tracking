import numpy as np
import trackpy as tp
import pandas as pd
import skimage.measure


def get_mask_properties(mask_image, properties=("centroid", "perimeter", "area", "label")):

    label_image = skimage.measure.label(mask_image)

    mask_properties = skimage.measure.regionprops_table(
        label_image, properties=properties
    )

    mask_properties = pd.DataFrame(mask_properties).rename(
        {"centroid-0": "y", "centroid-1": "x"}, axis="columns"
    )

    return mask_properties
