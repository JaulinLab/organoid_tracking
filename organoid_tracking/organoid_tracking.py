import numpy as np
import trackpy as tp
import pandas as pd
import skimage.measure
import tifffile
import os
import matplotlib.pyplot as plt

def get_mask_properties(mask_image, properties=("centroid", "perimeter", "area", "label")):

    """From a binary image, get the properties of the objects in the image.
    
    Parameters
    ----------
    mask_image : np.ndarray
        A binary image with objects labeled with integers.
    properties : tuple, optional
        A tuple of properties to extract from the image. The default is ("centroid", "perimeter", "area", "label").
    
    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame with the properties of the objects in the image.
    
    """

    # Check that mask_image is a binary image, binerize if not
    if not np.array_equal(np.unique(mask_image), [0, 1]):
        mask_image = mask_image > 0
        
    label_image = skimage.measure.label(mask_image)

    mask_properties = skimage.measure.regionprops_table(
        label_image, properties=properties
    )

    mask_properties = pd.DataFrame(mask_properties).rename(
        {"centroid-0": "y", "centroid-1": "x"}, axis="columns"
    )

    return mask_properties

def single_movie_trajs(image_sequence,
                       max_step = 50):

    """ From images sequences to trajectories """

    organoid_data = pd.DataFrame()

    for image, frame in zip(image_sequence, range(len(image_sequence))):
        
        properties = get_mask_properties(image)
        properties['frame'] = frame
        
        organoid_data = pd.concat([organoid_data, properties], 
                                ignore_index=True)

    organoid_data = tp.link(organoid_data, max_step, memory=3)

    return organoid_data

def get_particle_props(dataframe):

    for particle in dataframe.particle.unique():

        particleframe = dataframe[dataframe.particle == particle].copy()
        dataframe = dataframe.drop(dataframe[dataframe.particle == particle].index)

        particleframe['dx'] = particleframe.x - particleframe.x.shift()
        particleframe['dy'] = particleframe.y - particleframe.y.shift()
        particleframe['velocity'] = np.sqrt(particleframe.dx**2 + particleframe.dy**2)
        particleframe['cumulative_displacement'] = particleframe['velocity'].cumsum()

        xstart= particleframe.loc[particleframe.frame.idxmin(), 'x']
        ystart= particleframe.loc[particleframe.frame.idxmin(), 'y']

        particleframe['absolute_displacement'] = np.sqrt((particleframe.x - xstart)**2 + (particleframe.y - ystart)**2)

        dataframe = pd.concat([dataframe, particleframe])

    return dataframe

def movie_analysis(filename, output_directory):

    movie_name, _ = filename.split('.')
    image_sequence = tifffile.imread(filename)

    # if spurious channels remove extra RGB one
    if image_sequence.ndim == 4:
        single_channel_image_sequence = image_sequence[..., 0]
        image_sequence = single_channel_image_sequence
        del(single_channel_image_sequence)

    # check that dimensions are OK
    assert image_sequence.ndim == 3

    movie_frame = single_movie_trajs(image_sequence,
                       max_step = 100)
    movie_frame = get_particle_props(movie_frame)
    movie_frame['movie_name'] = movie_name
    movie_frame['filename'] = filename

    plot_verification_image(movie_frame, output_directory)

    return movie_frame

def plot_verification_image(movie_frame, output_directory):

    directory = os.path.dirname(movie_frame.filename.unique()[0])
    mvname = os.path.basename(movie_frame.filename.unique()[0])
    mvname, _ = mvname.split('.')
    mvname += '.pdf'
    directory = os.path.join(output_directory, 'verification')

    if not os.path.exists(directory):
        os.makedirs(directory)

    image = tifffile.imread(movie_frame.filename.unique()[0])[0]

    fig, ax = plt.subplots(figsize=(4,2))
    plt.imshow(image)
    ax = tp.plot_traj(movie_frame, color = 'k')

    for particle in movie_frame.particle.unique():
        xtext = int(movie_frame.loc[(movie_frame.particle == particle), 'x'].mean())
        ytext = int(movie_frame.loc[(movie_frame.particle == particle), 'y'].mean())

        ax.text(x = xtext, y = ytext, s = str(particle), c = 'w', fontsize = 10)

    fig.savefig(os.path.join(directory, mvname))
    plt.close()
    
    return


