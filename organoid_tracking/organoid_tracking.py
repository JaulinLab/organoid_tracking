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
                       search_range,
                       memory):

    """ From images sequences to trajectories """

    organoid_data = pd.DataFrame()

    for image, frame in zip(image_sequence, range(len(image_sequence))):
        
        properties = get_mask_properties(image)
        properties['frame'] = frame
        
        organoid_data = pd.concat([organoid_data, properties], 
                                ignore_index=True)

    organoid_data = tp.link(organoid_data, search_range=search_range, memory=memory)

    # loop through all particles, if a frame is missing insert an empty row

    maxframe = organoid_data.frame.max()

    for particle in organoid_data.particle.unique():
            
            particleframe = organoid_data[organoid_data.particle == particle].copy()
            organoid_data = organoid_data.drop(organoid_data[organoid_data.particle == particle].index)
    
            for frame in range(0, maxframe):
    
                if frame not in particleframe.frame.unique():

                    # Create the new row as a DataFrame
                    new_row = pd.DataFrame({'x':np.nan, 'y':np.nan, 'frame':frame, 'particle':particle}, index = [particleframe.index.max()+1])

                    # Append the new row to the DataFrame using pd.concat()
                    particleframe = pd.concat([particleframe, new_row], ignore_index=True)
        
            particleframe = particleframe.sort_values(by = 'frame')
            particleframe = particleframe.reset_index(drop = True)
            organoid_data = pd.concat([organoid_data, particleframe])

    return organoid_data

def get_particle_props(dataframe):

    for particle in dataframe.particle.unique():

        particleframe = dataframe[dataframe.particle == particle].copy()
        dataframe = dataframe.drop(dataframe[dataframe.particle == particle].index)

        particleframe['dx'] = particleframe.x - particleframe.x.shift()
        particleframe['dy'] = particleframe.y - particleframe.y.shift()
        particleframe['velocity'] = np.sqrt(particleframe.dx**2 + particleframe.dy**2)
        particleframe['cumulative_displacement'] = particleframe['velocity'].cumsum()
        particleframe['average_velocity'] = particleframe['velocity'].mean()

        if not particleframe.empty:

            min_non_empty_x = particleframe.x.notna().index.min()
            x_start = particleframe.loc[min_non_empty_x, 'x']
            y_start = particleframe.loc[min_non_empty_x, 'y']
            particleframe['absolute_displacement_x'] = particleframe['x'] - x_start
            particleframe['absolute_displacement_y'] = particleframe['y'] - y_start

        dataframe = pd.concat([dataframe, particleframe])

    return dataframe

def movie_analysis(filename, output_directory):

    movie_name, _ = filename.split('.')
    experience_name = movie_name.split('/')[-4]
    condition_name = movie_name.split('/')[-3]
    movie_name = movie_name.split('/')[-1]
    image_sequence = tifffile.imread(filename)

    # if spurious channels remove extra RGB one
    if image_sequence.ndim == 4:
        single_channel_image_sequence = image_sequence[..., 0]
        image_sequence = single_channel_image_sequence
        del(single_channel_image_sequence)

    # check that dimensions are OK
    assert image_sequence.ndim == 3

    if len(np.unique(image_sequence)) > 1:

        movie_frame = single_movie_trajs(image_sequence,
                        search_range = 250,
                        memory = 10)
        movie_frame = get_particle_props(movie_frame)
        movie_frame['movie_name'] = movie_name
        movie_frame['experience_name'] = experience_name
        movie_frame['condition_name'] = condition_name
        movie_frame['filename'] = filename

        #plot_verification_image(movie_frame, output_directory)

        return movie_frame
    
    else:

        return pd.DataFrame()

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


