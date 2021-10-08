# Python imports
from pathlib import Path
# Numpy/Scipy imports
import scipy.io
# JAX imports
import jax.numpy as jnp
from jax import random

# CR-Sparse imports
import cr.sparse.cluster as crcluster

# Data directory
here = Path(__file__).parent
experiments_dir = here.parent
package_dir = experiments_dir.parent
data_dir = package_dir / 'data'
yale_b_dir = data_dir / 'yale_b'
extend_yale_b_file = yale_b_dir / 'ExtendedYaleB.mat'
yale_b_crop_file = yale_b_dir / 'YaleBCrop025.mat'



class YaleFaces:

    def __init__(self):
        contents = scipy.io.loadmat(extend_yale_b_file, 
            squeeze_me=True, variable_names=('EYALEB_DATA', 'EYALEB_LABEL'))
        self.faces = jnp.array(contents['EYALEB_DATA']).T
        self.labels = jnp.array(contents['EYALEB_LABEL'] - 1)
        self.num_subjects = 38
        self.image_height = 48
        self.image_width = 42
        self.cluster_sizes = crcluster.sizes_from_labels_jit(self.labels, self.num_subjects)
        start, end = crcluster.start_end_indices(self.cluster_sizes)
        self.start_indices = start
        self.end_indices = end

    @property
    def num_total_faces(self):
        return self.faces.shape[0]

    def subject_images(self, i):
        """Returns the faces of a particular subject"""
        start = self.start_indices[i]
        end = self.end_indices[i]
        images = self.faces[start:end, :]
        n = end - start
        images = jnp.reshape(images, (n, self.image_width, self.image_height))
        # we need to transpose images
        images = jnp.transpose(images, (0, 2,1))
        return images

    def random_images(self, key, n):
        indices = random.permutation(key, self.num_total_faces)[:n]
        images = self.faces[indices, :]
        images = jnp.reshape(images, (n, self.image_width, self.image_height))
        # we need to transpose images
        images = jnp.transpose(images, (0, 2,1))
        return images


    def subject_data(self, i):
        """Returns subject data as columns
        """
        start = self.start_indices[i]
        end = self.end_indices[i]
        images = self.faces[start:end, :]
        return images.T


    def subject_list_data(self, subject_list):
        all_data = [self.subject_data(subject_idx) for subject_idx in subject_list]
        return jnp.hstack(all_data)
        

    def column_wise_data_to_images(self, data):
        n = data.shape[1]
        images = jnp.reshape(data.T, (n, self.image_width, self.image_height))
        # we need to transpose images
        images = jnp.transpose(images, (0, 2,1))
        return images
