# Python imports
from pathlib import Path
# Numpy/Scipy imports
import scipy.io
# JAX imports
import jax.numpy as jnp
from jax import random

# CR-Sparse imports
import cr.sparse.cluster as crcluster
import cr.sparse.cluster.ssc as ssc
import cr.sparse as crs

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


    def select_images(self, indices):
        images = self.faces[indices, :]
        n = len(indices)
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
        
    def subject_list_labels_sizes(self, subject_list):
        subject_list = jnp.asarray(subject_list)
        sizes = self.cluster_sizes[subject_list]
        labels = crcluster.labels_from_sizes(sizes)
        return labels, sizes

    def column_wise_data_to_images(self, data):
        n = data.shape[1]
        images = jnp.reshape(data.T, (n, self.image_width, self.image_height))
        # we need to transpose images
        images = jnp.transpose(images, (0, 2,1))
        return images


def analyze_subject_list(yale, subject_list):
    X = yale.subject_list_data(subject_list)
    labels, sizes = yale.subject_list_labels_sizes(subject_list)
    total = jnp.sum(sizes)
    print('Cluster sizes: ', sizes)
    print(f'Total images: {total}')
    angles = ssc.angles_between_points(X)
    # Minimum and maximum angles between any pairs of points in X
    print(f'Minimum angle between any pair of points: {crs.off_diagonal_min(angles):.2f} degrees.')
    print(f'Maximum angle between any pair of points: {crs.off_diagonal_max(angles):.2f} degrees.')
    # Minimum angle for  each point w.r.t. points inside the cluster to which it belongs
    inside_mins = ssc.min_angles_inside_cluster(angles, sizes)
    # Minimum angle for  each point w.r.t. points outside its own cluster
    outside_mins = ssc.min_angles_outside_cluster(angles, sizes)
    # Difference between the angle of the nearest point within the cluster and nearest point outside the cluster
    diff_mins = outside_mins - inside_mins
    # The points for which the closest point is outside the same cluster
    print('Number of points with nearest neighbor outside cluster: ',  jnp.sum(diff_mins < 0))   
    # indices of the nearest neighbors inside the cluster for each point
    inn = ssc.nearest_neighbors_inside_cluster(angles, sizes)
    # indices of the nearest neighbors outside the cluster for each point
    onn = ssc.nearest_neighbors_outside_cluster(angles, sizes)
    # indices for the sorted neighbors for each point across all points
    sorted_neighbors = ssc.sorted_neighbors(angles)
    # map the neighbor index lists to corresponding cluster labels
    sorted_neighbor_labels = yale.labels[sorted_neighbors]
    # Indices for the nearest neighbor inside the same cluster in the neighbor list for each point
    inn_positions = ssc.inn_positions(labels, sorted_neighbor_labels)
    # Percentage of points whose nearest neighbor is not within the same cluster
    jnp.sum(inn_positions > 0) * 100 / total
    # inn position statistics
    unique, counts = jnp.unique(inn_positions, return_counts=True)
    percentages = counts * 100. / total
    print('Inn position labels, counts, percentages: ')
    print(jnp.round(jnp.asarray((unique, counts, percentages)), 1))
    # percentage of points for which the inn position is 0
    zero_inn_position_perc = float(counts[0] * 100 / total)
    print(f'Zero inn position: {zero_inn_position_perc:.2f} %')
    non_zero_inn_position_perc = 100 - zero_inn_position_perc
    print(f'Non-zero inn position: {non_zero_inn_position_perc:.2f} %')








