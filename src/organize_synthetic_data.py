"""
static module for organizing synthetic motion data in the context of both efforts and similarity networks
"""

from src.batches import Batches
from pymo.parsers import BVHParser
from pymo.viz_tools import *
from sklearn.preprocessing import StandardScaler
from pymo.preprocessing import *
from os import path
import shutil
import conf
from sklearn.pipeline import Pipeline
import numpy as np
import seaborn as sns
import pickle
import warnings
import random

anim_ind = {'WALKING': 0, 'POINTING': 1, 'PICKING': 2, 'WAVING': 3, 'THROWING': 4, 'AIMING': 5, 'JUMPING': 6,
            'RUNNING': 7}
parser = BVHParser()

singleton_batches = Batches()


def visualize(file_bvh):
    """
    Visualize motion data from a BVH file.

    Args:
        file_bvh (str): The path to the BVH file to visualize.

    Returns:
        None
    """
    parsed_data = parser.parse(file_bvh)
    data_pipe = Pipeline([
        ('param1', MocapParameterizer('expmap')),
    ])
    data = parsed_data
    mp = MocapParameterizer('position')
    positions = mp.transform([data])[0]

    nb_play_mocap(positions, 'pos',
                  scale=2, camera_z=800, frame_time=1 / 30,
                  base_url='pymo/mocapplayer/playBuffer2.html')


def clear_file(file):
    """
    Remove character name from a text file.

    Args:
        file (str): The path to the text file.

    Returns:
        None
    """
    # removes character name from the file
    # read input file
    fin = open(file, "rt")
    # read file contents to string
    data = fin.read()
    # replace all occurrences of the required string
    data = data.replace('Carl:', '')
    # close the input file
    fin.close()
    # open the input file in write mode
    fin = open(file, "wt")
    # overwrite the input file with the resulting data
    fin.write(data)
    # close the file
    fin.close()


def prep_all_data_for_training(rotations=True, velocities=False):
    """
    Prepare motion data for training.

    Args:
        rotations (bool): Whether to include rotation values in exemplars.
        velocities (bool): Whether to include velocities in exemplars.

    Returns:
        None
    """
    def _preprocess_pipeline(parsed_data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data_pipe_expmap = Pipeline(steps=[
                ('param', MocapParameterizer('expmap')),
                ('np', Numpyfier()),
                # ('down', DownSampler(4))
            ])
        return data_pipe_expmap.fit_transform([parsed_data])[0]

    def _get_standardized_rotations(data_expmaps):
        data_expmaps = _z_score_generator(data_expmaps)
        return data_expmaps

    def _get_standardized_velocities(data_velocities):
        # needed for proper broadcasting of following step
        frame_rate_array = np.tile(bvh_frame_rate.pop(), (data_velocities.shape[0] - 1, data_velocities.shape[1]))
        # calculate velocities from positions
        data_velocities[1:] = (data_velocities[1:, :] - data_velocities[:-1, :]) / frame_rate_array
        data_velocities[0] = 0
        # standardize velocities
        data_velocities = _z_score_generator(data_velocities)
        return data_velocities

    def _z_score_generator(np_array):
        scaler = StandardScaler()
        scaler = scaler.fit(np_array)
        np_array = scaler.transform(np_array)
        return np_array

    def create_corr_matrix(np_array, name):
        print("NAME:", name)
        dataframe = pd.DataFrame(np_array)
        print("DATAFRAME :", dataframe)
        a = dataframe.corr()
        ax = sns.heatmap(
            a,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True
        )
        plt.title(name)
        plt.show()

    def apply_moving_window(batches, file_data):
        """
        nested function of prep_all_data_for_training()

        handles both construction of effort network batches (with rotations only, each batch has
         dim = batch_size x time_series_size x 87) and similarity network class to exemplar dict.

        Args:
            batches: instance of Batches class
            file_data: np.array comprised of preprocessed motion data + effort values + anim name

        Returns:
            None
        """

        start_index = conf.time_series_size
        end_index = file_data.shape[0]
        for i in range(start_index, end_index + conf.window_delta, conf.window_delta):
            indices = range(i - conf.time_series_size, i)
            # end of file corner case correction
            if i > end_index:
                indices = range(i - conf.time_series_size, end_index)
                exemplar = file_data[indices]
                exemplar = batches.append_to_end_file_exemplar(exemplar)
                batches.append_efforts_batch_and_labels(exemplar)
                if f == filenames[-1]:
                    batches.extend_final_batch(exemplar)
            else:
                exemplar = file_data[indices]
                batches.append_efforts_batch_and_labels(exemplar)
            if tuple_effort_list in batches.dict_similarity_exemplars.keys():
                batches.append_similarity_class_exemplar(tuple_effort_list, file_data[indices])
            if len(batches.current_batch[batches.batch_idx]) == conf.batch_size_efforts_network:
                batches.store_efforts_batch()

    bvh_counter = 0
    bvh_frame_rate = set()
    filenames = os.listdir(conf.bvh_files_dir)
    print(conf.bvh_files_dir)
    print(filenames)
    for f in filenames:
        if f.endswith("bvh"):
            name = path.splitext(f)[0]  # exclude extension bvh by returning the root
            name_split = name.split('_')  # get effort values from the file name
            anim = name_split[0]
            f_full_path = conf.bvh_files_dir + f
            efforts_list = [float(p) for p in name.split('_')[-4:]]
            tuple_effort_list = tuple(efforts_list)
            singleton_batches.state_drive_exemplar_idx = 0
            clear_file(f_full_path)  # remove the : from the file
            parsed_data = parser.parse(f_full_path)  # parsed file of type pymo.data.MocapData
            bvh_frame_rate.add(parsed_data.framerate)
            assert len(bvh_frame_rate) == 1, f"More than one frame rate present!!! {bvh_frame_rate}"
            if rotations and velocities:
                file_name = 'data/all_synthetic_motions_velocities_effort.csv'
                data_expmaps = _preprocess_pipeline(parsed_data)
                # remove root joint absolute positions
                data_expmaps = data_expmaps[:, 3:]
                data_velocities = _get_standardized_velocities(parsed_data)
                # stack expmap angles for all joints horizontally to data_velocities
                data = np.hstack((data_velocities, data_expmaps))
            elif not rotations and velocities:
                data_expmaps = _preprocess_pipeline(parsed_data)
                data = _get_standardized_velocities(data_expmaps)
            else:
                data_expmaps = _preprocess_pipeline(parsed_data)
                data = _get_standardized_rotations(data_expmaps)
            bvh_counter += 1
            if data.shape[0] < conf.time_series_size:
                assert False, f"Preprocessed file too small- {data.shape[0]} - relative to exemplar size -" \
                              f" {conf.time_series_size}"
            f_rep = np.tile(efforts_list, (data.shape[0], 1))
            # append anim as an additional column
            a_rep = np.tile(anim_ind[str.upper(anim)], (data.shape[0], 1))
            # animation name is fifth column
            file_data = np.concatenate((a_rep, data), axis=1)
            # append efforts (the first 4 column(s) will be the efforts)
            file_data = np.concatenate((f_rep, file_data), axis=1)
            apply_moving_window(singleton_batches, file_data)

    conf.bvh_file_num = bvh_counter
    singleton_batches.store_effort_labels_dict()
    singleton_batches.balance_similarity_classes()
    if conf.bool_fixed_neutral_embedding:
        singleton_batches.pop_similarity_dict_element(key=(0, 0, 0, 0))
    else:
        singleton_batches.move_tuple_to_similarity_dict_front(key=(0, 0, 0, 0))
    singleton_batches.convert_exemplar_np_arrays_to_tensors()
    singleton_batches.store_similarity_labels_exemplars_dict()
    assert singleton_batches.batch_idx == len(
        singleton_batches.dict_efforts_labels.values()) - 1, f"batch_idx: {singleton_batches.batch_idx}, " \
                                                             f"num" \
                                                             f"labels: {len(singleton_batches.dict_efforts_labels.values())}"
    singleton_batches.verify_dict_similarity_exemplars()


def prepare_data(rotations=True, velocities=False):
    """
    Invoke data preprocessing for both efforts and similarity networks.

    Args:
        rotations (bool): Whether to include rotations in the data preprocessing.
        velocities (bool): Whether to include velocities in the data preprocessing.

    Returns:
        None
    """
    prep_all_data_for_training(rotations=rotations, velocities=velocities)


def load_data(rotations=True, velocities=False):
    """
    Load motion data for training if available, otherwise prepare it first.

    Args:
        rotations (bool): Whether to include rotations in the loaded data.
        velocities (bool): Whether to include velocities in the loaded data.

    Returns:
        partition (dict): A dictionary containing the partitioned efforts network data.
        labels_dict (dict): A dictionary containing labels (efforts values).
    """
    csv_file = os.path.join(conf.output_metrics_dir, f'{conf.num_task}_{conf.window_delta}.csv')
    if path.exists(conf.effort_network_exemplars_dir) and not path.exists(csv_file):
        shutil.rmtree(conf.effort_network_exemplars_dir)
        os.makedirs(conf.effort_network_exemplars_dir)
        prepare_data(rotations=rotations, velocities=velocities)
    elif not path.exists(csv_file):
        prepare_data(rotations=rotations, velocities=velocities)
    partition, labels_dict = _partition_effort_ids_and_labels()
    return partition, labels_dict


def _partition_effort_ids_and_labels(train_val_split=0.8):
    """
    Partition effort IDs and labels for training, validation, and testing.

    Args:
        train_val_split (float): The percentage of data to be used for training.

    Returns:
        partition (dict): A dictionary containing the partitioned data.
        labels_dict (dict): A dictionary containing labels for the data.
    """
    with open(conf.effort_network_exemplars_dir + conf.efforts_labels_dict_file_name, 'rb') as handle:
        labels_dict = pickle.load(handle)
    batch_ids_list = list(labels_dict.keys())
    random.shuffle(batch_ids_list)
    train_size = int(train_val_split * len(batch_ids_list))
    test_val_size = int(((1 - train_val_split) * len(batch_ids_list)) / 2)
    partition = {'train': batch_ids_list[:train_size], 'validation': batch_ids_list[train_size:test_val_size],
                 'test': batch_ids_list[-test_val_size:]}
    return partition, labels_dict


def load_similarity_data(train_val_split=0.8):
    """
    Load similarity dict of all class exemplars and split across train, validation, and test sets.

    Args:
        train_val_split: float: percentage of data to be used for training versus validation and
        test sets

    Returns:
        similarity_dict: dict: partitioned similarity dict of all class exemplars
    """
    dict_similarity_classes_exemplars = pickle.load(open(
        conf.similarity_exemplars_dir + conf.similarity_dict_file_name, "rb"))
    singleton_batches.dict_similarity_exemplars = dict_similarity_classes_exemplars
    singleton_batches.verify_dict_similarity_exemplars()
    num_exemplars = len(dict_similarity_classes_exemplars[next(iter(dict_similarity_classes_exemplars.keys()))])
    print(f"Number of total exemplars per class: {num_exemplars}")
    p = np.random.permutation(num_exemplars-1)
    train_size = int(train_val_split * num_exemplars)
    val_and_test_size = int(((1 - train_val_split) * num_exemplars) / 2)
    print(f"train size: {train_size}, val and test size: {val_and_test_size}")

    train_data = {}
    validation_data = {}
    test_data = {}
    for k, v in dict_similarity_classes_exemplars.items():
        train_data[k] = v[:train_size]
        validation_data[k] = v[train_size:train_size + val_and_test_size]
        test_data[k] = v[-val_and_test_size:]

    print(f"len dict_train_class_value: {len(train_data[(0, -1, -1, -1)])}")
    print(f"dict_train_class_value element shape: {train_data[(0, -1, -1, -1)][0].shape}")
    print(f"dict_train_class_value_element_type: {type(train_data[(0, 1, 1, 0)][0])}")

    print(f"len dict_val_class_value: {len(validation_data[(0, -1, -1, -1)])}")
    print(f"dict_val_class_value element shape: {validation_data[(0, -1, -1, -1)][0].shape}")
    print(f"dict_val_class_value_element_type: {type(validation_data[(0, 1, 1, 0)][0])}")

    print(f"len dict_test_class_value: {len(test_data[(0, -1, -1, -1)])}")
    print(f"dict_test_class_value element shape: {test_data[(0, -1, -1, -1)][0].shape}")
    print(f"dict_test_class_value_element_type: {type(test_data[(0, 1, 1, 0)][0])}")

    return {
        'train': train_data,
        'validation': validation_data,
        'test': test_data
    }
