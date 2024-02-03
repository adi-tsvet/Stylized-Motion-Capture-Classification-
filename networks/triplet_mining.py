"""
static module for organizing triplet mining data as well as performing online triplet mining
"""
from pathlib import Path
import conf
import tensorflow as tf
import numpy as np
import pandas as pd
import ast
import pickle

# Define static state variables
dict_similarity_classes_exemplars = {}
matrix_alpha_left_right_right_left = None
matrix_alpha_left_neut_neut_left = None
matrix_alpha_right_neut_neut_right = None
matrix_bool_left_right = None
matrix_bool_right_left = None
matrix_bool_left_neut = None
matrix_bool_neut_left = None
matrix_bool_right_neut = None
matrix_bool_neut_right = None

num_states_drives = 0
tensor_dists_left_right_right_left = None
tensor_dists_class_neut = None
neutral_embedding = None


def initialize_triplet_mining():
    """
    Initialize the triplet mining module's state variables.

    This function loads necessary data, sets up state variables, and performs preprocessing.

    Args:
        None

    Returns:
        None
    """
    global dict_similarity_classes_exemplars
    global matrix_alpha_left_right_right_left
    global matrix_alpha_left_neut_neut_left
    global matrix_alpha_right_neut_neut_right
    global matrix_bool_left_right
    global matrix_bool_right_left
    global matrix_bool_left_neut
    global matrix_bool_neut_left
    global matrix_bool_right_neut
    global matrix_bool_neut_right
    global num_states_drives
    global tensor_dists_left_right_right_left
    global tensor_dists_class_neut
    global neutral_embedding

    print("Initializing Triplet Mining module state variables")
    dict_similarity_classes_exemplars = pickle.load(open(
        conf.similarity_exemplars_dir + conf.similarity_dict_file_name, "rb"))
    print(f"classes: {dict_similarity_classes_exemplars.keys()}")
    num_states_drives = len(dict_similarity_classes_exemplars.keys())
    print(f"triplet_mining:init: loaded: {num_states_drives} states + drives")
    (matrix_alpha_left_right_right_left, matrix_alpha_left_neut_neut_left, matrix_alpha_right_neut_neut_right,
     matrix_bool_left_right, matrix_bool_right_left, matrix_bool_left_neut, matrix_bool_neut_left,
     matrix_bool_right_neut, matrix_bool_neut_right) = [tf.Variable(
        initial_value=tf.zeros((num_states_drives, num_states_drives), dtype=tf.float32)) for _ in range(9)]
    tensor_dists_class_neut = tf.Variable(initial_value=tf.zeros((num_states_drives,)))
    neutral_embedding = tf.Variable(initial_value=tf.zeros((conf.embedding_size,)))
    subset_global_dict()
    pre_process_comparisons_data()


def subset_global_dict():
    """
    Subsets the global dictionary of similarity classes based on data in the comparisons DataFrame.

    Args:
        None

    Returns:
        None
    """
    global dict_similarity_classes_exemplars
    dict_label_to_id = {class_label: idx for idx, class_label in
                        enumerate(dict_similarity_classes_exemplars.keys())}
    print(dict_label_to_id)


def extract_neutral_embedding(embeddings):
    """
    Assigns the neutral embedding from the network output, shape (embedding_size,), to instance attribute

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)

    Returns:
        embeddings: tensor of shape (batch_size - 1, embed_dim)
    """
    neutral_embedding.assign(embeddings[0])
    return embeddings[1:]


def calculate_left_right_distances(embeddings, squared=True):
    """Compute the 2D matrix of distances between all 56 class embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """

    if not conf.bool_fixed_neutral_embedding:
        embeddings = extract_neutral_embedding(embeddings)

    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding (each embedding's dot product with itself
    # shape (batch_size,)
    square_norm = tf.linalg.diag_part(dot_product)

    # Compute the pairwise squared euclidean distance matrix:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative, so we put everything >= 0.0 (see unit test)
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # The gradient of sqrt is infinite when x == 0.0 (eg: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.cast(tf.equal(distances, 0.0), float)
        distances = distances + mask * 1e-16

        distances = tf.clip_by_value(distances, 1e-16, tf.reduce_max(distances))

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances of the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

        tf.debugging.check_numerics(distances, "NaN or Inf values found in distances")

        tf.debugging.assert_shapes([(tf.shape(distances), (tf.TensorShape([conf.similarity_batch_size,
                                                                           conf.similarity_batch_size]),))])

    # if not conf.bool_fixed_neutral_embedding:
    #     tf.debugging.assert_shapes([(tf.shape(distances), (tf.TensorShape([conf.similarity_batch_size - 1,
    #                                                                        conf.similarity_batch_size - 1]),))])
    # else:
    #     tf.debugging.assert_shapes([(tf.shape(distances), (tf.TensorShape([conf.similarity_batch_size,
    #                                                                        conf.similarity_batch_size]),))])
    return distances


def calculate_class_neut_distances(embeddings, squared=False):
    """
    Calculate 1D tensor of either squared L2 norm, or L2 norm, of differences between class embeddings and the
    neutral embedding.

    Args:
        embeddings: Tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, calculate squared L2 norm; if false, calculate L2 norm.

    Returns:
        None
    """
    if squared:
        # Compute squared Euclidean distance between each class embedding and the neutral embedding
        # tensor_dists_class_neut.assign(tf.reduce_sum(tf.square(embeddings - neutral_embedding), axis=1))
        tensor_dists_class_neut.assign(tf.square(tf.norm(embeddings - neutral_embedding, ord="euclidean", axis=1)))
    else:
        # Compute Euclidean distance between each class embedding and the neutral embedding
        tensor_dists_class_neut.assign(tf.norm(embeddings - neutral_embedding, ord="euclidean", axis=1))


def pre_process_comparisons_data():
    """
    Preprocess user comparison data and populate alpha matrices and masks based on the data.

    Args:
        None

    Returns:
        None
    """
    global dict_similarity_classes_exemplars

    def verify_comparison_data():
        # Check that the Dataframe has no repeated efforts_tuples values
        seen_tuples = set()
        for index, row in df_comparisons.iterrows():
            hashable_list = tuple(row['efforts_tuples'])
            # Check if the hashable list is already in the set
            if hashable_list in seen_tuples:
                assert False, f"Duplicate df_comparison efforts_tuples row at index {index}: {row['efforts_tuples']}"
            seen_tuples.add(hashable_list)
            # Check if efforts_left equals efforts_right
            assert row['efforts_tuples'][0] != row['efforts_tuples'][1], (f"efforts_left equals efforts_right at "
                                                                          f"index: {index}: "
                                                                          f"{row['efforts_tuples']}") \
 \
                # Check that the Dataframe does not include the neutral as a similarity class
        assert (0, 0, 0, 0) not in seen_tuples, "neutral similarity class included in comparisons data"
        # Check that the Dataframe has the correct number of similarity classes
        assert len(seen_tuples) == num_states_drives, (f"incomplete similarity class count in comparisons "
                                                            f"data: {len(seen_tuples)}")

    def _generate_df_alphas():
        """
        generate alpha_dataframes where each row is a comparison between two similarity classes (and the neutral) and
        contains the corresponding two out of six alpha values (each comparison has two alpha values, one for each
        of the positives.

       Args:
           None

       Returns:
           None
        """
        comparisons_list = []
        selection_values = [0, 1, 2]
        # Initialize new columns for per-comparison alpha values (two values created per comparison)
        df_comparisons['alpha_0_2'] = 0
        df_comparisons['alpha_2_0'] = 0
        df_comparisons['alpha_0_1'] = 0
        df_comparisons['alpha_2_1'] = 0
        df_comparisons['alpha_1_0'] = 0
        df_comparisons['alpha_1_2'] = 0
        # Iterate over three consecutive rows
        # selected_0 is either 0 or 1 and selected_1 is either 1 or 2 (else we terminate)
        for i in range(0, len(df_comparisons), 3):
            group = df_comparisons.iloc[i:i + 3]

            # Find the row within a triplet with the maximum 'count_normalized' value, thereby establishing the
            # positive pair (i.e., selected0 and selected1
            max_row = group.loc[group['count_normalized'].idxmax()]

            max_selected_0, max_selected_1 = max_row['selected0'], max_row['selected1']
            negative_index = next(x for x in selection_values if x != max_selected_0 and x != max_selected_1)
            # find the anchor_positive ratio value under the cases in which anchor is each of the positive pair,
            # respectively, and positive is the negative class
            if max_selected_0 == 0:
                ratio_positive_1_negative = \
                    group.loc[(group['selected0'] == max_selected_0) & (group['selected1'] ==
                                                                        negative_index)].iloc[0][
                        'count_normalized']
                # means max_row selected_0 is 0. Thusly, 2 is the negative index; grab it from the corresponding row
                if max_selected_1 == 1:
                    ratio_positive_2_negative = \
                        group.loc[(group['selected0'] == max_selected_1) & (group['selected1'] ==
                                                                            negative_index)].iloc[
                            0][
                            'count_normalized']
                elif max_selected_1 == 2:
                    ratio_positive_2_negative = \
                        group.loc[(group['selected0'] == negative_index) & (group['selected1'] ==
                                                                            max_selected_1)].iloc[
                            0][
                            'count_normalized']

                else:
                    assert False, "selected1 is not 1 or 2"

            elif max_selected_0 == 1:
                if max_selected_1 == 2:
                    ratio_positive_1_negative = \
                        group.loc[(group['selected0'] == negative_index) & (group['selected1'] ==
                                                                            max_selected_0)].iloc[
                            0][
                            'count_normalized']
                    ratio_positive_2_negative = \
                        group.loc[(group['selected0'] == negative_index) & (group['selected1'] ==
                                                                            max_selected_1)].iloc[
                            0][
                            'count_normalized']
                else:
                    assert False, "selected1 is not 1 or 2"
            else:
                assert False, "selected0 is not 0 or 1"
            # generate the two possible alpha values for a comparison (i.e., treating selected0 as anchor versus
            # treating selected1 as anchor)
            alpha_positive_1_positive_2 = max_row['count_normalized'] - ratio_positive_1_negative
            alpha_positive_2_positive_1 = max_row['count_normalized'] - ratio_positive_2_negative
            if max_row['efforts_tuples'] == '[-1,-1,-1,0]_[0,-1,-1,1]':
                print(f"{alpha_positive_1_positive_2} . {alpha_positive_2_positive_1}")

            # Concatenate selected0 and selected1 to pattern match the alpha anchor_positive column
            alpha_selected_0_selected_1_column = f"alpha_{max_selected_0}_{max_selected_1}"

            alpha_selected_1_selected_0_column = f"alpha_{max_selected_1}_{max_selected_0}"

            df_comparisons.loc[max_row.name, alpha_selected_0_selected_1_column] = alpha_positive_1_positive_2
            df_comparisons.loc[max_row.name, alpha_selected_1_selected_0_column] = alpha_positive_2_positive_1
            comparisons_list.append(df_comparisons.loc[max_row.name])

        alpha_dataframes = pd.DataFrame(comparisons_list)
        alpha_dataframes.reset_index(drop=True, inplace=True)
        # print(f'{alpha_dataframes=}')
        alpha_dataframes = alpha_dataframes[:10]
        return alpha_dataframes

    def _populate_alpha_matrices_and_masks():
        """
        Populate the alpha matrices and masks based on the data in the comparisons DataFrame.

        Args:
           None

       Returns:
           None
        """
        # Iterate over the rows of the comparisons DataFrame
        repeat_class_comparison_counter = 0
        equal_comparison_counter = 0
        bool_swap_left_right = False
        for index, row in df_alphas.iterrows():
            efforts_tuple = row['efforts_tuples']
            # enforce constraint that i < j always corresponds to left, right / i > j to right, left
            if dict_label_to_id[efforts_tuple[0]] > dict_label_to_id[efforts_tuple[1]]:
                row['efforts_tuples'] = [efforts_tuple[1], efforts_tuple[0]]
                bool_swap_left_right = True
            efforts_left = row['efforts_tuples'][0]
            efforts_right = row['efforts_tuples'][1]
            index_left = dict_label_to_id[efforts_left]
            index_right = dict_label_to_id[efforts_right]

            ### temporary fix for erroneous similarity class, and for self to self comparison, in comparisons
            ### DataFrame
            if efforts_left == (0, 0, 0, 0) or efforts_left == efforts_right:
                repeat_class_comparison_counter += 1
                continue
            # for any comparison, left_index < right_index
            # left, right indices indicate location for left, neut anchor_positive alpha with respect to Left,
            # Neutral Matrix (and right, neut anchor_positive alpha with respect to Right, Neutral Matrix)
            # whereas right, left indices indicate location for neut, left anchor_positive alpha and neut,
            # right anchor_positive alpha, respectively.
            if row['alpha_0_2'] != 0:
                print("entered alpha_0_2")
                left_right_alpha = row['alpha_0_2']
                right_left_alpha = row['alpha_2_0']
                if bool_swap_left_right:
                    left_right_alpha = row['alpha_2_0']
                    right_left_alpha = row['alpha_0_2']
                matrix_alpha_left_right_right_left.assign(tf.tensor_scatter_nd_add(
                    matrix_alpha_left_right_right_left,
                    indices=tf.constant([[index_left, index_right]]),
                    updates=tf.constant([left_right_alpha], dtype=tf.float32)
                ))
                matrix_bool_left_right.assign(tf.tensor_scatter_nd_update(
                    matrix_bool_left_right,
                    indices=tf.constant([[index_left, index_right]]),
                    updates=tf.constant([1.0], dtype=tf.float32)
                ))
                matrix_alpha_left_right_right_left.assign(tf.tensor_scatter_nd_add(
                    matrix_alpha_left_right_right_left,
                    indices=tf.constant([[index_right, index_left]]),
                    updates=tf.constant([right_left_alpha], dtype=tf.float32)
                ))
                matrix_bool_right_left.assign(tf.tensor_scatter_nd_update(
                    matrix_bool_right_left,
                    indices=tf.constant([[index_right, index_left]]),
                    updates=tf.constant([1.0], dtype=tf.float32)
                ))
            elif row['alpha_0_1'] != 0:
                print("entered alpha_0_1")
                left_neutral_alpha = row['alpha_0_1']
                neutral_left_alpha = row['alpha_1_0']
                if bool_swap_left_right:
                    left_neutral_alpha = row['alpha_2_1']
                    neutral_left_alpha = row['alpha_1_2']
                matrix_alpha_left_neut_neut_left.assign(tf.tensor_scatter_nd_add(
                    matrix_alpha_left_neut_neut_left,
                    indices=tf.constant([[index_left, index_right]]),
                    updates=tf.constant([left_neutral_alpha], dtype=tf.float32)
                ))
                matrix_bool_left_neut.assign(tf.tensor_scatter_nd_update(
                    matrix_bool_left_neut,
                    indices=tf.constant([[index_left, index_right]]),
                    updates=tf.constant([1.0], dtype=tf.float32)
                ))
                matrix_alpha_left_neut_neut_left.assign(tf.tensor_scatter_nd_add(
                    matrix_alpha_left_neut_neut_left,
                    indices=tf.constant([[index_right, index_left]]),
                    updates=tf.constant([neutral_left_alpha], dtype=tf.float32)
                ))
                matrix_bool_neut_left.assign(tf.tensor_scatter_nd_update(
                    matrix_bool_neut_left,
                    indices=tf.constant([[index_right, index_left]]),
                    updates=tf.constant([1.0], dtype=tf.float32)
                ))
            elif row['alpha_2_1'] != 0:
                print(f"entered alpha_2_1 at indices: {index_left} , {index_right}")
                right_neutral_alpha = row['alpha_2_1']
                neutral_right_alpha = row['alpha_1_2']
                if bool_swap_left_right:
                    right_neutral_alpha = row['alpha_0_1']
                    neutral_right_alpha = row['alpha_1_0']
                matrix_alpha_right_neut_neut_right.assign(tf.tensor_scatter_nd_add(
                    matrix_alpha_right_neut_neut_right,
                    indices=tf.constant([[index_left, index_right]]),
                    updates=tf.constant([right_neutral_alpha], dtype=tf.float32)
                ))
                matrix_bool_right_neut.assign(tf.tensor_scatter_nd_update(
                    matrix_bool_right_neut,
                    indices=tf.constant([[index_left, index_right]]),
                    updates=tf.constant([1.0], dtype=tf.float32)
                ))
                matrix_alpha_right_neut_neut_right.assign(tf.tensor_scatter_nd_add(
                    matrix_alpha_right_neut_neut_right,
                    indices=tf.constant([[index_right, index_left]]),
                    updates=tf.constant([neutral_right_alpha], dtype=tf.float32)
                ))
                matrix_bool_neut_right.assign(tf.tensor_scatter_nd_update(
                    matrix_bool_neut_right,
                    indices=tf.constant([[index_right, index_left]]),
                    updates=tf.constant([1.0], dtype=tf.float32)
                ))
            # implies equal selection (or no selection) across all three pairs of a triplet
            else:
                equal_comparison_counter += 1
            bool_swap_left_right = False

    # read_in R generated similarity comparisons ratios csv
    aux_folder_path = (Path(__file__) / '../../aux').resolve()
    csv_similarity_ratios_path = aux_folder_path / 'similarity_comparisons_ratios_11_17_23.csv'
    df_comparisons = pd.read_csv(csv_similarity_ratios_path)
    # Split the efforts_tuples values at the delimiter '_' and convert tokens to tuples
    df_comparisons['efforts_tuples'] = df_comparisons['efforts_tuples'].apply(
        lambda x: [tuple(ast.literal_eval(token)) for token in x.split('_')])
    ### (Temporary) Remove similarity classes from the dictionary that are not present in the comparisons DataFrame
    ### (e.g., (-1, -1, -1, 0), etc)
    set_comparison_classes = set([efforts_tuple[0] for efforts_tuple in df_comparisons['efforts_tuples']]).union(
        set([efforts_tuple[1] for efforts_tuple in df_comparisons['efforts_tuples']]))
    print(f"len set: {len(set_comparison_classes)}")
    dict_similarity_classes_exemplars = {key: value for key, value in
                                         dict_similarity_classes_exemplars.items() if
                                         key in
                                         set_comparison_classes}
    # Create a dictionary mapping the n similarity class labels to integers
    dict_label_to_id = {class_label: idx for idx, class_label in
                        enumerate(dict_similarity_classes_exemplars.keys())}
    dict_id_to_label = {idx: class_label for idx, class_label in
                        enumerate(dict_similarity_classes_exemplars.keys())}
    print(f"reduced dict label to id len: {len(dict_label_to_id)}")
    print(f"k,v of dict id to label: {dict_id_to_label.items()}")
    # verify_comparison_data()
    df_alphas = _generate_df_alphas()
    _populate_alpha_matrices_and_masks()
    print(f"type of matrix_bool_left_right: {type(matrix_bool_left_right)}")
