import networks.triplet_mining as triplet_mining
from conf import BATCH_STRATEGY
from conf import BatchStrategy
import tensorflow as tf


# consider alternative loss function based on cosine similarity:
# self.loss = tf.keras.losses.CosineSimilarity(axis=1)
# ap_distance = self.loss(anchor, positive)
# an_distance = self.loss(anchor, negative)
# loss = tf.maximum(ap_distance - an_distance + self.margin, 0.0)
def calculate_triplet_loss(y_true, y_pred, triplet_mining_instance, batch_strategy=BATCH_STRATEGY):
    """Calculate 1D tensor of either squared L2 norm or L2 norm of differences between class embeddings and the
            neutral embedding of shape (embedding_size,), resulting in tensor of shape (batch_size,).

                    Args:
                        y_true: supposed 'labels' of the batch (i.e., class indexes), tensor of size (batch_size,
                        ) where each element is singleton tensor of an integer in the range [1, 56]
                        y_pred: embeddings, tensor of shape (batch_size, embed_dim)
                        batch_strategy: Enum value indicating the batch strategy to use for triplet mining

                    Returns:
                        losses: tensor of shape (triplet_mining.num_states_drives, triplet_mining.num_states_drives)
            """
    # must flatten y_true to tensor of ints in order to sort it
    y_true_flat = tf.reshape(y_true, [-1])
    sorted_indices = tf.argsort(y_true_flat)
    y_true = tf.gather(y_true, sorted_indices)
    y_pred = tf.gather(y_pred, sorted_indices)
    classes_distances = triplet_mining_instance.calculate_left_right_distances(y_pred)
    triplet_mining_instance.calculate_class_neut_distances(y_pred)

    # Reshape the 1D tensor into a 2D tensor with shape (1, 56) (will allow for broadcasting)
    row_dists_class_neut = tf.reshape(triplet_mining_instance.tensor_dists_class_neut,
                                      (1, triplet_mining_instance.num_states_drives))
    # Reshape the 1D tensor into a 2D tensor of shape (56, 1) (will allow for broadcasting)
    column_dists_class_neut = tf.reshape(triplet_mining_instance.tensor_dists_class_neut,
                                         (triplet_mining_instance.num_states_drives, 1))

    # case 1: left and right are positives

    # L = anchor
    diff_lr_ln = classes_distances - column_dists_class_neut

    tf.debugging.assert_shapes(
        [(tf.shape(diff_lr_ln), (tf.TensorShape([triplet_mining_instance.num_states_drives,
                                                 triplet_mining_instance.num_states_drives]),))],
        message="Tensor diff_lr_ln has incorrect shape")
    # consider only cases where diff_lr_ln > 0 (i.e., l_n - l_r > 0)
    if batch_strategy == BatchStrategy.HARD:
        diff_lr_ln = tf.maximum(diff_lr_ln, 0.0)
        diff_lr_rl_alpha = diff_lr_ln + triplet_mining_instance.matrix_alpha_left_right_right_left
        diff_lr_alpha = tf.multiply(diff_lr_rl_alpha, triplet_mining_instance.matrix_bool_left_right)
        # ensure no negative losses
        triplet_loss_L_R = diff_lr_alpha
        tf.debugging.assert_equal(triplet_loss_L_R, tf.maximum(triplet_loss_L_R, 0.0), message="Negative losses exist")
    # consider only cases where diff_lr_ln < 0 yet triplet_loss_L_N > 0 (i.e., l_r - l_n < 0 and l_r - l_n + alpha > 0)
    elif batch_strategy == BatchStrategy.SEMI_HARD:
        diff_lr_ln = tf.where(diff_lr_ln < 0, diff_lr_ln, 0)
        diff_lr_rl_alpha = diff_lr_ln + triplet_mining_instance.matrix_alpha_left_right_right_left
        diff_lr_alpha = tf.multiply(diff_lr_rl_alpha, triplet_mining_instance.matrix_bool_left_right)
        triplet_loss_L_R = tf.maximum(diff_lr_alpha, 0.0)
    # consider all cases where diff_lr_ln + alpha > 0 (i.e., dist(l_r) - dist(l_n) + alpha > 0)
    elif batch_strategy == BatchStrategy.ALL:
        diff_lr_rl_alpha = diff_lr_ln + triplet_mining_instance.matrix_alpha_left_right_right_left
        diff_lr_alpha = tf.multiply(diff_lr_rl_alpha, triplet_mining_instance.matrix_bool_left_right)
        triplet_loss_L_R = tf.maximum(diff_lr_alpha, 0.0)

    # R = anchor
    diff_rl_rn = classes_distances - row_dists_class_neut
    if batch_strategy == BatchStrategy.HARD:
        diff_rl_rn = tf.maximum(diff_rl_rn, 0.0)
        diff_rl_rn_alpha = diff_rl_rn + triplet_mining_instance.matrix_alpha_left_right_right_left
        diff_rl_alpha = tf.multiply(diff_rl_rn_alpha, triplet_mining_instance.matrix_bool_right_left)
        triplet_loss_R_L = diff_rl_alpha
        tf.debugging.assert_equal(triplet_loss_R_L, tf.maximum(triplet_loss_R_L, 0.0), message="Negative losses exist")
    elif batch_strategy == BatchStrategy.SEMI_HARD:
        diff_rl_rn = tf.where(diff_rl_rn < 0, diff_rl_rn, 0)
        diff_rl_rn_alpha = diff_rl_rn + triplet_mining_instance.matrix_alpha_left_right_right_left
        diff_rl_alpha = tf.multiply(diff_rl_rn_alpha, triplet_mining_instance.matrix_bool_right_left)
        triplet_loss_R_L = tf.maximum(diff_rl_alpha, 0.0)
    elif batch_strategy == BatchStrategy.ALL:
        diff_rl_rn_alpha = diff_rl_rn + triplet_mining_instance.matrix_alpha_left_right_right_left
        diff_rl_alpha = tf.multiply(diff_rl_rn_alpha, triplet_mining_instance.matrix_bool_right_left)
        triplet_loss_R_L = tf.maximum(diff_rl_alpha, 0.0)
    # case 2: left and neutral are positives

    # L = anchor
    diff_ln_lr = column_dists_class_neut - classes_distances
    tf.debugging.assert_shapes([(tf.shape(diff_ln_lr), (tf.TensorShape([triplet_mining_instance.num_states_drives,
                                                                        triplet_mining_instance.num_states_drives]),))],
                               message="Tensor diff_ln_lr has incorrect shape")
    if batch_strategy == BatchStrategy.HARD:
        diff_ln_lr = tf.where(diff_ln_lr > 0, diff_ln_lr, 0)
        diff_ln_lr_alpha = diff_ln_lr + triplet_mining_instance.matrix_alpha_left_neut_neut_left
        # we care only for differences with corresponding alpha_left_neut values (i.e., relevant l_n cases)
        diff_ln_lr_alpha = tf.multiply(diff_ln_lr_alpha, triplet_mining_instance.matrix_bool_left_neut)
        # ensure no negative losses
        triplet_loss_L_N = diff_ln_lr_alpha
        tf.debugging.assert_equal(triplet_loss_L_N, tf.maximum(triplet_loss_L_N, 0.0), message="Negative losses exist")
    elif batch_strategy == BatchStrategy.SEMI_HARD:
        diff_ln_lr = tf.where(diff_ln_lr < 0, diff_ln_lr, 0)
        diff_ln_lr_alpha = diff_ln_lr + triplet_mining_instance.matrix_alpha_left_neut_neut_left
        diff_ln_lr_alpha = tf.multiply(diff_ln_lr_alpha, triplet_mining_instance.matrix_bool_left_neut)
        triplet_loss_L_N = tf.where(diff_ln_lr_alpha > 0, diff_ln_lr_alpha, 0)
    elif batch_strategy == BatchStrategy.ALL:
        diff_ln_lr_alpha = diff_ln_lr + triplet_mining_instance.matrix_alpha_right_neut_neut_right
        diff_ln_lr_alpha = tf.multiply(diff_ln_lr_alpha, triplet_mining_instance.matrix_bool_left_neut)
        triplet_loss_L_N = tf.where(diff_ln_lr_alpha > 0, diff_ln_lr_alpha, 0)
    # N = anchor
    # diff_nl_nr = column_dists_class_neut - row_dists_class_neut
    diff_nl_nr = row_dists_class_neut - column_dists_class_neut
    tf.debugging.assert_shapes([(tf.shape(diff_nl_nr), (tf.TensorShape([triplet_mining_instance.num_states_drives,
                                                                        triplet_mining_instance.num_states_drives]),))],
                               message="Tensor diff_nl_nr has incorrect shape")
    if batch_strategy == BatchStrategy.HARD:
        diff_nl_nr = tf.where(diff_nl_nr > 0, diff_nl_nr, 0)
        diff_nl_nr_alpha = diff_nl_nr + triplet_mining_instance.matrix_alpha_left_neut_neut_left
        diff_nl_nr_alpha = tf.multiply(diff_nl_nr_alpha, triplet_mining_instance.matrix_bool_neut_left)
        triplet_loss_N_L = diff_nl_nr_alpha
        tf.debugging.assert_equal(triplet_loss_N_L, tf.maximum(triplet_loss_N_L, 0.0), message="Negative losses exist")
    elif batch_strategy == BatchStrategy.SEMI_HARD:
        diff_nl_nr = tf.where(diff_nl_nr < 0, diff_nl_nr, 0)
        diff_nl_nr_alpha = diff_nl_nr + triplet_mining_instance.matrix_alpha_left_neut_neut_left
        diff_nl_nr_alpha = tf.multiply(diff_nl_nr_alpha, triplet_mining_instance.matrix_bool_neut_left)
        triplet_loss_N_L = tf.where(diff_nl_nr_alpha > 0, diff_nl_nr_alpha, 0)
    elif batch_strategy == BatchStrategy.ALL:
        diff_nl_nr_alpha = diff_nl_nr + triplet_mining_instance.matrix_alpha_left_neut_neut_left
        diff_nl_nr_alpha = tf.multiply(diff_nl_nr_alpha, triplet_mining_instance.matrix_bool_neut_left)
        triplet_loss_N_L = tf.where(diff_nl_nr_alpha > 0, diff_nl_nr_alpha, 0)

    ### case 3: right and neutral are positives

    # R = anchor
    diff_rn_rl = row_dists_class_neut - tf.transpose(classes_distances)
    if batch_strategy == BatchStrategy.HARD:
        diff_rn_rl = tf.where(diff_rn_rl > 0, diff_rn_rl, 0)
        diff_r_n_r_l_alpha = diff_rn_rl + triplet_mining_instance.matrix_alpha_right_neut_neut_right
        diff_r_n_r_l_alpha = tf.multiply(diff_r_n_r_l_alpha, triplet_mining_instance.matrix_bool_right_neut)
        # remove negative losses
        triplet_loss_R_N = diff_r_n_r_l_alpha
        tf.debugging.assert_equal(triplet_loss_R_N, tf.maximum(triplet_loss_R_N, 0.0), message="Negative losses exist")
    elif batch_strategy == BatchStrategy.SEMI_HARD:
        diff_rn_rl = tf.where(diff_rn_rl < 0, diff_rn_rl, 0)
        diff_r_n_r_l_alpha = diff_rn_rl + triplet_mining_instance.matrix_alpha_right_neut_neut_right
        diff_r_n_r_l_alpha = tf.multiply(diff_r_n_r_l_alpha, triplet_mining_instance.matrix_bool_right_neut)
        triplet_loss_R_N = tf.where(diff_r_n_r_l_alpha > 0, diff_r_n_r_l_alpha, 0)
    elif batch_strategy == BatchStrategy.ALL:
        diff_r_n_r_l_alpha = diff_rn_rl + triplet_mining_instance.matrix_alpha_right_neut_neut_right
        diff_r_n_r_l_alpha = tf.multiply(diff_r_n_r_l_alpha, triplet_mining_instance.matrix_bool_right_neut)
        triplet_loss_R_N = tf.where(diff_r_n_r_l_alpha > 0, diff_r_n_r_l_alpha, 0)
    # check if diff_r_n_r_l_alpha and diff_rn_rl contain non-zero values in the same locations
    # condition = tf.math.logical_and(tf.math.not_equal(diff_rn_rl, 0), tf.math.not_equal(diff_r_n_r_l_alpha, 0))
    # tf.debugging.Assert(tf.math.reduce_all(condition),
    #                     message="Difference rn_rl tensors do not have non-zero elements in the same cells")
    tf.debugging.assert_equal(triplet_loss_R_N,
                              tf.multiply(triplet_loss_R_N, triplet_mining_instance.matrix_bool_right_neut),
                              message="Improper triplet_loss_R_N generation")
    # N = anchor
    diff_nr_nl = column_dists_class_neut - row_dists_class_neut
    tf.debugging.assert_shapes(
        [(diff_nr_nl, (triplet_mining_instance.num_states_drives, triplet_mining_instance.num_states_drives))],
        message="Tensor diff_nr_nl has incorrect shape")
    if batch_strategy == BatchStrategy.HARD:
        diff_nr_nl = tf.where(diff_nr_nl > 0, diff_nr_nl, 0)
        diff_nr_nl_alpha = diff_nr_nl + triplet_mining_instance.matrix_alpha_right_neut_neut_right
        diff_nr_nl_alpha = tf.multiply(diff_nr_nl_alpha, triplet_mining_instance.matrix_bool_neut_right)
        triplet_loss_N_R = diff_nr_nl_alpha
        tf.debugging.assert_equal(triplet_loss_N_R, tf.maximum(triplet_loss_N_R, 0.0), message="Negative losses exist")
    elif batch_strategy == BatchStrategy.SEMI_HARD:
        diff_nr_nl_alpha = diff_nr_nl + triplet_mining_instance.matrix_alpha_right_neut_neut_right
        diff_nr_nl_alpha = tf.multiply(diff_nr_nl_alpha, triplet_mining_instance.matrix_bool_neut_right)
        triplet_loss_N_R = tf.where(diff_nr_nl_alpha > 0, diff_nr_nl_alpha, 0)
    elif batch_strategy == BatchStrategy.ALL:
        diff_nr_nl_alpha = diff_nr_nl + triplet_mining_instance.matrix_alpha_right_neut_neut_right
        diff_nr_nl_alpha = tf.multiply(diff_nr_nl_alpha, triplet_mining_instance.matrix_bool_neut_right)
        triplet_loss_N_R = tf.where(diff_nr_nl_alpha > 0, diff_nr_nl_alpha, 0)

    losses = (triplet_loss_L_R + triplet_loss_R_L + triplet_loss_L_N + triplet_loss_N_L + triplet_loss_R_N +
              triplet_loss_N_R)
    # assertion check that no negative losses exist
    tf.debugging.assert_equal(losses, tf.maximum(losses, 0.0), message="Negative losses exist")
    tf.debugging.assert_shapes([(tf.shape(losses), (tf.TensorShape([triplet_mining_instance.num_states_drives,
                                                                    triplet_mining_instance.num_states_drives]),))],
                               message="Comparisons loss tensor has incorrect shape")
    return losses

def batch_triplet_loss(y_true, y_pred):
    """Build triplet loss over a batch of embeddings.

       custom loss function——a wrapper for calculate_triplet_loss, passed to Keras' compile method;
        computes a(n) (aggregated) triplet loss for a batch of embeddings

        We calculate triplet losses for all anchor-positive possibilities, and mask for semi-hard cases only.

        Args:
            y_true: supposed 'labels' of the batch (i.e., class indexes), tensor of size (batch_size,)
            y_pred: embeddings, tensor of shape (batch_size, embed_dim)

        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """

    triplet_loss = calculate_triplet_loss(y_true, y_pred, triplet_mining)
    # Count number of positive err triplets (where triplet_loss > 0)
    valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), float)
    num_positive_triplets = tf.reduce_sum(valid_triplets)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)
    tf.debugging.assert_scalar(triplet_loss, message="Epoch loss is not a scalar")

    return triplet_loss

# possible eval metrics: counting the triplets for which the positive distance (anchor - positive) is less than
# the negative distance (anchor - negative) (by at least the margin) and then dividing by the total number of
# triplets in the batch (i.e., proportion of zero loss triplets)

# classification: model embeddings to classify motions (take an unseen motion exemplar (representing one class) and
# compare it - using L2 normalized Euclidean distance - with its nearest neighbor.

# ranking accuracy:
# compute L2 normalized distances between all pairs of classes in the embedding space, and generate Spearman
# rank correlation coefficient (SROCC) between the sorted distances (ascending order) and the complement of the
# user normalized similarity scores (i.e., 1 - normalized similarity scores).
