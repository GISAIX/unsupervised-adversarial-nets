import tensorflow as tf


def discriminator_entropy(prob, label):
    # input shape should be [batch, 2]
    # generative: 0, real: 1
    softmax = tf.nn.softmax(logits=prob)
    ground_truth = tf.one_hot(indices=label, depth=2)
    loss = 0
    for i in range(2):
        loss -= tf.reduce_mean(
            ground_truth[:, i] * tf.log(tf.clip_by_value(softmax[:, i], 0.005, 1)))
    return loss


def generator_entropy(prob, label):
    # only input generative: 0
    softmax = tf.nn.softmax(logits=prob)
    ground_truth = tf.one_hot(indices=label, depth=2)
    return - 2 * tf.reduce_mean(
        ground_truth[:, 0] * tf.log(tf.clip_by_value(softmax[:, 1], 0.005, 1)))


def reconstruction_error(generative, ground_truth):
    return tf.reduce_mean(
        (generative - ground_truth) * (generative - ground_truth))


def compute_gradient(images):
    # input shape: NDHWC
    pixel_diff_d = images[:, 1:, :, :, :] - images[:, :-1, :, :, :]
    pixel_diff_h = images[:, :, 1:, :, :] - images[:, :, :-1, :, :]
    pixel_diff_w = images[:, :, :, 1:, :] - images[:, :, :, :-1, :]
    gradient_d = tf.abs(pixel_diff_d)
    gradient_h = tf.abs(pixel_diff_h)
    gradient_w = tf.abs(pixel_diff_w)
    return gradient_d, gradient_h, gradient_w


def gradient_difference(generative, ground_truth):
    gradient_gen_x, gradient_gen_y, gradient_gen_z = compute_gradient(generative)
    gradient_label_x, gradient_label_y, gradient_label_z = compute_gradient(ground_truth)
    loss = tf.reduce_mean((gradient_gen_x - gradient_label_x) * (gradient_gen_x - gradient_label_x)) + tf.reduce_mean(
        (gradient_gen_y - gradient_label_y) * (gradient_gen_y - gradient_label_y)) + tf.reduce_mean(
        (gradient_gen_z - gradient_label_z) * (gradient_gen_z - gradient_label_z))
    return loss
