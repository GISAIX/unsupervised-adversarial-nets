import tensorflow as tf


def discriminator_entropy(prediction, label):
    # input shape should be (batch, 1)
    # generative: 0, real: 1
    # Todo: check
    return - tf.reduce_mean(
        label * tf.log(prediction) + (1 - label) * tf.log(1 - prediction))


def generator_entropy(prediction, label):
    # only input generative: 0
    return - tf.reduce_mean(
        (1 - label) * label * tf.log(prediction))


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
    loss = tf.reduce_sum((gradient_gen_x - gradient_label_x) * (gradient_gen_x - gradient_label_x)) + tf.reduce_sum(
        (gradient_gen_y - gradient_label_y) * (gradient_gen_y - gradient_label_y)) + tf.reduce_sum(
        (gradient_gen_z - gradient_label_z) * (gradient_gen_z - gradient_label_z))
    return loss
