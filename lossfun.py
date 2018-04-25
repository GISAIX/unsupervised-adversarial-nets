import tensorflow as tf


def dice_loss(prediction, label, class_num):
    # softmax processing
    softmax_prediction = tf.nn.softmax(logits=prediction)
    ground_truth = tf.one_hot(indices=label, depth=class_num)
    loss = 0
    # unique = len(tf.unique(label))
    for i in range(class_num):
        i_prediction = softmax_prediction[:, :, :, :, i]
        i_ground_truth = ground_truth[:, :, :, :, i]
        intersection = tf.reduce_sum(i_prediction * i_ground_truth)
        # square before
        union = tf.reduce_sum(i_prediction) + tf.reduce_sum(i_ground_truth) + 1e-5
        # adjusted weight
        weight = 1 - tf.reduce_sum(i_ground_truth) / tf.reduce_sum(ground_truth)
        loss += (1 - 2 * intersection / union) * weight
    return loss


def cross_entropy_loss(prediction, label, class_num):
    # loss = weight * - target * log(softmax(logits))
    softmax_prediction = tf.nn.softmax(logits=prediction)
    ground_truth = tf.one_hot(indices=label, depth=class_num)
    loss = 0
    for i in range(class_num):
        i_prediction = softmax_prediction[:, :, :, :, i]
        i_ground_truth = ground_truth[:, :, :, :, i]
        # adjusted weight
        weight = 1 - tf.reduce_sum(i_ground_truth) / tf.reduce_sum(ground_truth)
        loss -= tf.reduce_mean(weight * i_ground_truth * tf.log(
            tf.clip_by_value(t=i_prediction, clip_value_min=0.005, clip_value_max=1)))
    return loss


def discriminative_loss(prediction, label, class_num=2):
    softmax_prediction = tf.nn.softmax(logits=prediction)
    ground_truth = tf.one_hot(indices=label, depth=class_num)
    # input shape [Batch, class_num]
    loss = 0
    for i in range(class_num):
        i_prediction = softmax_prediction[:, i]
        i_ground_truth = ground_truth[:, i]
        loss -= tf.reduce_mean(i_ground_truth * tf.log(
            tf.clip_by_value(t=i_prediction, clip_value_min=0.005, clip_value_max=1)))
    return loss
