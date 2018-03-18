import tensorflow as tf


def dice_loss(prediction, label, class_num):
    # no softmax processing
    ground_truth = tf.one_hot(indices=label, depth=class_num)
    dice = 0
    for i in range(class_num):
        i_prediction = prediction[:, :, :, :, i]
        i_ground_truth = ground_truth[:, :, :, :, i]
        intersection = tf.reduce_sum(i_prediction * i_ground_truth)
        union_prediction = tf.reduce_sum(i_prediction * i_prediction)
        union_ground_truth = tf.reduce_sum(i_ground_truth * i_ground_truth)
        union = union_ground_truth + union_prediction
        # adjusted weight
        weight = tf.reduce_sum(i_ground_truth) / tf.reduce_sum(ground_truth)
        dice += (1 - 2 * intersection / union) * (1 - weight) / (class_num - 1)
    return dice


def cross_entropy_loss(prediction, label, class_num):
    # loss = weight * - target * log(softmax(logits))
    # prediction = logits
    softmax_prediction = tf.nn.softmax(logits=prediction)
    ground_truth = tf.one_hot(indices=label, depth=class_num)
    loss = 0
    for i in range(class_num):
        i_prediction = softmax_prediction[:, :, :, :, i]
        i_ground_truth = ground_truth[:, :, :, :, i]
        # adjusted weight
        weight = tf.reduce_sum(i_ground_truth) / tf.reduce_sum(ground_truth)
        loss -= tf.reduce_mean(
            (1 - weight) / (class_num-1) * i_ground_truth * tf.log(
                tf.clip_by_value(t=i_prediction, clip_value_min=0.005, clip_value_max=1)))
    return loss
