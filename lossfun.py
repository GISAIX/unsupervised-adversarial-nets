import tensorflow as tf


def domain_loss(prediction, label, class_num=2):
    softmax_prediction = tf.nn.softmax(logits=prediction)
    ground_truth = tf.one_hot(indices=label, depth=class_num)
    loss = 0
    for i in range(class_num):
        i_prediction = softmax_prediction[:, i]
        i_ground_truth = ground_truth[:, i]
        loss -= tf.reduce_mean(i_ground_truth * tf.log(
            tf.clip_by_value(t=i_prediction, clip_value_min=0.005, clip_value_max=1)))
    return loss
