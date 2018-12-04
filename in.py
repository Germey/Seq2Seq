import tensorflow as tf


def contains(history, a):
    if not history:
        return False
    
    stacked = tf.stack(history)
    print(stacked)
    with tf.Session() as sess:
        result = tf.reduce_sum(tf.cast(tf.equal(a, stacked), tf.float32)).eval()
        print(result)
        return result


history = []

a = tf.constant(1, tf.float32)
if not contains(history, a):
    print('Not Contains')
    history.append(a)

b = tf.constant(2, tf.float32)
if not contains(history, b):
    print('Not Contains')
    history.append(b)

print(history)
