import tensorflow as tf


scalar_tensor = tf.constant(4)
print(scalar_tensor)

vector_tensor = tf.constant([1, 2, 3])
print(vector_tensor)

matrix_tensor = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float16)
print(matrix_tensor)
