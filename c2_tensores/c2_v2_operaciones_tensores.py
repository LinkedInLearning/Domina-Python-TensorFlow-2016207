import tensorflow as tf


tensor_1 = tf.constant([[1, 2], [3, 4]])
tensor_2 = tf.constant([[1, 1], [1, 1]])

sum_result = tf.add(tensor_1, tensor_2)
print(sum_result)

mult_result = tf.divide(tensor_1, tensor_2)
print(mult_result)

matrix_product = tf.matmul(tensor_1, tensor_2)
print(matrix_product)

transpose = tf.transpose(tensor_1)
print(transpose)

reduce_sum = tf.reduce_sum(tensor_1)
print(reduce_sum)

reduce_max = tf.reduce_sum(tensor_1)
print(reduce_sum)
