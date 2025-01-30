import tensorflow as tf

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

dataset = tf.data.Dataset.from_tensor_slices(data)

dataset = dataset.shuffle(buffer_size=10)
dataset = dataset.batch(3)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

for batch in dataset:
    print(batch.numpy())
