import tensorflow as tf


data = [1, 2, 3, 4, 5]
dataset = tf.data.Dataset.from_tensor_slices(data)

dataset = dataset.map(lambda x: x * 2)

print(dataset)
for record in dataset:
    print(record, record.numpy())
