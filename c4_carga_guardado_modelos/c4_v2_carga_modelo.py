import tensorflow as tf

path = "models/mnist_model.h5"
mnist_model = tf.keras.models.load_model(path)

mnist_model.summary()
