import tensorflow as tf


resnet_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)
resnet_model.trainable = False
