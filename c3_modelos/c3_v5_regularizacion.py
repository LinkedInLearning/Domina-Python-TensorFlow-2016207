import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

early_stopping = EarlyStopping(monitor="loss")

model.fit(x_train, y_train, epochs=50, callbacks=[early_stopping])

y_pred = model.predict(x_test)
y_pred_labels = y_pred.argmax(axis=1)
