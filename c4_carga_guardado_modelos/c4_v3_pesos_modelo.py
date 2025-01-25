import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train_save = x_train[:30000, :, :]
y_train_save = y_train[:30000]

x_train_load = x_train[30000:, :, :]
y_train_load = y_train[30000:]

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

# Guardado de los pesos después de entrenar
# model.fit(x_train_save, y_train_save, epochs=5)
# model.save_weights("models/mnist_weigths.weights.h5")

# Carga de los pesos para continuar el entrenamiento
model.load_weights("models/mnist_weigths.weights.h5")
model.fit(x_train_load, y_train_load, epochs=5)
