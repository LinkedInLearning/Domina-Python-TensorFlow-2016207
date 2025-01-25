import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split


np.random.seed(0)
tf.random.set_seed(15)

x = np.random.rand(100, 1) * 10 
y = x + 2

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=0
)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_dim=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=100)
y_pred = model.predict(x_test)


plt.scatter(x_train, y_train, color="blue")
plt.scatter(x_test, y_pred, color="orange")
plt.show()
