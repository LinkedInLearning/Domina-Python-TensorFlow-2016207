import tensorflow as tf

dataset = tf.data.experimental.make_csv_dataset(
    "data/datos.csv",
    batch_size=10,
    header=True,
    label_name="etiqueta",
    shuffle=True,
    num_epochs=1,
)

print(dataset.take(1))

for features, labels in dataset.take(2):
    print(features)
    print(labels)
