import pandas as pd
import tensorflow as tf


# Normalización
df = pd.read_csv("data/datos.csv")

tensor_data = tf.convert_to_tensor(df.values, dtype=tf.float32)
normalizer = tf.keras.layers.Rescaling(1./tf.reduce_max(tensor_data, axis=0))
normalized_tensor = normalizer(tensor_data)

df_normalized = pd.DataFrame(normalized_tensor.numpy(), columns=df.columns)
print(df_normalized.head())


# One-Hot Encoding
labels = [1, 2, 3]
one_hot = tf.keras.utils.to_categorical(labels)
print(one_hot)


# Tokenizar textos
texts = [
    "Había una vez una iguana",
    "Con una ruana de lana",
    "Peinandose la melena",
    "Junto al río Magdalena"
]

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
print(sequences)
