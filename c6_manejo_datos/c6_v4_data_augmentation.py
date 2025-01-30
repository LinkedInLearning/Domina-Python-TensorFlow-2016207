import os
import tensorflow as tf


def read_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image


def augment_image(image):
    augmented = tf.image.resize(image, [1024, 1024])
    augmented = tf.image.random_flip_left_right(augmented)
    augmented = tf.image.random_flip_up_down(augmented)
    augmented = tf.image.random_contrast(augmented, 0.2, 2.0)
    return augmented


def save_image(image, filename):
    image = tf.cast(image, tf.uint8)
    image = tf.image.encode_jpeg(image)
    tf.io.write_file(filename, image) 


output_dir = "augmented_images/"
os.makedirs(output_dir, exist_ok=True)

images_dir = "images/"
original_images_path = tf.data.Dataset.list_files(os.path.join(images_dir, '*.jpeg'))

for i, image_path in enumerate(original_images_path):
    image = read_image(image_path)
    augmented = augment_image(image)
    augmented_path = os.path.join(output_dir, f"augmented_{i}.jpeg")
    save_image(augmented, augmented_path)
