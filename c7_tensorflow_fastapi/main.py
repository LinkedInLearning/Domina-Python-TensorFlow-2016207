import numpy as np
import tensorflow as tf

from io import BytesIO
from PIL import Image

from fastapi import FastAPI, File, UploadFile


app = FastAPI()

MODEL = tf.keras.models.load_model("mnist_model.keras")



@app.get("/")
def get_info():
    return {"message": "API to predict MNIST digits."}


@app.post("/predict")
def predict(image: UploadFile = File(...)):

    image_bytes = image.file.read()

    img = Image.open(BytesIO(image_bytes))
    img = img.convert("L")
    img = img.resize((28, 28))

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=-1)

    prediction = MODEL.predict(np.expand_dims(img_array, axis=0))
    predicted_class = np.argmax(prediction, axis=1)

    return {"file": image.filename, "class": str(predicted_class[0])}
