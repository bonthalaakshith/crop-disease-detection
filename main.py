import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def normalize_image(image):
    image = 255 * (image + 1.0) / 2.0
    return tf.cast(image, tf.uint8)

def deepdream(model, image, steps, step_size,image_path):
    image = tf.convert_to_tensor(image)
    for _ in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(image)
            activations = model(image)
            loss = tf.reduce_sum(activations)
        gradients = tape.gradient(loss, image)
        gradients /= tf.math.reduce_std(gradients) + 1e-8
        image = image + gradients * step_size
        image = tf.clip_by_value(image, -1, 1)
    return image

def deepseg(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    segmented_image = cv2.bitwise_and(image, image, mask=mask)
    cv2.imwrite('static/uploads/segmented_image.jpg', segmented_image)
    outputpath1='static/uploads/segmented_image.jpg'
    image1 = cv2.imread(image_path)
    image2 = cv2.imread(outputpath1)
    img = cv2.addWeighted(image1, 0.2, image2, 0.9, 0)
    cv2.imwrite('static/uploads/imageseg.jpg', img)
    return


base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

layer_names = ['mixed3', 'mixed5']
layers = [base_model.get_layer(name).output for name in layer_names]

dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

image_path = 'static/uploads/original.jpg'
img_raw = tf.io.read_file(image_path)
img = tf.image.decode_image(img_raw)
img = tf.image.convert_image_dtype(img, tf.float32)
img = tf.image.resize(img, (224, 224))
img = img[tf.newaxis, :]

steps = 100
step_size = 0.01

def deepdream_seg():
    dream_img =deepseg(image_path)
    return "finish"


