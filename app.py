import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
import cv2
import streamlit as st
import pickle

with open('Dog-cat-classifier-model.pkl', 'rb') as file:
    model = pickle.load(file)


test_img2 = cv2.imread('/content/dog.jpg')
test_img2 = cv2.resize(test_img2, (256, 256))
test_input2 = test_img2.reshape((1, 256, 256, 3))

model.predict(test_input2)
