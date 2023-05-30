import tensorflow as tf

model = 'cnn_fruits.hdf5'

tflite_model = tf.keras.models.load_model(model)
converter = tf.lite.TFLiteConverter.from_keras_model(tflite_model)
tflite_save = converter.convert()
open("cnn_fruits.tflite", "wb").write(tflite_save)