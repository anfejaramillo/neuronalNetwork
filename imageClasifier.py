import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflowjs as tfjs

data, meta = tfds.load("fashion_mnist", as_supervised=True, with_info=True)
labels = meta.features["label"].names

#Normalize dataset function
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

#Normalize dataset
trainData = data["train"].map(normalize)
testData = data["test"].map(normalize)

#cache data
trainData = trainData.cache()
testData = testData.cache()

#Model definition
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)), #Input layer
    tf.keras.layers.Dense(50, activation = tf.nn.relu), #Hidden layer 1
    tf.keras.layers.Dense(50, activation = tf.nn.relu), #Hidden layer 1
    tf.keras.layers.Dense(10, activation = tf.nn.softmax), #Output layer (Softmax to output layers in classfication problems)
    ])
#Model compilation
model.compile(
    optimizer= "adam",
    loss=tf.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)
#Model calculation parameters
num_training_data = meta.splits["train"].num_examples
num_test_data = meta.splits["test"].num_examples
batchSize = 32
#Train model
trainData = trainData.repeat().shuffle(num_training_data).batch(batchSize)
testData = testData.repeat().shuffle(num_test_data).batch(batchSize)
historial = model.fit(trainData, epochs=5, steps_per_epoch= math.ceil(num_training_data/batchSize))
#Print testing result of images in test dateset
for imagenes_prueba, etiquetas_prueba in testData.take(1):
  imagenes_prueba = imagenes_prueba.numpy()
  etiquetas_prueba = etiquetas_prueba.numpy()
  predicciones = model.predict(imagenes_prueba)
  
def graficar_imagen(i, arr_predicciones, etiquetas_reales, imagenes):
  arr_predicciones, etiqueta_real, img = arr_predicciones[i], etiquetas_reales[i], imagenes[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img[...,0], cmap=plt.cm.binary)

  etiqueta_prediccion = np.argmax(arr_predicciones)
  if etiqueta_prediccion == etiqueta_real:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(labels[etiqueta_prediccion],
                                100*np.max(arr_predicciones),
                                labels[etiqueta_real]),
                                color=color)

def graficar_valor_arreglo(i, arr_predicciones, etiqueta_real):
  arr_predicciones, etiqueta_real = arr_predicciones[i], etiqueta_real[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  grafica = plt.bar(range(10), arr_predicciones, color="#777777")
  plt.ylim([0, 1]) 
  etiqueta_prediccion = np.argmax(arr_predicciones)
  
  grafica[etiqueta_prediccion].set_color('red')
  grafica[etiqueta_real].set_color('blue')

#Run plot
filas = 5
columnas = 5
num_imagenes = filas*columnas
plt.figure(figsize=(2*2*columnas, 2*filas))
for i in range(num_imagenes):
  plt.subplot(filas, 2*columnas, 2*i+1)
  graficar_imagen(i, predicciones, etiquetas_prueba, imagenes_prueba)
  plt.subplot(filas, 2*columnas, 2*i+2)
  graficar_valor_arreglo(i, predicciones, etiquetas_prueba)
plt.show()

#Save model
model.save("modelTopologyAndWeights_Clasifier.h5")

#export model
tfjs.converters.save_keras_model(model, "modelJS")

print("Stop")