import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#input Data array
celsius = np.array( [0, 100, 200, 500, -270,420, 80, 56], dtype=float)
#Output data array
farenheit = np.array( [32, 212, 392, 932, -454, 788, 176, 132.8], dtype=float)

#One layer, one neuron
layer = tf.keras.layers.Dense(units=1, input_shape=[1])
#Sequential model
model = tf.keras.models.Sequential([layer])
#Model compilation
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)
# Lets begin training
historial = model.fit(celsius, farenheit, epochs=1000, verbose= False);
# Model Trained

#Loss function graph
plt.xlabel("# Epoch")
plt.ylabel("Loss function")
plt.plot(historial.history["loss"])
plt.show()

print(model.predict(np.array([100])))

#To stop script
print("To stop")