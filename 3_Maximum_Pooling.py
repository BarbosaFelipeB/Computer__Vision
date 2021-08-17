# Condense with Maximum Pooling
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Conv2D(filters=64, kernel_size=3),
    layers.MaxPool2D(pool_size=2),
    # More layers follow
])

# Aplly Maximum Pooling

import tensorflow as tf

image_condense = tf.nn.pool(
    input=image_detect,
    window_shape=(2, 2),
    pooling_type="MAX",
    strides=(2, 2),
    padding='SAME',
)

plt.figure(fig_size=(6, 6))
plt.imshow(tf.squeeze(image_condense))
plt.axis('off')
plt.show();