# Define Pretrained Base
pretrained_base = tf.keras.models.load_models(
    '../input/cv-course-models/vgg16-pretrained-base',
)
pretrained_base.trainable = False

# Attach Head
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    pretrained_base,
    layers.Flatten() # Flatten transforms the two dimensional outputs of the base into the one dimensional inputs needed by the head
    layers.Dense(6, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])

# Train
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=30,
    verbose=0,
)

# The history object contains loss and metric in a dictionary history.history
# Use Pandas to convert this dictionary to a dataframe and plot it with a built-in method
import pandas as pd

history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot();
