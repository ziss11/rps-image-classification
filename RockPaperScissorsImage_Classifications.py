# %%
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input

# %%
base_dir = '../../Datasets/images/scene_images/image'

# %%
datagen = ImageDataGenerator(
    rotation_range=20,
    rescale=1./255.0,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2,
)

train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    class_mode='categorical',
    color_mode='rgb',
    batch_size=64,
    subset='training',
)
val_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    class_mode='categorical',
    color_mode='rgb',
    batch_size=64,
    subset='validation',
)

# %%
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                           input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax'),
])
model.summary()

# %%
lr = 1e-5
loss = 'categorical_crossentropy'
optimizer = tf.optimizers.Adam(learning_rate=lr)
model.compile(
    loss=loss,
    optimizer=optimizer,
    metrics=['accuracy']
)

# %%
callbacks = tf.keras.callbacks.EarlyStopping(
    monitor='accuracy', patience=3, mode='max')

# %%
epoch = 30
hist = model.fit(
    train_generator,
    epochs=epoch,
    validation_data=val_generator,
    batch_size=64,
    callbacks=[callbacks],
)

# %%
plt.figure(figsize=(18, 5))

plt.subplot(1, 2, 1)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Akurasi model'),
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['train', 'test'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss model'),
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['train', 'test'], loc='upper left')

plt.show()

# %%
# # Menyimpan model dalam format SavedModel
# export_dir = 'saved_model/'
# tf.saved_model.save(model, export_dir)

# # Convert SavedModel menjadi vegs.tflite
# converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
# tflite_model = converter.convert()

# tflite_model_file = pathlib.Path('vegs.tflite')
# tflite_model_file.write_bytes(tflite_model)
