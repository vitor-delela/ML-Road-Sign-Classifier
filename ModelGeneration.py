import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from DatasetMgmt import DatasetTransito
from GermanDatasetMgmt import GermanDataLoader
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from PIL import Image
from keras.optimizers import Adam, RMSprop, Adadelta, Adagrad, SGD
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

my_loader = GermanDataLoader() 
(train_images, train_labels), \
(valid_images, valid_labels), \
(test_images, test_labels) = my_loader.load_data()

# Train features shape: (34799, 32, 32, 3)
# Valid features shape: (4410, 32, 32, 3)
# Test features shape: (12630, 32, 32, 3)

train_images = train_images.reshape((len(train_images), 32, 32, 3)).astype('float32') / 255
test_images = test_images.reshape((len(test_images), 32, 32, 3)).astype('float32') / 255
valid_images = valid_images.reshape((len(valid_images), 32, 32, 3)).astype('float32') / 255

# one-hot-encoding ex: [0. 0. 1. 0.]
categorical_train_labels = to_categorical(train_labels)
categorical_valid_labels = to_categorical(valid_labels)
categorical_test_labels = to_categorical(test_labels)

model = models.Sequential()
model.add(layers.Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(32,32,3),padding='same'))
model.add(layers.MaxPooling2D((2, 2),padding='same'))
model.add(layers.Dropout(0.25)) # dropout 1 camada
model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(layers.Dropout(0.25))  # dropout 2 camada
model.add(layers.Conv2D(128, (3, 3), activation='relu',padding='same'))                 
model.add(layers.MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(layers.Dropout(0.4))  # dropout 3 camada
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))    
model.add(layers.Dropout(0.3))      
model.add(layers.Dense(43, activation='softmax'))

optimizer = RMSprop(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
#  ------------------------------------------------------------------------------------------------------------

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.4,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(train_images)

# model_train = model.fit(datagen.flow(train_images, categorical_train_labels, batch_size=64), epochs=10,verbose=1, validation_data=(valid_images, categorical_valid_labels)) # com data augmentation

model_train = model.fit(train_images, categorical_train_labels, epochs=10, batch_size=64, verbose=1, validation_data=(valid_images, categorical_valid_labels)) # sem data augmentation

test_loss, test_acc = model.evaluate(test_images, categorical_test_labels)
print(f'\n\nAcurácia no conjunto de teste: {test_acc}')
print(f'Loss no conjunto de teste: {test_loss}')

# model.save('model_10epochs_rmsprop_dropout.h5')

# Generated model predictions 
predicted_classes = model.predict(test_images)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
correct = np.where(predicted_classes==test_labels)[0]
print ("Found %d correct labels" % len(correct))

target_names = ["Class {}".format(i) for i in range(43)]
print(classification_report(test_labels, predicted_classes, target_names=target_names))

# Accuracy & Loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(model_train.history['accuracy'])
plt.plot(model_train.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(model_train.history['loss'])
plt.plot(model_train.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
