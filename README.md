# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

MNIST Handwritten Digit Classification Dataset is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9.

The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively.

![image](https://github.com/ShriramGH/mnist-classification/assets/117991122/01002512-e6da-46e2-8690-4015976dc662)


## Neural Network Model

![Screenshot 2024-03-15 143809](https://github.com/ShriramGH/mnist-classification/assets/117991122/d2fc6768-4c58-4c89-b1f7-70b57772669f)


## DESIGN STEPS

### STEP 1:
Import tensorflow and preprocessing libraries
### STEP 2:

Build a CNN model
### STEP 3:
Compile and fit the model and then predict

## PROGRAM

### Name: SHRIRAM S
### Register Number: 212222240098

```py
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
```
```py
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```
```py
X_train.shape

X_test.shape
```
```py
single_image= X_train[0]

single_image.shape
```
```py
plt.imshow(single_image,cmap='gray')

y_train.shape

X_train.min()

X_train.max()
```
```py
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
```
```py
X_train_scaled.min()

X_train_scaled.max()
```
```py
y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
```
```py
type(y_train_onehot)
```
```py
y_train_onehot.shape

single_image = X_train[500]
plt.imshow(single_image,cmap='gray')
```
```py
y_train_onehot[500]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)
```
```py
model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32, kernel_size=(3,3),  activation='relu')),
model.add(layers.MaxPool2D(pool_size=(2, 2))),
model.add(layers.Flatten()),
model.add(layers.Dense(64, activation='relu')),
model.add(layers.Dense(10, activation='softmax'))
```
```py
model.summary()
```
```py
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')
```
```py
model.fit(X_train_scaled ,y_train_onehot, epochs=10,
          batch_size=128,
          validation_data=(X_test_scaled,y_test_onehot))
```
```py
metrics = pd.DataFrame(model.history.history)
```
```py
metrics.head()
```
```py
metrics[['accuracy','val_accuracy']].plot()

metrics[['loss','val_loss']].plot()
```
```py
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
```
```py
print(confusion_matrix(y_test,x_test_predictions))
```
```py
print(classification_report(y_test,x_test_predictions))
```
```py
img = image.load_img('/content/Screenshot 2024-03-11 083455.png')
```
```py
type(img)
```
```py
img = image.load_img('/content/mnist1.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
```
```py
x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

```


## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/ShriramGH/mnist-classification/assets/117991122/5e0c97e3-15df-4a2c-9dd6-88557498b17c)


![image](https://github.com/ShriramGH/mnist-classification/assets/117991122/2e09ad23-56bd-4aba-b45c-c38fe7710fdc)


### Classification Report

![image](https://github.com/ShriramGH/mnist-classification/assets/117991122/11c371da-1cce-4bd2-97c8-c42c2be24a90)


### Confusion Matrix

![image](https://github.com/ShriramGH/mnist-classification/assets/117991122/1d568d4f-8819-4418-8c69-4c22684ed686)

### New Sample Data Prediction

![image](https://github.com/ShriramGH/mnist-classification/assets/117991122/ffe0be88-21d8-4d8b-a892-150070c834a2)


## RESULT
Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.
