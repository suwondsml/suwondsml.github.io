---
layout: post
title: TFLite-sound
authors: [SiEun Kim]
categories: [1기 AI/SW developers(개인 프로젝트)]
---


```python
flag_colab = 1
flag_hub = 0
if flag_colab == 1:
    from google.colab import drive
    drive.mount('/content/gdrive/')
    path_root = '/content/gdrive/MyDrive/proj/soundsep/dataset_img_224'
    
    !pip install h5py==2.10.0
    !pip install tensorflow_hub
else:
    path_root = 'c:/proj/soundsep/dataset_img_224'
```

    Mounted at /content/gdrive/
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting h5py==2.10.0
      Downloading h5py-2.10.0-cp37-cp37m-manylinux1_x86_64.whl (2.9 MB)
    [K     |████████████████████████████████| 2.9 MB 5.2 MB/s 
    [?25hRequirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from h5py==2.10.0) (1.15.0)
    Requirement already satisfied: numpy>=1.7 in /usr/local/lib/python3.7/dist-packages (from h5py==2.10.0) (1.21.6)
    Installing collected packages: h5py
      Attempting uninstall: h5py
        Found existing installation: h5py 3.1.0
        Uninstalling h5py-3.1.0:
          Successfully uninstalled h5py-3.1.0
    Successfully installed h5py-2.10.0
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: tensorflow_hub in /usr/local/lib/python3.7/dist-packages (0.12.0)
    Requirement already satisfied: protobuf>=3.8.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow_hub) (3.17.3)
    Requirement already satisfied: numpy>=1.12.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow_hub) (1.21.6)
    Requirement already satisfied: six>=1.9 in /usr/local/lib/python3.7/dist-packages (from protobuf>=3.8.0->tensorflow_hub) (1.15.0)
    


```python
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
```


```python
print("Version: ", tf.__version__)
#print("Hub version: ", hub.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")
```

    WARNING:tensorflow:From <ipython-input-3-f613a4bc459a>:4: is_gpu_available (from tensorflow.python.framework.test_util) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.config.list_physical_devices('GPU')` instead.
    

    Version:  2.8.2
    Eager mode:  True
    GPU is available
    

### Dataset preparation


```python
# Learn more about data batches
#Image batch shape:  (32, 224, 224, 3)
#Label batch shape:  (32, 5)
```


```python
folder_list = os.listdir(path_root)
print('folder_list',folder_list)
num_classes = len(folder_list)
print('num_classes', num_classes)
```

    folder_list ['howling', 'car_noise', 'white_noise', 'babbling', 'voice']
    num_classes 5
    


```python
from keras.preprocessing.image import ImageDataGenerator
```


```python
# Create data generator for training and validation

IMAGE_SHAPE = (224, 224)
TRAINING_DATA_DIR = str(path_root)

datagen_kwargs = dict(rescale=1./255, validation_split=.20)

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
valid_generator = valid_datagen.flow_from_directory(
    TRAINING_DATA_DIR, 
    subset="validation", 
    shuffle=True,
    target_size=IMAGE_SHAPE
)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
train_generator = train_datagen.flow_from_directory(
    TRAINING_DATA_DIR, 
    subset="training", 
    shuffle=True,
    target_size=IMAGE_SHAPE)
```

    Found 1000 images belonging to 5 classes.
    Found 4000 images belonging to 5 classes.
    


```python
# Learn more about data batches

image_batch_train, label_batch_train = next(iter(train_generator))
print("Image batch shape: ", image_batch_train.shape)
print("Label batch shape: ", label_batch_train.shape)
```

    Image batch shape:  (32, 224, 224, 3)
    Label batch shape:  (32, 5)
    


```python
# Learn about dataset labels

dataset_labels = sorted(train_generator.class_indices.items(), key=lambda pair:pair[1])
dataset_labels = np.array([key.title() for key, value in dataset_labels])
print(dataset_labels)
```

    ['Babbling' 'Car_Noise' 'Howling' 'Voice' 'White_Noise']
    


```python
# Get images and labels batch from validation dataset generator

val_image_batch, val_label_batch = next(iter(valid_generator))
true_label_ids = np.argmax(val_label_batch, axis=-1)

print("Validation batch shape:", val_image_batch.shape)
```

    Validation batch shape: (32, 224, 224, 3)
    

### Model architecture, training

As a base model for transfer learning, we'll use MobileNet v2 model stored on TensorFlow Hub. Presented model can be used only in TensorFlow 2.0 implementation (TF Hub contains also models for TensorFlow 1.x).

Basic information about feature vector:
- Input shape: 224x224x3 (224x224 pixels, 3 chanels each, RGB format),
- Each channel has value in range [0, 1],
- Feature vector output shape: 1280 (number of labels classified by MobileNet is 1001 - this info isn't important here)

For more details check feature vector page:
https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4


```python
if flag_hub == 1:
    base_model = hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4", 
                 output_shape=[1280],
                 trainable=False)
else:    
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        alpha=0.35,
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        pooling=None,
        classes=num_classes,
        classifier_activation='softmax'
    )
    base_model.trainable = False

```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_224_no_top.h5
    2023424/2019640 [==============================] - 0s 0us/step
    2031616/2019640 [==============================] - 0s 0us/step
    


```python
rate_dropout = 0.2
```


```python
model = tf.keras.Sequential()
model.add(base_model)

if flag_hub == 0:
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    
model.add(tf.keras.layers.Dropout(rate_dropout))

model.add(tf.keras.layers.Dense(360, activation='relu'))
model.add(tf.keras.layers.Dropout(rate_dropout))
model.add(tf.keras.layers.Dense(180, activation='relu'))
model.add(tf.keras.layers.Dropout(rate_dropout))
model.add(tf.keras.layers.Dense(30, activation='relu'))
model.add(tf.keras.layers.Dropout(rate_dropout))

model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

```


```python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```


```python
model.fit(
        train_generator,
        steps_per_epoch=50,
        epochs=10,
        validation_data=valid_generator,
        validation_steps=10)
```

    Epoch 1/10
    50/50 [==============================] - 467s 9s/step - loss: 1.1575 - accuracy: 0.5100 - val_loss: 0.8943 - val_accuracy: 0.5875
    Epoch 2/10
    50/50 [==============================] - 288s 6s/step - loss: 0.7457 - accuracy: 0.7262 - val_loss: 0.6046 - val_accuracy: 0.6781
    Epoch 3/10
    50/50 [==============================] - 173s 3s/step - loss: 0.5373 - accuracy: 0.8138 - val_loss: 0.5198 - val_accuracy: 0.8031
    Epoch 4/10
    50/50 [==============================] - 118s 2s/step - loss: 0.4675 - accuracy: 0.8294 - val_loss: 0.5532 - val_accuracy: 0.7437
    Epoch 5/10
    50/50 [==============================] - 77s 2s/step - loss: 0.4016 - accuracy: 0.8481 - val_loss: 0.2807 - val_accuracy: 0.9187
    Epoch 6/10
    50/50 [==============================] - 49s 986ms/step - loss: 0.3455 - accuracy: 0.8725 - val_loss: 0.3139 - val_accuracy: 0.8875
    Epoch 7/10
    50/50 [==============================] - 30s 603ms/step - loss: 0.2960 - accuracy: 0.8969 - val_loss: 0.4946 - val_accuracy: 0.7875
    Epoch 8/10
    50/50 [==============================] - 24s 482ms/step - loss: 0.3243 - accuracy: 0.8888 - val_loss: 0.3133 - val_accuracy: 0.8594
    Epoch 9/10
    50/50 [==============================] - 15s 309ms/step - loss: 0.3004 - accuracy: 0.8981 - val_loss: 0.1924 - val_accuracy: 0.9625
    Epoch 10/10
    50/50 [==============================] - 13s 256ms/step - loss: 0.2703 - accuracy: 0.9056 - val_loss: 0.1275 - val_accuracy: 0.9781
    




    <keras.callbacks.History at 0x7fda67d92ed0>



model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss='categorical_crossentropy',
  metrics=['acc'])

# Run model training

hist = model.fit(feature_train, label_train, validation_split = 0.3, epochs=20, verbose=1).history


```python
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
```


```python
image_w = 224
image_h = 224
data_feature = []
data_label = []
idx_plot = 0
for index in range(len(folder_list)):
    path_file = os.path.join(path_root, folder_list[index])
    
    img_list = os.listdir(path_file)
    for img in img_list:
        
        img_path = os.path.join(path_file, img)
        '''img = cv2.cv2.imread(img_path, cv2.IMREAD_COLOR)'''
        img = cv2.imread(img_path) #, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
        # 1) standardize or
        img = tf.image.per_image_standardization(img)
        data_feature.append(img)
        # 2) standardize 
        data_label.append(index)
        
```


```python
num_color = 3
data_feature = tf.reshape(data_feature, [-1, image_w, image_h, num_color]) 
print('np.shape(data_feature)', np.shape(data_feature))
```

    np.shape(data_feature) (5000, 224, 224, 3)
    


```python
data_label = to_categorical(data_label, num_classes)
```


```python
data_feature = np.array(data_feature)
data_label = np.array(data_label)
```


```python
from sklearn.model_selection import train_test_split
feature_train, feature_test, label_train, label_test = train_test_split(data_feature,data_label)
print('label_train',label_train)
```

    label_train [[0. 0. 0. 0. 1.]
     [1. 0. 0. 0. 0.]
     [0. 0. 1. 0. 0.]
     ...
     [0. 1. 0. 0. 0.]
     [0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 1.]]
    


```python
hist = model.fit(feature_train, label_train, validation_split = 0.3, epochs=20, verbose=1).history
```

    Epoch 1/20
    83/83 [==============================] - 9s 79ms/step - loss: 1.4182 - acc: 0.5375 - val_loss: 0.5490 - val_acc: 0.8187
    Epoch 2/20
    83/83 [==============================] - 3s 36ms/step - loss: 0.5443 - acc: 0.7958 - val_loss: 0.3245 - val_acc: 0.8818
    Epoch 3/20
    83/83 [==============================] - 3s 37ms/step - loss: 0.4372 - acc: 0.8495 - val_loss: 0.2012 - val_acc: 0.9440
    Epoch 4/20
    83/83 [==============================] - 3s 36ms/step - loss: 0.3131 - acc: 0.8933 - val_loss: 0.1733 - val_acc: 0.9458
    Epoch 5/20
    83/83 [==============================] - 3s 36ms/step - loss: 0.2661 - acc: 0.9139 - val_loss: 0.1140 - val_acc: 0.9653
    Epoch 6/20
    83/83 [==============================] - 3s 36ms/step - loss: 0.2226 - acc: 0.9250 - val_loss: 0.0937 - val_acc: 0.9680
    Epoch 7/20
    83/83 [==============================] - 3s 37ms/step - loss: 0.2127 - acc: 0.9333 - val_loss: 0.1009 - val_acc: 0.9653
    Epoch 8/20
    83/83 [==============================] - 3s 37ms/step - loss: 0.1603 - acc: 0.9459 - val_loss: 0.0971 - val_acc: 0.9707
    Epoch 9/20
    83/83 [==============================] - 3s 36ms/step - loss: 0.1727 - acc: 0.9436 - val_loss: 0.0925 - val_acc: 0.9707
    Epoch 10/20
    83/83 [==============================] - 3s 36ms/step - loss: 0.1587 - acc: 0.9528 - val_loss: 0.0961 - val_acc: 0.9671
    Epoch 11/20
    83/83 [==============================] - 3s 36ms/step - loss: 0.1607 - acc: 0.9448 - val_loss: 0.0959 - val_acc: 0.9689
    Epoch 12/20
    83/83 [==============================] - 3s 36ms/step - loss: 0.1763 - acc: 0.9482 - val_loss: 0.0914 - val_acc: 0.9689
    Epoch 13/20
    83/83 [==============================] - 3s 36ms/step - loss: 0.1233 - acc: 0.9611 - val_loss: 0.0894 - val_acc: 0.9707
    Epoch 14/20
    83/83 [==============================] - 3s 36ms/step - loss: 0.1213 - acc: 0.9604 - val_loss: 0.0946 - val_acc: 0.9707
    Epoch 15/20
    83/83 [==============================] - 3s 42ms/step - loss: 0.1156 - acc: 0.9657 - val_loss: 0.0797 - val_acc: 0.9733
    Epoch 16/20
    83/83 [==============================] - 3s 42ms/step - loss: 0.1292 - acc: 0.9543 - val_loss: 0.1445 - val_acc: 0.9378
    Epoch 17/20
    83/83 [==============================] - 3s 36ms/step - loss: 0.2073 - acc: 0.9330 - val_loss: 0.0755 - val_acc: 0.9751
    Epoch 18/20
    83/83 [==============================] - 3s 36ms/step - loss: 0.1675 - acc: 0.9459 - val_loss: 0.0808 - val_acc: 0.9733
    Epoch 19/20
    83/83 [==============================] - 3s 37ms/step - loss: 0.1198 - acc: 0.9604 - val_loss: 0.0897 - val_acc: 0.9751
    Epoch 20/20
    83/83 [==============================] - 3s 36ms/step - loss: 0.1426 - acc: 0.9528 - val_loss: 0.0774 - val_acc: 0.9742
    


```python
# Visualize training process

plt.figure()
plt.ylabel("Loss (training and validation)")
plt.xlabel("Training Steps")
plt.ylim([0,2])
plt.plot(hist["loss"])
plt.plot(hist["val_loss"])

plt.figure()
plt.ylabel("Accuracy (training and validation)")
plt.xlabel("Training Steps")
plt.ylim([0,1])
plt.plot(hist["acc"])
plt.plot(hist["val_acc"])
```




    [<matplotlib.lines.Line2D at 0x7f59a27edc10>]




    
![png](output_26_1.png)
    



    
![png](output_26_2.png)
    


# Measure accuracy and loss after training

final_loss, final_accuracy = model.evaluate(feature_test, label_test)

#print(feature_test)
print(np.unique(feature_test))

print("Final loss: {:.2f}".format(final_loss))
print("Final accuracy: {:.2f}%".format(final_accuracy * 100))


```python
final_loss, final_accuracy = model.evaluate(feature_test, label_test)
```

    40/40 [==============================] - 1s 33ms/step - loss: 0.0798 - acc: 0.9720
    


```python

```


```python
SOUNDSEP_SAVED_MODEL = './model'
tf.saved_model.save(model, SOUNDSEP_SAVED_MODEL)
```

    WARNING:absl:Function `_wrapped_model` contains input name(s) mobilenetv2_0.35_224_input with unsupported characters which will be renamed to mobilenetv2_0_35_224_input in the SavedModel.
    


```python
# Load SavedModel
soundsep_model = tf.saved_model.load(SOUNDSEP_SAVED_MODEL)

#soundsep_model = hub.load(SOUNDSEP_SAVED_MODEL)
print(soundsep_model)
```

    <tensorflow.python.saved_model.load.Loader._recreate_base_user_object.<locals>._UserObject object at 0x7f56776de350>
    


```python
# TF Lite 특징

# 모델 사이즈를 줄이기 위한 Quantization 등과 같은 모델 최적화를 위한 툴을 제공함
# 이러한 툴들은 모델 사이즈를 줄이고, 속도를 향상시키면서 정확도의 손실은 최소화함
```


```python
from keras.preprocessing.image import ImageDataGenerator
```


```python
# Create data generator for training and validation

datagen_kwargs = dict(rescale=1./255, validation_split=.20)

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
valid_generator = valid_datagen.flow_from_directory(
    TRAINING_DATA_DIR, 
    subset="validation", 
    shuffle=True,
    target_size=IMAGE_SHAPE
)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
train_generator = train_datagen.flow_from_directory(
    TRAINING_DATA_DIR, 
    subset="training", 
    shuffle=True,
    target_size=IMAGE_SHAPE)
```

    Found 1000 images belonging to 5 classes.
    Found 4000 images belonging to 5 classes.
    


```python
# Learn more about data batches

image_batch_train, label_batch_train = next(iter(train_generator))
print("Image batch shape: ", image_batch_train.shape)
print("Label batch shape: ", label_batch_train.shape)
```

    Image batch shape:  (32, 224, 224, 3)
    Label batch shape:  (32, 5)
    


```python
# Learn about dataset labels

dataset_labels = sorted(train_generator.class_indices.items(), key=lambda pair:pair[1])
dataset_labels = np.array([key.title() for key, value in dataset_labels])
print(dataset_labels)
```

    ['Babbling' 'Car_Noise' 'Howling' 'Voice' 'White_Noise']
    


```python
# Get images and labels batch from validation dataset generator

val_image_batch, val_label_batch = next(iter(valid_generator))
true_label_ids = np.argmax(val_label_batch, axis=-1)

print("Validation batch shape:", val_image_batch.shape)
```

    Validation batch shape: (32, 224, 224, 3)
    


```python
tf_model_predictions = soundsep_model(val_image_batch)
print("Prediction results shape:", tf_model_predictions.shape)
```

    Prediction results shape: (32, 5)
    


```python
# Convert prediction results to Pandas dataframe, for better visualization

tf_pred_dataframe = pd.DataFrame(tf_model_predictions.numpy())
tf_pred_dataframe.columns = dataset_labels

print("Prediction results for the first elements")
tf_pred_dataframe
```

    Prediction results for the first elements
    





  <div id="df-96de3eee-a6f5-4551-b306-e41e5a1cb3f1">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Babbling</th>
      <th>Car_Noise</th>
      <th>Howling</th>
      <th>Voice</th>
      <th>White_Noise</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.996504</td>
      <td>0.003487</td>
      <td>2.118114e-08</td>
      <td>5.414993e-09</td>
      <td>9.312005e-06</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.757290</td>
      <td>0.199218</td>
      <td>2.382135e-03</td>
      <td>5.897117e-05</td>
      <td>4.105059e-02</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.950432</td>
      <td>0.027393</td>
      <td>6.521549e-05</td>
      <td>4.468872e-06</td>
      <td>2.210522e-02</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000388</td>
      <td>0.000078</td>
      <td>1.112759e-05</td>
      <td>9.131792e-06</td>
      <td>9.995135e-01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.672573</td>
      <td>0.314011</td>
      <td>9.655589e-04</td>
      <td>3.414283e-05</td>
      <td>1.241581e-02</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.823960</td>
      <td>0.147186</td>
      <td>8.854305e-04</td>
      <td>2.055392e-05</td>
      <td>2.794863e-02</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.999659</td>
      <td>0.000223</td>
      <td>5.976393e-08</td>
      <td>5.216237e-10</td>
      <td>1.180826e-04</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.023955</td>
      <td>0.975011</td>
      <td>1.561883e-04</td>
      <td>4.750659e-05</td>
      <td>8.304650e-04</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.999959</td>
      <td>0.000040</td>
      <td>4.840833e-10</td>
      <td>8.829574e-12</td>
      <td>7.326586e-07</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.976733</td>
      <td>0.023264</td>
      <td>4.753583e-09</td>
      <td>2.573430e-09</td>
      <td>3.328823e-06</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.108852</td>
      <td>0.005634</td>
      <td>2.559468e-04</td>
      <td>2.423567e-05</td>
      <td>8.852341e-01</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.009138</td>
      <td>0.989342</td>
      <td>2.325181e-04</td>
      <td>5.623966e-05</td>
      <td>1.231087e-03</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.000034</td>
      <td>0.999961</td>
      <td>4.666264e-07</td>
      <td>8.925396e-08</td>
      <td>4.661028e-06</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.977362</td>
      <td>0.021305</td>
      <td>1.455781e-06</td>
      <td>2.483222e-07</td>
      <td>1.331296e-03</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.655747</td>
      <td>0.324101</td>
      <td>4.146743e-03</td>
      <td>2.242918e-04</td>
      <td>1.578104e-02</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.099664</td>
      <td>0.002909</td>
      <td>2.919936e-05</td>
      <td>1.771856e-06</td>
      <td>8.973963e-01</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.505737</td>
      <td>0.478238</td>
      <td>1.276309e-03</td>
      <td>2.202171e-04</td>
      <td>1.452795e-02</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.010122</td>
      <td>0.988985</td>
      <td>2.128173e-04</td>
      <td>6.483349e-05</td>
      <td>6.151345e-04</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.003277</td>
      <td>0.996559</td>
      <td>1.970047e-05</td>
      <td>1.753853e-05</td>
      <td>1.270773e-04</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.022765</td>
      <td>0.975087</td>
      <td>2.908912e-04</td>
      <td>4.438279e-05</td>
      <td>1.813432e-03</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.628636</td>
      <td>0.335183</td>
      <td>5.140227e-03</td>
      <td>7.765427e-05</td>
      <td>3.096247e-02</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.456359</td>
      <td>0.527099</td>
      <td>9.799633e-04</td>
      <td>3.779203e-05</td>
      <td>1.552406e-02</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.003381</td>
      <td>0.996086</td>
      <td>1.229097e-04</td>
      <td>3.323230e-05</td>
      <td>3.766246e-04</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.214557</td>
      <td>0.766768</td>
      <td>3.119119e-03</td>
      <td>9.305734e-05</td>
      <td>1.546322e-02</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.755188</td>
      <td>0.112618</td>
      <td>5.317152e-03</td>
      <td>1.750705e-04</td>
      <td>1.267016e-01</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.057193</td>
      <td>0.936839</td>
      <td>1.332142e-03</td>
      <td>6.606263e-05</td>
      <td>4.569109e-03</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.003129</td>
      <td>0.000773</td>
      <td>4.676589e-05</td>
      <td>2.220226e-05</td>
      <td>9.960296e-01</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.037648</td>
      <td>0.005585</td>
      <td>1.837620e-03</td>
      <td>1.612826e-04</td>
      <td>9.547681e-01</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.508140</td>
      <td>0.477954</td>
      <td>4.182220e-03</td>
      <td>2.533464e-05</td>
      <td>9.698475e-03</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.120749</td>
      <td>0.867692</td>
      <td>1.940413e-03</td>
      <td>2.010439e-04</td>
      <td>9.417455e-03</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.967495</td>
      <td>0.011674</td>
      <td>2.719503e-05</td>
      <td>1.396623e-06</td>
      <td>2.080238e-02</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0.001143</td>
      <td>0.000519</td>
      <td>1.304594e-04</td>
      <td>9.595477e-05</td>
      <td>9.981116e-01</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-96de3eee-a6f5-4551-b306-e41e5a1cb3f1')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-96de3eee-a6f5-4551-b306-e41e5a1cb3f1 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-96de3eee-a6f5-4551-b306-e41e5a1cb3f1');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
tf_model_predictions[0]
```




    <tf.Tensor: shape=(5,), dtype=float32, numpy=
    array([9.9650383e-01, 3.4869262e-03, 2.1181135e-08, 5.4149933e-09,
           9.3120052e-06], dtype=float32)>




```python
predicted_ids = np.argmax(tf_model_predictions, axis=-1)
predicted_labels = dataset_labels[predicted_ids]
```


```python
# Print images batch and labels predictions

plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(val_image_batch[n])
  color = "green" if predicted_ids[n] == true_label_ids[n] else "red"
  plt.title(predicted_labels[n].title(), color=color)
  plt.axis('off')
_ = plt.suptitle("Model predictions (green: correct, red: incorrect)")
```


    
![png](output_44_0.png)
    


## Convert model to TFLite

Convert recently loaded model to TensorFlow Lite models (standard and quantized with a [post-training quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)).

Because of TensorFlow 2.0 nature, we'll need to convert TensorFlow model into concrete function and then do conversion to TFLite. More about it [here](https://www.tensorflow.org/lite/r2/convert/concrete_function).

!mkdir "tflite_models"


```python
!mkdir "tflite_models"
```


```python
TFLITE_MODEL = "tflite_models/soundsep.tflite" 
TFLITE_QUANT_MODEL = "tflite_models/soundsep_quant.tflite"
```


```python
# Get the concrete function from the Keras model.
run_model = tf.function(lambda x : soundsep_model(x)) ## Keras 모델에서 구체적인 함수를 가져옴

# Save the concrete function.
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
)

# Convert the model
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converted_tflite_model = converter.convert()
open(TFLITE_MODEL, "wb").write(converted_tflite_model)

# Convert the model to quantized version with post-training quantization
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func]) ## 훈련 후 양자화를 통해 모델을 양자화 버전으로 변환
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
open(TFLITE_QUANT_MODEL, "wb").write(tflite_quant_model)

print("TFLite models and their sizes:")
# !ls "tflite_models" -lh  # 리눅스에서만 가능
```

    WARNING:absl:Please consider providing the trackable_obj argument in the from_concrete_functions. Providing without the trackable_obj argument is deprecated and it will use the deprecated conversion path.
    WARNING:absl:Please consider providing the trackable_obj argument in the from_concrete_functions. Providing without the trackable_obj argument is deprecated and it will use the deprecated conversion path.
    WARNING:absl:Optimization option OPTIMIZE_FOR_SIZE is deprecated, please use optimizations=[Optimize.DEFAULT] instead.
    WARNING:absl:Optimization option OPTIMIZE_FOR_SIZE is deprecated, please use optimizations=[Optimize.DEFAULT] instead.
    WARNING:absl:Optimization option OPTIMIZE_FOR_SIZE is deprecated, please use optimizations=[Optimize.DEFAULT] instead.
    

    TFLite models and their sizes:
    

### Load TFLite model

Load TensorFlow lite model with interpreter interface.

# Load TFLite model and see some details about input/output

tflite_interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)

input_details = tflite_interpreter.get_input_details()
output_details = tflite_interpreter.get_output_details()

print("== Input details ==")
print("name:", input_details[0]['name'])
print("shape:", input_details[0]['shape'])
print("type:", input_details[0]['dtype'])

print("\n== Output details ==")
print("name:", output_details[0]['name'])
print("shape:", output_details[0]['shape'])
print("type:", output_details[0]['dtype'])


```python
# Load TFLite model and see some details about input/output

tflite_interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL) ## TFLite 모델 로드

input_details = tflite_interpreter.get_input_details()
output_details = tflite_interpreter.get_output_details()

print("== Input details ==")
print("name:", input_details[0]['name'])
print("shape:", input_details[0]['shape']) ## 32로 되어야함
print("type:", input_details[0]['dtype'])

print("\n== Output details ==")
print("name:", output_details[0]['name'])
print("shape:", output_details[0]['shape']) ## 32로 되어야함
print("type:", output_details[0]['dtype'])
```

    == Input details ==
    name: x
    shape: [1 1 1 1]
    type: <class 'numpy.float32'>
    
    == Output details ==
    name: Identity
    shape: [1 5]
    type: <class 'numpy.float32'>
    

#### Resize input and output tensors shapes

Input shape of loaded TFLite model is 1x224x224x3, what means that we can make predictions for single image.

Let's resize input and output tensors, so we can make predictions for batch of 32 images.


```python
tflite_interpreter.resize_tensor_input(input_details[0]['index'], (32, 224, 224, num_color)) ## 32로 바꿔줌
tflite_interpreter.resize_tensor_input(output_details[0]['index'], (32, num_classes)) ## 32로 바꿔줌
tflite_interpreter.allocate_tensors()

input_details = tflite_interpreter.get_input_details()
output_details = tflite_interpreter.get_output_details()

print("== Input details ==")
print("name:", input_details[0]['name'])
print("shape:", input_details[0]['shape'])
print("type:", input_details[0]['dtype'])

print("\n== Output details ==")
print("name:", output_details[0]['name'])
print("shape:", output_details[0]['shape'])
print("type:", output_details[0]['dtype'])
```

    == Input details ==
    name: x
    shape: [ 32 224 224   3]
    type: <class 'numpy.float32'>
    
    == Output details ==
    name: Identity
    shape: [32  5]
    type: <class 'numpy.float32'>
    


```python
tflite_interpreter.set_tensor(input_details[0]['index'], val_image_batch)

tflite_interpreter.invoke()

tflite_model_predictions = tflite_interpreter.get_tensor(output_details[0]['index'])
#print("Prediction results shape:", tflite_model_predictions.shape)'
```

# Convert prediction results to Pandas dataframe, for better visualization

tflite_pred_dataframe = pd.DataFrame(tflite_model_predictions)
tflite_pred_dataframe.columns = dataset_labels

print("TFLite prediction results for the first elements")
tflite_pred_dataframe.head()


```python
# Convert prediction results to Pandas dataframe, for better visualization

tflite_pred_dataframe = pd.DataFrame(tflite_model_predictions) ## 더 나은 시각화를 위해Pandas 데이터 프레임으로 변환
tflite_pred_dataframe.columns = dataset_labels

print("TFLite prediction results for the first elements")
tflite_pred_dataframe.head()
```

    TFLite prediction results for the first elements
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Babbling</th>
      <th>Car_Noise</th>
      <th>Howling</th>
      <th>Voice</th>
      <th>White_Noise</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.328565</td>
      <td>0.659741</td>
      <td>0.000850</td>
      <td>0.009887</td>
      <td>0.000957</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.086753</td>
      <td>0.900019</td>
      <td>0.003624</td>
      <td>0.004102</td>
      <td>0.005503</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.999607</td>
      <td>0.000313</td>
      <td>0.000002</td>
      <td>0.000070</td>
      <td>0.000008</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.146890</td>
      <td>0.851376</td>
      <td>0.000202</td>
      <td>0.001320</td>
      <td>0.000212</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000018</td>
      <td>0.000044</td>
      <td>0.000090</td>
      <td>0.999844</td>
      <td>0.000004</td>
    </tr>
  </tbody>
</table>
</div>



Now let's do the same for TFLite quantized model:
- Load model,
- Reshape input to handle batch of images,
- Run prediction


```python
# Load quantized TFLite model
tflite_interpreter_quant = tf.lite.Interpreter(model_path=TFLITE_QUANT_MODEL)

# Learn about its input and output details
input_details = tflite_interpreter_quant.get_input_details()
output_details = tflite_interpreter_quant.get_output_details()

# Resize input and output tensors to handle batch of 32 images
tflite_interpreter_quant.resize_tensor_input(input_details[0]['index'], (32, 224, 224, 3))
tflite_interpreter_quant.resize_tensor_input(output_details[0]['index'], (32, 5))
tflite_interpreter_quant.allocate_tensors()

input_details = tflite_interpreter_quant.get_input_details()
output_details = tflite_interpreter_quant.get_output_details()

print("== Input details ==")
print("name:", input_details[0]['name'])
print("shape:", input_details[0]['shape'])
print("type:", input_details[0]['dtype'])

print("\n== Output details ==")
print("name:", output_details[0]['name'])
print("shape:", output_details[0]['shape'])
print("type:", output_details[0]['dtype'])

# Run inference
tflite_interpreter_quant.set_tensor(input_details[0]['index'], val_image_batch)

tflite_interpreter_quant.invoke()

tflite_q_model_predictions = tflite_interpreter_quant.get_tensor(output_details[0]['index'])
print("\nPrediction results shape:", tflite_q_model_predictions.shape)
```

    == Input details ==
    name: x
    shape: [ 32 224 224   3]
    type: <class 'numpy.float32'>
    
    == Output details ==
    name: Identity
    shape: [32  5]
    type: <class 'numpy.float32'>
    
    Prediction results shape: (32, 5)
    


```python
# Convert prediction results to Pandas dataframe, for better visualization

tflite_q_pred_dataframe = pd.DataFrame(tflite_q_model_predictions)
tflite_q_pred_dataframe.columns = dataset_labels

print("Quantized TFLite model prediction results for the first elements")
tflite_q_pred_dataframe.head()
```

    Quantized TFLite model prediction results for the first elements
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Babbling</th>
      <th>Car_Noise</th>
      <th>Howling</th>
      <th>Voice</th>
      <th>White_Noise</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.756865</td>
      <td>0.218985</td>
      <td>0.000736</td>
      <td>0.022934</td>
      <td>0.000480</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.697224</td>
      <td>0.289005</td>
      <td>0.001754</td>
      <td>0.008107</td>
      <td>0.003911</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.999588</td>
      <td>0.000301</td>
      <td>0.000003</td>
      <td>0.000101</td>
      <td>0.000008</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.789285</td>
      <td>0.208162</td>
      <td>0.000117</td>
      <td>0.002304</td>
      <td>0.000132</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000014</td>
      <td>0.000021</td>
      <td>0.000038</td>
      <td>0.999924</td>
      <td>0.000003</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Concatenate results from all models



## 예측 결과 비교
## Pandas를 사용하여 3가지 모델 모두의 결과를 시각화하고 차이점을 찾음


all_models_dataframe = pd.concat([tf_pred_dataframe, 
                                  tflite_pred_dataframe, 
                                  tflite_q_pred_dataframe], 
                                 keys=['TF Model', 'TFLite', 'TFLite quantized'],
                                 axis='columns')
all_models_dataframe.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="5" halign="left">TF Model</th>
      <th colspan="5" halign="left">TFLite</th>
      <th colspan="5" halign="left">TFLite quantized</th>
    </tr>
    <tr>
      <th></th>
      <th>Babbling</th>
      <th>Car_Noise</th>
      <th>Howling</th>
      <th>Voice</th>
      <th>White_Noise</th>
      <th>Babbling</th>
      <th>Car_Noise</th>
      <th>Howling</th>
      <th>Voice</th>
      <th>White_Noise</th>
      <th>Babbling</th>
      <th>Car_Noise</th>
      <th>Howling</th>
      <th>Voice</th>
      <th>White_Noise</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.326438</td>
      <td>0.661902</td>
      <td>0.000850</td>
      <td>0.009851</td>
      <td>0.000959</td>
      <td>0.328565</td>
      <td>0.659741</td>
      <td>0.000850</td>
      <td>0.009887</td>
      <td>0.000957</td>
      <td>0.756865</td>
      <td>0.218985</td>
      <td>0.000736</td>
      <td>0.022934</td>
      <td>0.000480</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.085018</td>
      <td>0.901815</td>
      <td>0.003631</td>
      <td>0.004064</td>
      <td>0.005471</td>
      <td>0.086753</td>
      <td>0.900019</td>
      <td>0.003624</td>
      <td>0.004102</td>
      <td>0.005503</td>
      <td>0.697224</td>
      <td>0.289005</td>
      <td>0.001754</td>
      <td>0.008107</td>
      <td>0.003911</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.999603</td>
      <td>0.000316</td>
      <td>0.000002</td>
      <td>0.000071</td>
      <td>0.000008</td>
      <td>0.999607</td>
      <td>0.000313</td>
      <td>0.000002</td>
      <td>0.000070</td>
      <td>0.000008</td>
      <td>0.999588</td>
      <td>0.000301</td>
      <td>0.000003</td>
      <td>0.000101</td>
      <td>0.000008</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.144312</td>
      <td>0.853968</td>
      <td>0.000201</td>
      <td>0.001308</td>
      <td>0.000212</td>
      <td>0.146890</td>
      <td>0.851376</td>
      <td>0.000202</td>
      <td>0.001320</td>
      <td>0.000212</td>
      <td>0.789285</td>
      <td>0.208162</td>
      <td>0.000117</td>
      <td>0.002304</td>
      <td>0.000132</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000018</td>
      <td>0.000045</td>
      <td>0.000091</td>
      <td>0.999843</td>
      <td>0.000004</td>
      <td>0.000018</td>
      <td>0.000044</td>
      <td>0.000090</td>
      <td>0.999844</td>
      <td>0.000004</td>
      <td>0.000014</td>
      <td>0.000021</td>
      <td>0.000038</td>
      <td>0.999924</td>
      <td>0.000003</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Swap columns to hava side by side comparison

all_models_dataframe = all_models_dataframe.swaplevel(axis='columns')[tflite_pred_dataframe.columns]
all_models_dataframe.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">Babbling</th>
      <th colspan="3" halign="left">Car_Noise</th>
      <th colspan="3" halign="left">Howling</th>
      <th colspan="3" halign="left">Voice</th>
      <th colspan="3" halign="left">White_Noise</th>
    </tr>
    <tr>
      <th></th>
      <th>TF Model</th>
      <th>TFLite</th>
      <th>TFLite quantized</th>
      <th>TF Model</th>
      <th>TFLite</th>
      <th>TFLite quantized</th>
      <th>TF Model</th>
      <th>TFLite</th>
      <th>TFLite quantized</th>
      <th>TF Model</th>
      <th>TFLite</th>
      <th>TFLite quantized</th>
      <th>TF Model</th>
      <th>TFLite</th>
      <th>TFLite quantized</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.326438</td>
      <td>0.328565</td>
      <td>0.756865</td>
      <td>0.661902</td>
      <td>0.659741</td>
      <td>0.218985</td>
      <td>0.000850</td>
      <td>0.000850</td>
      <td>0.000736</td>
      <td>0.009851</td>
      <td>0.009887</td>
      <td>0.022934</td>
      <td>0.000959</td>
      <td>0.000957</td>
      <td>0.000480</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.085018</td>
      <td>0.086753</td>
      <td>0.697224</td>
      <td>0.901815</td>
      <td>0.900019</td>
      <td>0.289005</td>
      <td>0.003631</td>
      <td>0.003624</td>
      <td>0.001754</td>
      <td>0.004064</td>
      <td>0.004102</td>
      <td>0.008107</td>
      <td>0.005471</td>
      <td>0.005503</td>
      <td>0.003911</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.999603</td>
      <td>0.999607</td>
      <td>0.999588</td>
      <td>0.000316</td>
      <td>0.000313</td>
      <td>0.000301</td>
      <td>0.000002</td>
      <td>0.000002</td>
      <td>0.000003</td>
      <td>0.000071</td>
      <td>0.000070</td>
      <td>0.000101</td>
      <td>0.000008</td>
      <td>0.000008</td>
      <td>0.000008</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.144312</td>
      <td>0.146890</td>
      <td>0.789285</td>
      <td>0.853968</td>
      <td>0.851376</td>
      <td>0.208162</td>
      <td>0.000201</td>
      <td>0.000202</td>
      <td>0.000117</td>
      <td>0.001308</td>
      <td>0.001320</td>
      <td>0.002304</td>
      <td>0.000212</td>
      <td>0.000212</td>
      <td>0.000132</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000018</td>
      <td>0.000018</td>
      <td>0.000014</td>
      <td>0.000045</td>
      <td>0.000044</td>
      <td>0.000021</td>
      <td>0.000091</td>
      <td>0.000090</td>
      <td>0.000038</td>
      <td>0.999843</td>
      <td>0.999844</td>
      <td>0.999924</td>
      <td>0.000004</td>
      <td>0.000004</td>
      <td>0.000003</td>
    </tr>
  </tbody>
</table>
</div>




```python

# Highlight TFLite models predictions that are different from original model

def highlight_diff(data, color='yellow'):
    attr = 'background-color: {}'.format(color)
    other = data.xs('TF Model', axis='columns', level=-1)
    return pd.DataFrame(np.where(data.ne(other, level=0), attr, ''),
                        index=data.index, columns=data.columns)

all_models_dataframe.style.apply(highlight_diff, axis=None)
```




<style type="text/css">
#T_9577f_row0_col1, #T_9577f_row0_col2, #T_9577f_row0_col4, #T_9577f_row0_col5, #T_9577f_row0_col7, #T_9577f_row0_col8, #T_9577f_row0_col10, #T_9577f_row0_col11, #T_9577f_row0_col13, #T_9577f_row0_col14, #T_9577f_row1_col1, #T_9577f_row1_col2, #T_9577f_row1_col4, #T_9577f_row1_col5, #T_9577f_row1_col7, #T_9577f_row1_col8, #T_9577f_row1_col10, #T_9577f_row1_col11, #T_9577f_row1_col13, #T_9577f_row1_col14, #T_9577f_row2_col1, #T_9577f_row2_col2, #T_9577f_row2_col4, #T_9577f_row2_col5, #T_9577f_row2_col7, #T_9577f_row2_col8, #T_9577f_row2_col10, #T_9577f_row2_col11, #T_9577f_row2_col13, #T_9577f_row2_col14, #T_9577f_row3_col1, #T_9577f_row3_col2, #T_9577f_row3_col4, #T_9577f_row3_col5, #T_9577f_row3_col7, #T_9577f_row3_col8, #T_9577f_row3_col10, #T_9577f_row3_col11, #T_9577f_row3_col13, #T_9577f_row3_col14, #T_9577f_row4_col1, #T_9577f_row4_col2, #T_9577f_row4_col4, #T_9577f_row4_col5, #T_9577f_row4_col7, #T_9577f_row4_col8, #T_9577f_row4_col10, #T_9577f_row4_col11, #T_9577f_row4_col13, #T_9577f_row4_col14, #T_9577f_row5_col1, #T_9577f_row5_col2, #T_9577f_row5_col4, #T_9577f_row5_col5, #T_9577f_row5_col7, #T_9577f_row5_col8, #T_9577f_row5_col10, #T_9577f_row5_col11, #T_9577f_row5_col13, #T_9577f_row5_col14, #T_9577f_row6_col1, #T_9577f_row6_col2, #T_9577f_row6_col4, #T_9577f_row6_col5, #T_9577f_row6_col7, #T_9577f_row6_col8, #T_9577f_row6_col10, #T_9577f_row6_col11, #T_9577f_row6_col13, #T_9577f_row6_col14, #T_9577f_row7_col1, #T_9577f_row7_col2, #T_9577f_row7_col4, #T_9577f_row7_col5, #T_9577f_row7_col7, #T_9577f_row7_col8, #T_9577f_row7_col10, #T_9577f_row7_col11, #T_9577f_row7_col13, #T_9577f_row7_col14, #T_9577f_row8_col1, #T_9577f_row8_col2, #T_9577f_row8_col4, #T_9577f_row8_col5, #T_9577f_row8_col7, #T_9577f_row8_col8, #T_9577f_row8_col10, #T_9577f_row8_col11, #T_9577f_row8_col13, #T_9577f_row8_col14, #T_9577f_row9_col1, #T_9577f_row9_col2, #T_9577f_row9_col4, #T_9577f_row9_col5, #T_9577f_row9_col7, #T_9577f_row9_col8, #T_9577f_row9_col10, #T_9577f_row9_col11, #T_9577f_row9_col13, #T_9577f_row9_col14, #T_9577f_row10_col1, #T_9577f_row10_col2, #T_9577f_row10_col4, #T_9577f_row10_col5, #T_9577f_row10_col7, #T_9577f_row10_col8, #T_9577f_row10_col10, #T_9577f_row10_col11, #T_9577f_row10_col13, #T_9577f_row10_col14, #T_9577f_row11_col1, #T_9577f_row11_col2, #T_9577f_row11_col4, #T_9577f_row11_col5, #T_9577f_row11_col7, #T_9577f_row11_col8, #T_9577f_row11_col10, #T_9577f_row11_col11, #T_9577f_row11_col13, #T_9577f_row11_col14, #T_9577f_row12_col1, #T_9577f_row12_col2, #T_9577f_row12_col4, #T_9577f_row12_col5, #T_9577f_row12_col7, #T_9577f_row12_col8, #T_9577f_row12_col10, #T_9577f_row12_col11, #T_9577f_row12_col13, #T_9577f_row12_col14, #T_9577f_row13_col1, #T_9577f_row13_col2, #T_9577f_row13_col4, #T_9577f_row13_col5, #T_9577f_row13_col7, #T_9577f_row13_col8, #T_9577f_row13_col10, #T_9577f_row13_col11, #T_9577f_row13_col13, #T_9577f_row13_col14, #T_9577f_row14_col1, #T_9577f_row14_col2, #T_9577f_row14_col4, #T_9577f_row14_col5, #T_9577f_row14_col7, #T_9577f_row14_col8, #T_9577f_row14_col10, #T_9577f_row14_col11, #T_9577f_row14_col13, #T_9577f_row14_col14, #T_9577f_row15_col1, #T_9577f_row15_col2, #T_9577f_row15_col4, #T_9577f_row15_col5, #T_9577f_row15_col7, #T_9577f_row15_col8, #T_9577f_row15_col10, #T_9577f_row15_col11, #T_9577f_row15_col13, #T_9577f_row15_col14, #T_9577f_row16_col1, #T_9577f_row16_col2, #T_9577f_row16_col4, #T_9577f_row16_col5, #T_9577f_row16_col7, #T_9577f_row16_col8, #T_9577f_row16_col10, #T_9577f_row16_col11, #T_9577f_row16_col13, #T_9577f_row16_col14, #T_9577f_row17_col1, #T_9577f_row17_col2, #T_9577f_row17_col4, #T_9577f_row17_col5, #T_9577f_row17_col7, #T_9577f_row17_col8, #T_9577f_row17_col10, #T_9577f_row17_col11, #T_9577f_row17_col13, #T_9577f_row17_col14, #T_9577f_row18_col1, #T_9577f_row18_col2, #T_9577f_row18_col4, #T_9577f_row18_col5, #T_9577f_row18_col7, #T_9577f_row18_col8, #T_9577f_row18_col10, #T_9577f_row18_col11, #T_9577f_row18_col13, #T_9577f_row18_col14, #T_9577f_row19_col1, #T_9577f_row19_col2, #T_9577f_row19_col4, #T_9577f_row19_col5, #T_9577f_row19_col7, #T_9577f_row19_col8, #T_9577f_row19_col10, #T_9577f_row19_col11, #T_9577f_row19_col13, #T_9577f_row19_col14, #T_9577f_row20_col1, #T_9577f_row20_col2, #T_9577f_row20_col4, #T_9577f_row20_col5, #T_9577f_row20_col7, #T_9577f_row20_col8, #T_9577f_row20_col10, #T_9577f_row20_col11, #T_9577f_row20_col13, #T_9577f_row20_col14, #T_9577f_row21_col1, #T_9577f_row21_col2, #T_9577f_row21_col4, #T_9577f_row21_col5, #T_9577f_row21_col7, #T_9577f_row21_col8, #T_9577f_row21_col10, #T_9577f_row21_col11, #T_9577f_row21_col13, #T_9577f_row21_col14, #T_9577f_row22_col1, #T_9577f_row22_col2, #T_9577f_row22_col4, #T_9577f_row22_col5, #T_9577f_row22_col7, #T_9577f_row22_col8, #T_9577f_row22_col10, #T_9577f_row22_col11, #T_9577f_row22_col13, #T_9577f_row22_col14, #T_9577f_row23_col1, #T_9577f_row23_col2, #T_9577f_row23_col4, #T_9577f_row23_col5, #T_9577f_row23_col7, #T_9577f_row23_col8, #T_9577f_row23_col10, #T_9577f_row23_col11, #T_9577f_row23_col13, #T_9577f_row23_col14, #T_9577f_row24_col1, #T_9577f_row24_col2, #T_9577f_row24_col4, #T_9577f_row24_col5, #T_9577f_row24_col7, #T_9577f_row24_col8, #T_9577f_row24_col10, #T_9577f_row24_col11, #T_9577f_row24_col13, #T_9577f_row24_col14, #T_9577f_row25_col1, #T_9577f_row25_col2, #T_9577f_row25_col4, #T_9577f_row25_col5, #T_9577f_row25_col7, #T_9577f_row25_col8, #T_9577f_row25_col10, #T_9577f_row25_col11, #T_9577f_row25_col13, #T_9577f_row25_col14, #T_9577f_row26_col1, #T_9577f_row26_col2, #T_9577f_row26_col4, #T_9577f_row26_col5, #T_9577f_row26_col7, #T_9577f_row26_col8, #T_9577f_row26_col10, #T_9577f_row26_col11, #T_9577f_row26_col13, #T_9577f_row26_col14, #T_9577f_row27_col1, #T_9577f_row27_col2, #T_9577f_row27_col4, #T_9577f_row27_col5, #T_9577f_row27_col7, #T_9577f_row27_col8, #T_9577f_row27_col10, #T_9577f_row27_col11, #T_9577f_row27_col13, #T_9577f_row27_col14, #T_9577f_row28_col1, #T_9577f_row28_col2, #T_9577f_row28_col4, #T_9577f_row28_col5, #T_9577f_row28_col7, #T_9577f_row28_col8, #T_9577f_row28_col10, #T_9577f_row28_col11, #T_9577f_row28_col13, #T_9577f_row28_col14, #T_9577f_row29_col1, #T_9577f_row29_col2, #T_9577f_row29_col4, #T_9577f_row29_col5, #T_9577f_row29_col7, #T_9577f_row29_col8, #T_9577f_row29_col10, #T_9577f_row29_col11, #T_9577f_row29_col13, #T_9577f_row29_col14, #T_9577f_row30_col1, #T_9577f_row30_col2, #T_9577f_row30_col4, #T_9577f_row30_col5, #T_9577f_row30_col7, #T_9577f_row30_col8, #T_9577f_row30_col10, #T_9577f_row30_col11, #T_9577f_row30_col13, #T_9577f_row30_col14, #T_9577f_row31_col1, #T_9577f_row31_col2, #T_9577f_row31_col4, #T_9577f_row31_col5, #T_9577f_row31_col7, #T_9577f_row31_col8, #T_9577f_row31_col10, #T_9577f_row31_col11, #T_9577f_row31_col13, #T_9577f_row31_col14 {
  background-color: yellow;
}
</style>
<table id="T_9577f">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_9577f_level0_col0" class="col_heading level0 col0" colspan="3">Babbling</th>
      <th id="T_9577f_level0_col3" class="col_heading level0 col3" colspan="3">Car_Noise</th>
      <th id="T_9577f_level0_col6" class="col_heading level0 col6" colspan="3">Howling</th>
      <th id="T_9577f_level0_col9" class="col_heading level0 col9" colspan="3">Voice</th>
      <th id="T_9577f_level0_col12" class="col_heading level0 col12" colspan="3">White_Noise</th>
    </tr>
    <tr>
      <th class="blank level1" >&nbsp;</th>
      <th id="T_9577f_level1_col0" class="col_heading level1 col0" >TF Model</th>
      <th id="T_9577f_level1_col1" class="col_heading level1 col1" >TFLite</th>
      <th id="T_9577f_level1_col2" class="col_heading level1 col2" >TFLite quantized</th>
      <th id="T_9577f_level1_col3" class="col_heading level1 col3" >TF Model</th>
      <th id="T_9577f_level1_col4" class="col_heading level1 col4" >TFLite</th>
      <th id="T_9577f_level1_col5" class="col_heading level1 col5" >TFLite quantized</th>
      <th id="T_9577f_level1_col6" class="col_heading level1 col6" >TF Model</th>
      <th id="T_9577f_level1_col7" class="col_heading level1 col7" >TFLite</th>
      <th id="T_9577f_level1_col8" class="col_heading level1 col8" >TFLite quantized</th>
      <th id="T_9577f_level1_col9" class="col_heading level1 col9" >TF Model</th>
      <th id="T_9577f_level1_col10" class="col_heading level1 col10" >TFLite</th>
      <th id="T_9577f_level1_col11" class="col_heading level1 col11" >TFLite quantized</th>
      <th id="T_9577f_level1_col12" class="col_heading level1 col12" >TF Model</th>
      <th id="T_9577f_level1_col13" class="col_heading level1 col13" >TFLite</th>
      <th id="T_9577f_level1_col14" class="col_heading level1 col14" >TFLite quantized</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_9577f_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_9577f_row0_col0" class="data row0 col0" >0.326438</td>
      <td id="T_9577f_row0_col1" class="data row0 col1" >0.328565</td>
      <td id="T_9577f_row0_col2" class="data row0 col2" >0.756865</td>
      <td id="T_9577f_row0_col3" class="data row0 col3" >0.661902</td>
      <td id="T_9577f_row0_col4" class="data row0 col4" >0.659741</td>
      <td id="T_9577f_row0_col5" class="data row0 col5" >0.218985</td>
      <td id="T_9577f_row0_col6" class="data row0 col6" >0.000850</td>
      <td id="T_9577f_row0_col7" class="data row0 col7" >0.000850</td>
      <td id="T_9577f_row0_col8" class="data row0 col8" >0.000736</td>
      <td id="T_9577f_row0_col9" class="data row0 col9" >0.009851</td>
      <td id="T_9577f_row0_col10" class="data row0 col10" >0.009887</td>
      <td id="T_9577f_row0_col11" class="data row0 col11" >0.022934</td>
      <td id="T_9577f_row0_col12" class="data row0 col12" >0.000959</td>
      <td id="T_9577f_row0_col13" class="data row0 col13" >0.000957</td>
      <td id="T_9577f_row0_col14" class="data row0 col14" >0.000480</td>
    </tr>
    <tr>
      <th id="T_9577f_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_9577f_row1_col0" class="data row1 col0" >0.085018</td>
      <td id="T_9577f_row1_col1" class="data row1 col1" >0.086753</td>
      <td id="T_9577f_row1_col2" class="data row1 col2" >0.697224</td>
      <td id="T_9577f_row1_col3" class="data row1 col3" >0.901815</td>
      <td id="T_9577f_row1_col4" class="data row1 col4" >0.900019</td>
      <td id="T_9577f_row1_col5" class="data row1 col5" >0.289005</td>
      <td id="T_9577f_row1_col6" class="data row1 col6" >0.003631</td>
      <td id="T_9577f_row1_col7" class="data row1 col7" >0.003624</td>
      <td id="T_9577f_row1_col8" class="data row1 col8" >0.001754</td>
      <td id="T_9577f_row1_col9" class="data row1 col9" >0.004064</td>
      <td id="T_9577f_row1_col10" class="data row1 col10" >0.004102</td>
      <td id="T_9577f_row1_col11" class="data row1 col11" >0.008107</td>
      <td id="T_9577f_row1_col12" class="data row1 col12" >0.005471</td>
      <td id="T_9577f_row1_col13" class="data row1 col13" >0.005503</td>
      <td id="T_9577f_row1_col14" class="data row1 col14" >0.003911</td>
    </tr>
    <tr>
      <th id="T_9577f_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_9577f_row2_col0" class="data row2 col0" >0.999603</td>
      <td id="T_9577f_row2_col1" class="data row2 col1" >0.999607</td>
      <td id="T_9577f_row2_col2" class="data row2 col2" >0.999588</td>
      <td id="T_9577f_row2_col3" class="data row2 col3" >0.000316</td>
      <td id="T_9577f_row2_col4" class="data row2 col4" >0.000313</td>
      <td id="T_9577f_row2_col5" class="data row2 col5" >0.000301</td>
      <td id="T_9577f_row2_col6" class="data row2 col6" >0.000002</td>
      <td id="T_9577f_row2_col7" class="data row2 col7" >0.000002</td>
      <td id="T_9577f_row2_col8" class="data row2 col8" >0.000003</td>
      <td id="T_9577f_row2_col9" class="data row2 col9" >0.000071</td>
      <td id="T_9577f_row2_col10" class="data row2 col10" >0.000070</td>
      <td id="T_9577f_row2_col11" class="data row2 col11" >0.000101</td>
      <td id="T_9577f_row2_col12" class="data row2 col12" >0.000008</td>
      <td id="T_9577f_row2_col13" class="data row2 col13" >0.000008</td>
      <td id="T_9577f_row2_col14" class="data row2 col14" >0.000008</td>
    </tr>
    <tr>
      <th id="T_9577f_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_9577f_row3_col0" class="data row3 col0" >0.144312</td>
      <td id="T_9577f_row3_col1" class="data row3 col1" >0.146890</td>
      <td id="T_9577f_row3_col2" class="data row3 col2" >0.789285</td>
      <td id="T_9577f_row3_col3" class="data row3 col3" >0.853968</td>
      <td id="T_9577f_row3_col4" class="data row3 col4" >0.851376</td>
      <td id="T_9577f_row3_col5" class="data row3 col5" >0.208162</td>
      <td id="T_9577f_row3_col6" class="data row3 col6" >0.000201</td>
      <td id="T_9577f_row3_col7" class="data row3 col7" >0.000202</td>
      <td id="T_9577f_row3_col8" class="data row3 col8" >0.000117</td>
      <td id="T_9577f_row3_col9" class="data row3 col9" >0.001308</td>
      <td id="T_9577f_row3_col10" class="data row3 col10" >0.001320</td>
      <td id="T_9577f_row3_col11" class="data row3 col11" >0.002304</td>
      <td id="T_9577f_row3_col12" class="data row3 col12" >0.000212</td>
      <td id="T_9577f_row3_col13" class="data row3 col13" >0.000212</td>
      <td id="T_9577f_row3_col14" class="data row3 col14" >0.000132</td>
    </tr>
    <tr>
      <th id="T_9577f_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_9577f_row4_col0" class="data row4 col0" >0.000018</td>
      <td id="T_9577f_row4_col1" class="data row4 col1" >0.000018</td>
      <td id="T_9577f_row4_col2" class="data row4 col2" >0.000014</td>
      <td id="T_9577f_row4_col3" class="data row4 col3" >0.000045</td>
      <td id="T_9577f_row4_col4" class="data row4 col4" >0.000044</td>
      <td id="T_9577f_row4_col5" class="data row4 col5" >0.000021</td>
      <td id="T_9577f_row4_col6" class="data row4 col6" >0.000091</td>
      <td id="T_9577f_row4_col7" class="data row4 col7" >0.000090</td>
      <td id="T_9577f_row4_col8" class="data row4 col8" >0.000038</td>
      <td id="T_9577f_row4_col9" class="data row4 col9" >0.999843</td>
      <td id="T_9577f_row4_col10" class="data row4 col10" >0.999844</td>
      <td id="T_9577f_row4_col11" class="data row4 col11" >0.999924</td>
      <td id="T_9577f_row4_col12" class="data row4 col12" >0.000004</td>
      <td id="T_9577f_row4_col13" class="data row4 col13" >0.000004</td>
      <td id="T_9577f_row4_col14" class="data row4 col14" >0.000003</td>
    </tr>
    <tr>
      <th id="T_9577f_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_9577f_row5_col0" class="data row5 col0" >0.334202</td>
      <td id="T_9577f_row5_col1" class="data row5 col1" >0.337468</td>
      <td id="T_9577f_row5_col2" class="data row5 col2" >0.572676</td>
      <td id="T_9577f_row5_col3" class="data row5 col3" >0.605221</td>
      <td id="T_9577f_row5_col4" class="data row5 col4" >0.601936</td>
      <td id="T_9577f_row5_col5" class="data row5 col5" >0.231093</td>
      <td id="T_9577f_row5_col6" class="data row5 col6" >0.008452</td>
      <td id="T_9577f_row5_col7" class="data row5 col7" >0.008434</td>
      <td id="T_9577f_row5_col8" class="data row5 col8" >0.009669</td>
      <td id="T_9577f_row5_col9" class="data row5 col9" >0.040733</td>
      <td id="T_9577f_row5_col10" class="data row5 col10" >0.040818</td>
      <td id="T_9577f_row5_col11" class="data row5 col11" >0.177342</td>
      <td id="T_9577f_row5_col12" class="data row5 col12" >0.011393</td>
      <td id="T_9577f_row5_col13" class="data row5 col13" >0.011344</td>
      <td id="T_9577f_row5_col14" class="data row5 col14" >0.009220</td>
    </tr>
    <tr>
      <th id="T_9577f_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_9577f_row6_col0" class="data row6 col0" >0.998934</td>
      <td id="T_9577f_row6_col1" class="data row6 col1" >0.998941</td>
      <td id="T_9577f_row6_col2" class="data row6 col2" >0.997762</td>
      <td id="T_9577f_row6_col3" class="data row6 col3" >0.000727</td>
      <td id="T_9577f_row6_col4" class="data row6 col4" >0.000722</td>
      <td id="T_9577f_row6_col5" class="data row6 col5" >0.001173</td>
      <td id="T_9577f_row6_col6" class="data row6 col6" >0.000011</td>
      <td id="T_9577f_row6_col7" class="data row6 col7" >0.000010</td>
      <td id="T_9577f_row6_col8" class="data row6 col8" >0.000031</td>
      <td id="T_9577f_row6_col9" class="data row6 col9" >0.000307</td>
      <td id="T_9577f_row6_col10" class="data row6 col10" >0.000305</td>
      <td id="T_9577f_row6_col11" class="data row6 col11" >0.000983</td>
      <td id="T_9577f_row6_col12" class="data row6 col12" >0.000021</td>
      <td id="T_9577f_row6_col13" class="data row6 col13" >0.000021</td>
      <td id="T_9577f_row6_col14" class="data row6 col14" >0.000052</td>
    </tr>
    <tr>
      <th id="T_9577f_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_9577f_row7_col0" class="data row7 col0" >0.986404</td>
      <td id="T_9577f_row7_col1" class="data row7 col1" >0.986593</td>
      <td id="T_9577f_row7_col2" class="data row7 col2" >0.974499</td>
      <td id="T_9577f_row7_col3" class="data row7 col3" >0.001415</td>
      <td id="T_9577f_row7_col4" class="data row7 col4" >0.001399</td>
      <td id="T_9577f_row7_col5" class="data row7 col5" >0.001610</td>
      <td id="T_9577f_row7_col6" class="data row7 col6" >0.000123</td>
      <td id="T_9577f_row7_col7" class="data row7 col7" >0.000121</td>
      <td id="T_9577f_row7_col8" class="data row7 col8" >0.000186</td>
      <td id="T_9577f_row7_col9" class="data row7 col9" >0.011944</td>
      <td id="T_9577f_row7_col10" class="data row7 col10" >0.011774</td>
      <td id="T_9577f_row7_col11" class="data row7 col11" >0.023565</td>
      <td id="T_9577f_row7_col12" class="data row7 col12" >0.000114</td>
      <td id="T_9577f_row7_col13" class="data row7 col13" >0.000112</td>
      <td id="T_9577f_row7_col14" class="data row7 col14" >0.000139</td>
    </tr>
    <tr>
      <th id="T_9577f_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_9577f_row8_col0" class="data row8 col0" >0.000040</td>
      <td id="T_9577f_row8_col1" class="data row8 col1" >0.000039</td>
      <td id="T_9577f_row8_col2" class="data row8 col2" >0.000024</td>
      <td id="T_9577f_row8_col3" class="data row8 col3" >0.000077</td>
      <td id="T_9577f_row8_col4" class="data row8 col4" >0.000076</td>
      <td id="T_9577f_row8_col5" class="data row8 col5" >0.000028</td>
      <td id="T_9577f_row8_col6" class="data row8 col6" >0.000267</td>
      <td id="T_9577f_row8_col7" class="data row8 col7" >0.000262</td>
      <td id="T_9577f_row8_col8" class="data row8 col8" >0.000077</td>
      <td id="T_9577f_row8_col9" class="data row8 col9" >0.999596</td>
      <td id="T_9577f_row8_col10" class="data row8 col10" >0.999602</td>
      <td id="T_9577f_row8_col11" class="data row8 col11" >0.999858</td>
      <td id="T_9577f_row8_col12" class="data row8 col12" >0.000020</td>
      <td id="T_9577f_row8_col13" class="data row8 col13" >0.000020</td>
      <td id="T_9577f_row8_col14" class="data row8 col14" >0.000012</td>
    </tr>
    <tr>
      <th id="T_9577f_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_9577f_row9_col0" class="data row9 col0" >0.000007</td>
      <td id="T_9577f_row9_col1" class="data row9 col1" >0.000007</td>
      <td id="T_9577f_row9_col2" class="data row9 col2" >0.000006</td>
      <td id="T_9577f_row9_col3" class="data row9 col3" >0.000037</td>
      <td id="T_9577f_row9_col4" class="data row9 col4" >0.000037</td>
      <td id="T_9577f_row9_col5" class="data row9 col5" >0.000024</td>
      <td id="T_9577f_row9_col6" class="data row9 col6" >0.000103</td>
      <td id="T_9577f_row9_col7" class="data row9 col7" >0.000102</td>
      <td id="T_9577f_row9_col8" class="data row9 col8" >0.000072</td>
      <td id="T_9577f_row9_col9" class="data row9 col9" >0.999840</td>
      <td id="T_9577f_row9_col10" class="data row9 col10" >0.999841</td>
      <td id="T_9577f_row9_col11" class="data row9 col11" >0.999889</td>
      <td id="T_9577f_row9_col12" class="data row9 col12" >0.000012</td>
      <td id="T_9577f_row9_col13" class="data row9 col13" >0.000012</td>
      <td id="T_9577f_row9_col14" class="data row9 col14" >0.000009</td>
    </tr>
    <tr>
      <th id="T_9577f_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_9577f_row10_col0" class="data row10 col0" >0.999842</td>
      <td id="T_9577f_row10_col1" class="data row10 col1" >0.999843</td>
      <td id="T_9577f_row10_col2" class="data row10 col2" >0.999763</td>
      <td id="T_9577f_row10_col3" class="data row10 col3" >0.000139</td>
      <td id="T_9577f_row10_col4" class="data row10 col4" >0.000138</td>
      <td id="T_9577f_row10_col5" class="data row10 col5" >0.000193</td>
      <td id="T_9577f_row10_col6" class="data row10 col6" >0.000000</td>
      <td id="T_9577f_row10_col7" class="data row10 col7" >0.000000</td>
      <td id="T_9577f_row10_col8" class="data row10 col8" >0.000001</td>
      <td id="T_9577f_row10_col9" class="data row10 col9" >0.000016</td>
      <td id="T_9577f_row10_col10" class="data row10 col10" >0.000016</td>
      <td id="T_9577f_row10_col11" class="data row10 col11" >0.000037</td>
      <td id="T_9577f_row10_col12" class="data row10 col12" >0.000003</td>
      <td id="T_9577f_row10_col13" class="data row10 col13" >0.000003</td>
      <td id="T_9577f_row10_col14" class="data row10 col14" >0.000006</td>
    </tr>
    <tr>
      <th id="T_9577f_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_9577f_row11_col0" class="data row11 col0" >0.000047</td>
      <td id="T_9577f_row11_col1" class="data row11 col1" >0.000047</td>
      <td id="T_9577f_row11_col2" class="data row11 col2" >0.000039</td>
      <td id="T_9577f_row11_col3" class="data row11 col3" >0.000084</td>
      <td id="T_9577f_row11_col4" class="data row11 col4" >0.000083</td>
      <td id="T_9577f_row11_col5" class="data row11 col5" >0.000027</td>
      <td id="T_9577f_row11_col6" class="data row11 col6" >0.000178</td>
      <td id="T_9577f_row11_col7" class="data row11 col7" >0.000176</td>
      <td id="T_9577f_row11_col8" class="data row11 col8" >0.000046</td>
      <td id="T_9577f_row11_col9" class="data row11 col9" >0.999670</td>
      <td id="T_9577f_row11_col10" class="data row11 col10" >0.999673</td>
      <td id="T_9577f_row11_col11" class="data row11 col11" >0.999875</td>
      <td id="T_9577f_row11_col12" class="data row11 col12" >0.000021</td>
      <td id="T_9577f_row11_col13" class="data row11 col13" >0.000021</td>
      <td id="T_9577f_row11_col14" class="data row11 col14" >0.000013</td>
    </tr>
    <tr>
      <th id="T_9577f_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_9577f_row12_col0" class="data row12 col0" >0.011601</td>
      <td id="T_9577f_row12_col1" class="data row12 col1" >0.011741</td>
      <td id="T_9577f_row12_col2" class="data row12 col2" >0.033350</td>
      <td id="T_9577f_row12_col3" class="data row12 col3" >0.026834</td>
      <td id="T_9577f_row12_col4" class="data row12 col4" >0.026875</td>
      <td id="T_9577f_row12_col5" class="data row12 col5" >0.046837</td>
      <td id="T_9577f_row12_col6" class="data row12 col6" >0.000520</td>
      <td id="T_9577f_row12_col7" class="data row12 col7" >0.000522</td>
      <td id="T_9577f_row12_col8" class="data row12 col8" >0.001477</td>
      <td id="T_9577f_row12_col9" class="data row12 col9" >0.000197</td>
      <td id="T_9577f_row12_col10" class="data row12 col10" >0.000199</td>
      <td id="T_9577f_row12_col11" class="data row12 col11" >0.001026</td>
      <td id="T_9577f_row12_col12" class="data row12 col12" >0.960848</td>
      <td id="T_9577f_row12_col13" class="data row12 col13" >0.960663</td>
      <td id="T_9577f_row12_col14" class="data row12 col14" >0.917310</td>
    </tr>
    <tr>
      <th id="T_9577f_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_9577f_row13_col0" class="data row13 col0" >0.999909</td>
      <td id="T_9577f_row13_col1" class="data row13 col1" >0.999910</td>
      <td id="T_9577f_row13_col2" class="data row13 col2" >0.999850</td>
      <td id="T_9577f_row13_col3" class="data row13 col3" >0.000041</td>
      <td id="T_9577f_row13_col4" class="data row13 col4" >0.000041</td>
      <td id="T_9577f_row13_col5" class="data row13 col5" >0.000050</td>
      <td id="T_9577f_row13_col6" class="data row13 col6" >0.000000</td>
      <td id="T_9577f_row13_col7" class="data row13 col7" >0.000000</td>
      <td id="T_9577f_row13_col8" class="data row13 col8" >0.000000</td>
      <td id="T_9577f_row13_col9" class="data row13 col9" >0.000049</td>
      <td id="T_9577f_row13_col10" class="data row13 col10" >0.000049</td>
      <td id="T_9577f_row13_col11" class="data row13 col11" >0.000100</td>
      <td id="T_9577f_row13_col12" class="data row13 col12" >0.000000</td>
      <td id="T_9577f_row13_col13" class="data row13 col13" >0.000000</td>
      <td id="T_9577f_row13_col14" class="data row13 col14" >0.000000</td>
    </tr>
    <tr>
      <th id="T_9577f_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_9577f_row14_col0" class="data row14 col0" >0.020672</td>
      <td id="T_9577f_row14_col1" class="data row14 col1" >0.020726</td>
      <td id="T_9577f_row14_col2" class="data row14 col2" >0.039804</td>
      <td id="T_9577f_row14_col3" class="data row14 col3" >0.300943</td>
      <td id="T_9577f_row14_col4" class="data row14 col4" >0.300160</td>
      <td id="T_9577f_row14_col5" class="data row14 col5" >0.265811</td>
      <td id="T_9577f_row14_col6" class="data row14 col6" >0.010572</td>
      <td id="T_9577f_row14_col7" class="data row14 col7" >0.010540</td>
      <td id="T_9577f_row14_col8" class="data row14 col8" >0.011109</td>
      <td id="T_9577f_row14_col9" class="data row14 col9" >0.004654</td>
      <td id="T_9577f_row14_col10" class="data row14 col10" >0.004670</td>
      <td id="T_9577f_row14_col11" class="data row14 col11" >0.015244</td>
      <td id="T_9577f_row14_col12" class="data row14 col12" >0.663159</td>
      <td id="T_9577f_row14_col13" class="data row14 col13" >0.663904</td>
      <td id="T_9577f_row14_col14" class="data row14 col14" >0.668032</td>
    </tr>
    <tr>
      <th id="T_9577f_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_9577f_row15_col0" class="data row15 col0" >0.000075</td>
      <td id="T_9577f_row15_col1" class="data row15 col1" >0.000075</td>
      <td id="T_9577f_row15_col2" class="data row15 col2" >0.000037</td>
      <td id="T_9577f_row15_col3" class="data row15 col3" >0.000310</td>
      <td id="T_9577f_row15_col4" class="data row15 col4" >0.000306</td>
      <td id="T_9577f_row15_col5" class="data row15 col5" >0.000108</td>
      <td id="T_9577f_row15_col6" class="data row15 col6" >0.000892</td>
      <td id="T_9577f_row15_col7" class="data row15 col7" >0.000880</td>
      <td id="T_9577f_row15_col8" class="data row15 col8" >0.000304</td>
      <td id="T_9577f_row15_col9" class="data row15 col9" >0.998627</td>
      <td id="T_9577f_row15_col10" class="data row15 col10" >0.998643</td>
      <td id="T_9577f_row15_col11" class="data row15 col11" >0.999501</td>
      <td id="T_9577f_row15_col12" class="data row15 col12" >0.000096</td>
      <td id="T_9577f_row15_col13" class="data row15 col13" >0.000096</td>
      <td id="T_9577f_row15_col14" class="data row15 col14" >0.000050</td>
    </tr>
    <tr>
      <th id="T_9577f_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_9577f_row16_col0" class="data row16 col0" >0.929916</td>
      <td id="T_9577f_row16_col1" class="data row16 col1" >0.930613</td>
      <td id="T_9577f_row16_col2" class="data row16 col2" >0.984311</td>
      <td id="T_9577f_row16_col3" class="data row16 col3" >0.067100</td>
      <td id="T_9577f_row16_col4" class="data row16 col4" >0.066421</td>
      <td id="T_9577f_row16_col5" class="data row16 col5" >0.013671</td>
      <td id="T_9577f_row16_col6" class="data row16 col6" >0.000241</td>
      <td id="T_9577f_row16_col7" class="data row16 col7" >0.000239</td>
      <td id="T_9577f_row16_col8" class="data row16 col8" >0.000092</td>
      <td id="T_9577f_row16_col9" class="data row16 col9" >0.002372</td>
      <td id="T_9577f_row16_col10" class="data row16 col10" >0.002360</td>
      <td id="T_9577f_row16_col11" class="data row16 col11" >0.001813</td>
      <td id="T_9577f_row16_col12" class="data row16 col12" >0.000371</td>
      <td id="T_9577f_row16_col13" class="data row16 col13" >0.000367</td>
      <td id="T_9577f_row16_col14" class="data row16 col14" >0.000114</td>
    </tr>
    <tr>
      <th id="T_9577f_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_9577f_row17_col0" class="data row17 col0" >0.000000</td>
      <td id="T_9577f_row17_col1" class="data row17 col1" >0.000000</td>
      <td id="T_9577f_row17_col2" class="data row17 col2" >0.000000</td>
      <td id="T_9577f_row17_col3" class="data row17 col3" >0.000008</td>
      <td id="T_9577f_row17_col4" class="data row17 col4" >0.000008</td>
      <td id="T_9577f_row17_col5" class="data row17 col5" >0.000009</td>
      <td id="T_9577f_row17_col6" class="data row17 col6" >0.999992</td>
      <td id="T_9577f_row17_col7" class="data row17 col7" >0.999992</td>
      <td id="T_9577f_row17_col8" class="data row17 col8" >0.999991</td>
      <td id="T_9577f_row17_col9" class="data row17 col9" >0.000000</td>
      <td id="T_9577f_row17_col10" class="data row17 col10" >0.000000</td>
      <td id="T_9577f_row17_col11" class="data row17 col11" >0.000000</td>
      <td id="T_9577f_row17_col12" class="data row17 col12" >0.000000</td>
      <td id="T_9577f_row17_col13" class="data row17 col13" >0.000000</td>
      <td id="T_9577f_row17_col14" class="data row17 col14" >0.000000</td>
    </tr>
    <tr>
      <th id="T_9577f_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_9577f_row18_col0" class="data row18 col0" >0.987711</td>
      <td id="T_9577f_row18_col1" class="data row18 col1" >0.987820</td>
      <td id="T_9577f_row18_col2" class="data row18 col2" >0.985856</td>
      <td id="T_9577f_row18_col3" class="data row18 col3" >0.010972</td>
      <td id="T_9577f_row18_col4" class="data row18 col4" >0.010868</td>
      <td id="T_9577f_row18_col5" class="data row18 col5" >0.011941</td>
      <td id="T_9577f_row18_col6" class="data row18 col6" >0.000092</td>
      <td id="T_9577f_row18_col7" class="data row18 col7" >0.000092</td>
      <td id="T_9577f_row18_col8" class="data row18 col8" >0.000156</td>
      <td id="T_9577f_row18_col9" class="data row18 col9" >0.000981</td>
      <td id="T_9577f_row18_col10" class="data row18 col10" >0.000978</td>
      <td id="T_9577f_row18_col11" class="data row18 col11" >0.001625</td>
      <td id="T_9577f_row18_col12" class="data row18 col12" >0.000244</td>
      <td id="T_9577f_row18_col13" class="data row18 col13" >0.000242</td>
      <td id="T_9577f_row18_col14" class="data row18 col14" >0.000422</td>
    </tr>
    <tr>
      <th id="T_9577f_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_9577f_row19_col0" class="data row19 col0" >0.000001</td>
      <td id="T_9577f_row19_col1" class="data row19 col1" >0.000001</td>
      <td id="T_9577f_row19_col2" class="data row19 col2" >0.000001</td>
      <td id="T_9577f_row19_col3" class="data row19 col3" >0.000003</td>
      <td id="T_9577f_row19_col4" class="data row19 col4" >0.000003</td>
      <td id="T_9577f_row19_col5" class="data row19 col5" >0.000002</td>
      <td id="T_9577f_row19_col6" class="data row19 col6" >0.000012</td>
      <td id="T_9577f_row19_col7" class="data row19 col7" >0.000011</td>
      <td id="T_9577f_row19_col8" class="data row19 col8" >0.000007</td>
      <td id="T_9577f_row19_col9" class="data row19 col9" >0.999984</td>
      <td id="T_9577f_row19_col10" class="data row19 col10" >0.999984</td>
      <td id="T_9577f_row19_col11" class="data row19 col11" >0.999989</td>
      <td id="T_9577f_row19_col12" class="data row19 col12" >0.000001</td>
      <td id="T_9577f_row19_col13" class="data row19 col13" >0.000001</td>
      <td id="T_9577f_row19_col14" class="data row19 col14" >0.000001</td>
    </tr>
    <tr>
      <th id="T_9577f_level0_row20" class="row_heading level0 row20" >20</th>
      <td id="T_9577f_row20_col0" class="data row20 col0" >0.499295</td>
      <td id="T_9577f_row20_col1" class="data row20 col1" >0.504845</td>
      <td id="T_9577f_row20_col2" class="data row20 col2" >0.549568</td>
      <td id="T_9577f_row20_col3" class="data row20 col3" >0.497049</td>
      <td id="T_9577f_row20_col4" class="data row20 col4" >0.491498</td>
      <td id="T_9577f_row20_col5" class="data row20 col5" >0.444141</td>
      <td id="T_9577f_row20_col6" class="data row20 col6" >0.000136</td>
      <td id="T_9577f_row20_col7" class="data row20 col7" >0.000135</td>
      <td id="T_9577f_row20_col8" class="data row20 col8" >0.000160</td>
      <td id="T_9577f_row20_col9" class="data row20 col9" >0.003458</td>
      <td id="T_9577f_row20_col10" class="data row20 col10" >0.003461</td>
      <td id="T_9577f_row20_col11" class="data row20 col11" >0.006079</td>
      <td id="T_9577f_row20_col12" class="data row20 col12" >0.000061</td>
      <td id="T_9577f_row20_col13" class="data row20 col13" >0.000061</td>
      <td id="T_9577f_row20_col14" class="data row20 col14" >0.000052</td>
    </tr>
    <tr>
      <th id="T_9577f_level0_row21" class="row_heading level0 row21" >21</th>
      <td id="T_9577f_row21_col0" class="data row21 col0" >0.300814</td>
      <td id="T_9577f_row21_col1" class="data row21 col1" >0.302605</td>
      <td id="T_9577f_row21_col2" class="data row21 col2" >0.127053</td>
      <td id="T_9577f_row21_col3" class="data row21 col3" >0.698281</td>
      <td id="T_9577f_row21_col4" class="data row21 col4" >0.696490</td>
      <td id="T_9577f_row21_col5" class="data row21 col5" >0.872176</td>
      <td id="T_9577f_row21_col6" class="data row21 col6" >0.000095</td>
      <td id="T_9577f_row21_col7" class="data row21 col7" >0.000095</td>
      <td id="T_9577f_row21_col8" class="data row21 col8" >0.000086</td>
      <td id="T_9577f_row21_col9" class="data row21 col9" >0.000476</td>
      <td id="T_9577f_row21_col10" class="data row21 col10" >0.000478</td>
      <td id="T_9577f_row21_col11" class="data row21 col11" >0.000416</td>
      <td id="T_9577f_row21_col12" class="data row21 col12" >0.000334</td>
      <td id="T_9577f_row21_col13" class="data row21 col13" >0.000332</td>
      <td id="T_9577f_row21_col14" class="data row21 col14" >0.000270</td>
    </tr>
    <tr>
      <th id="T_9577f_level0_row22" class="row_heading level0 row22" >22</th>
      <td id="T_9577f_row22_col0" class="data row22 col0" >0.007566</td>
      <td id="T_9577f_row22_col1" class="data row22 col1" >0.007675</td>
      <td id="T_9577f_row22_col2" class="data row22 col2" >0.008837</td>
      <td id="T_9577f_row22_col3" class="data row22 col3" >0.992177</td>
      <td id="T_9577f_row22_col4" class="data row22 col4" >0.992065</td>
      <td id="T_9577f_row22_col5" class="data row22 col5" >0.990482</td>
      <td id="T_9577f_row22_col6" class="data row22 col6" >0.000034</td>
      <td id="T_9577f_row22_col7" class="data row22 col7" >0.000035</td>
      <td id="T_9577f_row22_col8" class="data row22 col8" >0.000080</td>
      <td id="T_9577f_row22_col9" class="data row22 col9" >0.000130</td>
      <td id="T_9577f_row22_col10" class="data row22 col10" >0.000131</td>
      <td id="T_9577f_row22_col11" class="data row22 col11" >0.000490</td>
      <td id="T_9577f_row22_col12" class="data row22 col12" >0.000093</td>
      <td id="T_9577f_row22_col13" class="data row22 col13" >0.000094</td>
      <td id="T_9577f_row22_col14" class="data row22 col14" >0.000112</td>
    </tr>
    <tr>
      <th id="T_9577f_level0_row23" class="row_heading level0 row23" >23</th>
      <td id="T_9577f_row23_col0" class="data row23 col0" >0.699777</td>
      <td id="T_9577f_row23_col1" class="data row23 col1" >0.702576</td>
      <td id="T_9577f_row23_col2" class="data row23 col2" >0.970618</td>
      <td id="T_9577f_row23_col3" class="data row23 col3" >0.291226</td>
      <td id="T_9577f_row23_col4" class="data row23 col4" >0.288466</td>
      <td id="T_9577f_row23_col5" class="data row23 col5" >0.026521</td>
      <td id="T_9577f_row23_col6" class="data row23 col6" >0.001129</td>
      <td id="T_9577f_row23_col7" class="data row23 col7" >0.001123</td>
      <td id="T_9577f_row23_col8" class="data row23 col8" >0.000255</td>
      <td id="T_9577f_row23_col9" class="data row23 col9" >0.002879</td>
      <td id="T_9577f_row23_col10" class="data row23 col10" >0.002869</td>
      <td id="T_9577f_row23_col11" class="data row23 col11" >0.001734</td>
      <td id="T_9577f_row23_col12" class="data row23 col12" >0.004989</td>
      <td id="T_9577f_row23_col13" class="data row23 col13" >0.004966</td>
      <td id="T_9577f_row23_col14" class="data row23 col14" >0.000873</td>
    </tr>
    <tr>
      <th id="T_9577f_level0_row24" class="row_heading level0 row24" >24</th>
      <td id="T_9577f_row24_col0" class="data row24 col0" >0.000102</td>
      <td id="T_9577f_row24_col1" class="data row24 col1" >0.000101</td>
      <td id="T_9577f_row24_col2" class="data row24 col2" >0.000044</td>
      <td id="T_9577f_row24_col3" class="data row24 col3" >0.000460</td>
      <td id="T_9577f_row24_col4" class="data row24 col4" >0.000454</td>
      <td id="T_9577f_row24_col5" class="data row24 col5" >0.000108</td>
      <td id="T_9577f_row24_col6" class="data row24 col6" >0.001099</td>
      <td id="T_9577f_row24_col7" class="data row24 col7" >0.001082</td>
      <td id="T_9577f_row24_col8" class="data row24 col8" >0.000203</td>
      <td id="T_9577f_row24_col9" class="data row24 col9" >0.998289</td>
      <td id="T_9577f_row24_col10" class="data row24 col10" >0.998314</td>
      <td id="T_9577f_row24_col11" class="data row24 col11" >0.999619</td>
      <td id="T_9577f_row24_col12" class="data row24 col12" >0.000050</td>
      <td id="T_9577f_row24_col13" class="data row24 col13" >0.000049</td>
      <td id="T_9577f_row24_col14" class="data row24 col14" >0.000026</td>
    </tr>
    <tr>
      <th id="T_9577f_level0_row25" class="row_heading level0 row25" >25</th>
      <td id="T_9577f_row25_col0" class="data row25 col0" >0.000000</td>
      <td id="T_9577f_row25_col1" class="data row25 col1" >0.000000</td>
      <td id="T_9577f_row25_col2" class="data row25 col2" >0.000000</td>
      <td id="T_9577f_row25_col3" class="data row25 col3" >0.000024</td>
      <td id="T_9577f_row25_col4" class="data row25 col4" >0.000025</td>
      <td id="T_9577f_row25_col5" class="data row25 col5" >0.000044</td>
      <td id="T_9577f_row25_col6" class="data row25 col6" >0.999975</td>
      <td id="T_9577f_row25_col7" class="data row25 col7" >0.999975</td>
      <td id="T_9577f_row25_col8" class="data row25 col8" >0.999956</td>
      <td id="T_9577f_row25_col9" class="data row25 col9" >0.000000</td>
      <td id="T_9577f_row25_col10" class="data row25 col10" >0.000000</td>
      <td id="T_9577f_row25_col11" class="data row25 col11" >0.000000</td>
      <td id="T_9577f_row25_col12" class="data row25 col12" >0.000000</td>
      <td id="T_9577f_row25_col13" class="data row25 col13" >0.000000</td>
      <td id="T_9577f_row25_col14" class="data row25 col14" >0.000000</td>
    </tr>
    <tr>
      <th id="T_9577f_level0_row26" class="row_heading level0 row26" >26</th>
      <td id="T_9577f_row26_col0" class="data row26 col0" >0.000011</td>
      <td id="T_9577f_row26_col1" class="data row26 col1" >0.000011</td>
      <td id="T_9577f_row26_col2" class="data row26 col2" >0.000004</td>
      <td id="T_9577f_row26_col3" class="data row26 col3" >0.000058</td>
      <td id="T_9577f_row26_col4" class="data row26 col4" >0.000057</td>
      <td id="T_9577f_row26_col5" class="data row26 col5" >0.000011</td>
      <td id="T_9577f_row26_col6" class="data row26 col6" >0.000193</td>
      <td id="T_9577f_row26_col7" class="data row26 col7" >0.000191</td>
      <td id="T_9577f_row26_col8" class="data row26 col8" >0.000031</td>
      <td id="T_9577f_row26_col9" class="data row26 col9" >0.999725</td>
      <td id="T_9577f_row26_col10" class="data row26 col10" >0.999728</td>
      <td id="T_9577f_row26_col11" class="data row26 col11" >0.999949</td>
      <td id="T_9577f_row26_col12" class="data row26 col12" >0.000013</td>
      <td id="T_9577f_row26_col13" class="data row26 col13" >0.000013</td>
      <td id="T_9577f_row26_col14" class="data row26 col14" >0.000005</td>
    </tr>
    <tr>
      <th id="T_9577f_level0_row27" class="row_heading level0 row27" >27</th>
      <td id="T_9577f_row27_col0" class="data row27 col0" >0.000010</td>
      <td id="T_9577f_row27_col1" class="data row27 col1" >0.000010</td>
      <td id="T_9577f_row27_col2" class="data row27 col2" >0.000008</td>
      <td id="T_9577f_row27_col3" class="data row27 col3" >0.000056</td>
      <td id="T_9577f_row27_col4" class="data row27 col4" >0.000055</td>
      <td id="T_9577f_row27_col5" class="data row27 col5" >0.000031</td>
      <td id="T_9577f_row27_col6" class="data row27 col6" >0.000200</td>
      <td id="T_9577f_row27_col7" class="data row27 col7" >0.000197</td>
      <td id="T_9577f_row27_col8" class="data row27 col8" >0.000100</td>
      <td id="T_9577f_row27_col9" class="data row27 col9" >0.999725</td>
      <td id="T_9577f_row27_col10" class="data row27 col10" >0.999728</td>
      <td id="T_9577f_row27_col11" class="data row27 col11" >0.999850</td>
      <td id="T_9577f_row27_col12" class="data row27 col12" >0.000009</td>
      <td id="T_9577f_row27_col13" class="data row27 col13" >0.000009</td>
      <td id="T_9577f_row27_col14" class="data row27 col14" >0.000010</td>
    </tr>
    <tr>
      <th id="T_9577f_level0_row28" class="row_heading level0 row28" >28</th>
      <td id="T_9577f_row28_col0" class="data row28 col0" >0.000000</td>
      <td id="T_9577f_row28_col1" class="data row28 col1" >0.000000</td>
      <td id="T_9577f_row28_col2" class="data row28 col2" >0.000000</td>
      <td id="T_9577f_row28_col3" class="data row28 col3" >0.000314</td>
      <td id="T_9577f_row28_col4" class="data row28 col4" >0.000317</td>
      <td id="T_9577f_row28_col5" class="data row28 col5" >0.000509</td>
      <td id="T_9577f_row28_col6" class="data row28 col6" >0.999686</td>
      <td id="T_9577f_row28_col7" class="data row28 col7" >0.999683</td>
      <td id="T_9577f_row28_col8" class="data row28 col8" >0.999490</td>
      <td id="T_9577f_row28_col9" class="data row28 col9" >0.000000</td>
      <td id="T_9577f_row28_col10" class="data row28 col10" >0.000000</td>
      <td id="T_9577f_row28_col11" class="data row28 col11" >0.000001</td>
      <td id="T_9577f_row28_col12" class="data row28 col12" >0.000000</td>
      <td id="T_9577f_row28_col13" class="data row28 col13" >0.000000</td>
      <td id="T_9577f_row28_col14" class="data row28 col14" >0.000000</td>
    </tr>
    <tr>
      <th id="T_9577f_level0_row29" class="row_heading level0 row29" >29</th>
      <td id="T_9577f_row29_col0" class="data row29 col0" >0.813665</td>
      <td id="T_9577f_row29_col1" class="data row29 col1" >0.814995</td>
      <td id="T_9577f_row29_col2" class="data row29 col2" >0.825261</td>
      <td id="T_9577f_row29_col3" class="data row29 col3" >0.185230</td>
      <td id="T_9577f_row29_col4" class="data row29 col4" >0.183900</td>
      <td id="T_9577f_row29_col5" class="data row29 col5" >0.173346</td>
      <td id="T_9577f_row29_col6" class="data row29 col6" >0.000065</td>
      <td id="T_9577f_row29_col7" class="data row29 col7" >0.000065</td>
      <td id="T_9577f_row29_col8" class="data row29 col8" >0.000071</td>
      <td id="T_9577f_row29_col9" class="data row29 col9" >0.000901</td>
      <td id="T_9577f_row29_col10" class="data row29 col10" >0.000902</td>
      <td id="T_9577f_row29_col11" class="data row29 col11" >0.001178</td>
      <td id="T_9577f_row29_col12" class="data row29 col12" >0.000139</td>
      <td id="T_9577f_row29_col13" class="data row29 col13" >0.000139</td>
      <td id="T_9577f_row29_col14" class="data row29 col14" >0.000144</td>
    </tr>
    <tr>
      <th id="T_9577f_level0_row30" class="row_heading level0 row30" >30</th>
      <td id="T_9577f_row30_col0" class="data row30 col0" >0.000140</td>
      <td id="T_9577f_row30_col1" class="data row30 col1" >0.000139</td>
      <td id="T_9577f_row30_col2" class="data row30 col2" >0.000067</td>
      <td id="T_9577f_row30_col3" class="data row30 col3" >0.001023</td>
      <td id="T_9577f_row30_col4" class="data row30 col4" >0.001013</td>
      <td id="T_9577f_row30_col5" class="data row30 col5" >0.000395</td>
      <td id="T_9577f_row30_col6" class="data row30 col6" >0.004468</td>
      <td id="T_9577f_row30_col7" class="data row30 col7" >0.004416</td>
      <td id="T_9577f_row30_col8" class="data row30 col8" >0.001678</td>
      <td id="T_9577f_row30_col9" class="data row30 col9" >0.994274</td>
      <td id="T_9577f_row30_col10" class="data row30 col10" >0.994337</td>
      <td id="T_9577f_row30_col11" class="data row30 col11" >0.997808</td>
      <td id="T_9577f_row30_col12" class="data row30 col12" >0.000095</td>
      <td id="T_9577f_row30_col13" class="data row30 col13" >0.000095</td>
      <td id="T_9577f_row30_col14" class="data row30 col14" >0.000052</td>
    </tr>
    <tr>
      <th id="T_9577f_level0_row31" class="row_heading level0 row31" >31</th>
      <td id="T_9577f_row31_col0" class="data row31 col0" >0.999057</td>
      <td id="T_9577f_row31_col1" class="data row31 col1" >0.999065</td>
      <td id="T_9577f_row31_col2" class="data row31 col2" >0.999131</td>
      <td id="T_9577f_row31_col3" class="data row31 col3" >0.000814</td>
      <td id="T_9577f_row31_col4" class="data row31 col4" >0.000806</td>
      <td id="T_9577f_row31_col5" class="data row31 col5" >0.000721</td>
      <td id="T_9577f_row31_col6" class="data row31 col6" >0.000003</td>
      <td id="T_9577f_row31_col7" class="data row31 col7" >0.000003</td>
      <td id="T_9577f_row31_col8" class="data row31 col8" >0.000004</td>
      <td id="T_9577f_row31_col9" class="data row31 col9" >0.000057</td>
      <td id="T_9577f_row31_col10" class="data row31 col10" >0.000057</td>
      <td id="T_9577f_row31_col11" class="data row31 col11" >0.000085</td>
      <td id="T_9577f_row31_col12" class="data row31 col12" >0.000069</td>
      <td id="T_9577f_row31_col13" class="data row31 col13" >0.000069</td>
      <td id="T_9577f_row31_col14" class="data row31 col14" >0.000058</td>
    </tr>
  </tbody>
</table>




As we can see, in most cases predictions are different between all models, usually by small factors. High-confidence predictions between TensorFlow and TensorFlow Lite models are very close to each other (in some cases there are even similar).  
Quantized model outstands the most, but this is the cost of optimizations (model weights 3-4 times less).

To make prediction results even more readable, let's simplify dataframes, to show only the highest-score prediction and the corresponding label.

# Concatenation of argmax and max value for each row
def max_values_only(data):
  argmax_col = np.argmax(data, axis=1).reshape(-1, 1)
  max_col = np.max(data, axis=1).reshape(-1, 1)
  return np.concatenate([argmax_col, max_col], axis=1)

# Build simplified prediction tables
tf_model_pred_simplified = max_values_only(tf_model_predictions)
tflite_model_pred_simplified = max_values_only(tflite_model_predictions)
tflite_q_model_pred_simplified = max_values_only(tflite_q_model_predictions)


```python
# Concatenation of argmax and max value for each row
def max_values_only(data):
  argmax_col = np.argmax(data, axis=1).reshape(-1, 1)
  max_col = np.max(data, axis=1).reshape(-1, 1)
  return np.concatenate([argmax_col, max_col], axis=1)

# Build simplified prediction tables
tf_model_pred_simplified = max_values_only(tf_model_predictions)
tflite_model_pred_simplified = max_values_only(tflite_model_predictions)
tflite_q_model_pred_simplified = max_values_only(tflite_q_model_predictions)
```

# Build DataFrames and present example
columns_names = ["Label_id", "Confidence"]
tf_model_simple_dataframe = pd.DataFrame(tf_model_pred_simplified)
tf_model_simple_dataframe.columns = columns_names

tflite_model_simple_dataframe = pd.DataFrame(tflite_model_pred_simplified)
tflite_model_simple_dataframe.columns = columns_names

tflite_q_model_simple_dataframe = pd.DataFrame(tflite_q_model_pred_simplified)
tflite_q_model_simple_dataframe.columns = columns_names

tf_model_simple_dataframe.head()


```python
# Build DataFrames and present example
columns_names = ["Label_id", "Confidence"]
tf_model_simple_dataframe = pd.DataFrame(tf_model_pred_simplified)
tf_model_simple_dataframe.columns = columns_names

tflite_model_simple_dataframe = pd.DataFrame(tflite_model_pred_simplified)
tflite_model_simple_dataframe.columns = columns_names

tflite_q_model_simple_dataframe = pd.DataFrame(tflite_q_model_pred_simplified)
tflite_q_model_simple_dataframe.columns = columns_names

tf_model_simple_dataframe.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label_id</th>
      <th>Confidence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.661902</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.901815</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.999603</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0.853968</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.0</td>
      <td>0.999843</td>
    </tr>
  </tbody>
</table>
</div>



# Concatenate results from all models
all_models_simple_dataframe = pd.concat([tf_model_simple_dataframe, 
                                         tflite_model_simple_dataframe, 
                                         tflite_q_model_simple_dataframe], 
                                        keys=['TF Model', 'TFLite', 'TFLite quantized'],
                                        axis='columns')

# Swap columns for side-by-side comparison
all_models_simple_dataframe = all_models_simple_dataframe.swaplevel(axis='columns')[tf_model_simple_dataframe.columns]

# Highlight differences
all_models_simple_dataframe.style.apply(highlight_diff, axis=None)


```python
# Concatenate results from all models
all_models_simple_dataframe = pd.concat([tf_model_simple_dataframe, 
                                         tflite_model_simple_dataframe, 
                                         tflite_q_model_simple_dataframe], 
                                        keys=['TF Model', 'TFLite', 'TFLite quantized'],
                                        axis='columns')

# Swap columns for side-by-side comparison
all_models_simple_dataframe = all_models_simple_dataframe.swaplevel(axis='columns')[tf_model_simple_dataframe.columns]

# Highlight differences
all_models_simple_dataframe.style.apply(highlight_diff, axis=None)
```




<style type="text/css">
#T_0755b_row0_col2, #T_0755b_row0_col4, #T_0755b_row0_col5, #T_0755b_row1_col2, #T_0755b_row1_col4, #T_0755b_row1_col5, #T_0755b_row2_col4, #T_0755b_row2_col5, #T_0755b_row3_col2, #T_0755b_row3_col4, #T_0755b_row3_col5, #T_0755b_row4_col4, #T_0755b_row4_col5, #T_0755b_row5_col2, #T_0755b_row5_col4, #T_0755b_row5_col5, #T_0755b_row6_col4, #T_0755b_row6_col5, #T_0755b_row7_col4, #T_0755b_row7_col5, #T_0755b_row8_col4, #T_0755b_row8_col5, #T_0755b_row9_col4, #T_0755b_row9_col5, #T_0755b_row10_col4, #T_0755b_row10_col5, #T_0755b_row11_col4, #T_0755b_row11_col5, #T_0755b_row12_col4, #T_0755b_row12_col5, #T_0755b_row13_col4, #T_0755b_row13_col5, #T_0755b_row14_col4, #T_0755b_row14_col5, #T_0755b_row15_col4, #T_0755b_row15_col5, #T_0755b_row16_col4, #T_0755b_row16_col5, #T_0755b_row17_col4, #T_0755b_row17_col5, #T_0755b_row18_col4, #T_0755b_row18_col5, #T_0755b_row19_col4, #T_0755b_row19_col5, #T_0755b_row20_col4, #T_0755b_row20_col5, #T_0755b_row21_col4, #T_0755b_row21_col5, #T_0755b_row22_col4, #T_0755b_row22_col5, #T_0755b_row23_col4, #T_0755b_row23_col5, #T_0755b_row24_col4, #T_0755b_row24_col5, #T_0755b_row25_col4, #T_0755b_row25_col5, #T_0755b_row26_col4, #T_0755b_row26_col5, #T_0755b_row27_col4, #T_0755b_row27_col5, #T_0755b_row28_col4, #T_0755b_row28_col5, #T_0755b_row29_col4, #T_0755b_row29_col5, #T_0755b_row30_col4, #T_0755b_row30_col5, #T_0755b_row31_col4, #T_0755b_row31_col5 {
  background-color: yellow;
}
</style>
<table id="T_0755b">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_0755b_level0_col0" class="col_heading level0 col0" colspan="3">Label_id</th>
      <th id="T_0755b_level0_col3" class="col_heading level0 col3" colspan="3">Confidence</th>
    </tr>
    <tr>
      <th class="blank level1" >&nbsp;</th>
      <th id="T_0755b_level1_col0" class="col_heading level1 col0" >TF Model</th>
      <th id="T_0755b_level1_col1" class="col_heading level1 col1" >TFLite</th>
      <th id="T_0755b_level1_col2" class="col_heading level1 col2" >TFLite quantized</th>
      <th id="T_0755b_level1_col3" class="col_heading level1 col3" >TF Model</th>
      <th id="T_0755b_level1_col4" class="col_heading level1 col4" >TFLite</th>
      <th id="T_0755b_level1_col5" class="col_heading level1 col5" >TFLite quantized</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_0755b_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_0755b_row0_col0" class="data row0 col0" >1.000000</td>
      <td id="T_0755b_row0_col1" class="data row0 col1" >1.000000</td>
      <td id="T_0755b_row0_col2" class="data row0 col2" >0.000000</td>
      <td id="T_0755b_row0_col3" class="data row0 col3" >0.661902</td>
      <td id="T_0755b_row0_col4" class="data row0 col4" >0.659741</td>
      <td id="T_0755b_row0_col5" class="data row0 col5" >0.756865</td>
    </tr>
    <tr>
      <th id="T_0755b_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_0755b_row1_col0" class="data row1 col0" >1.000000</td>
      <td id="T_0755b_row1_col1" class="data row1 col1" >1.000000</td>
      <td id="T_0755b_row1_col2" class="data row1 col2" >0.000000</td>
      <td id="T_0755b_row1_col3" class="data row1 col3" >0.901815</td>
      <td id="T_0755b_row1_col4" class="data row1 col4" >0.900019</td>
      <td id="T_0755b_row1_col5" class="data row1 col5" >0.697224</td>
    </tr>
    <tr>
      <th id="T_0755b_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_0755b_row2_col0" class="data row2 col0" >0.000000</td>
      <td id="T_0755b_row2_col1" class="data row2 col1" >0.000000</td>
      <td id="T_0755b_row2_col2" class="data row2 col2" >0.000000</td>
      <td id="T_0755b_row2_col3" class="data row2 col3" >0.999603</td>
      <td id="T_0755b_row2_col4" class="data row2 col4" >0.999607</td>
      <td id="T_0755b_row2_col5" class="data row2 col5" >0.999588</td>
    </tr>
    <tr>
      <th id="T_0755b_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_0755b_row3_col0" class="data row3 col0" >1.000000</td>
      <td id="T_0755b_row3_col1" class="data row3 col1" >1.000000</td>
      <td id="T_0755b_row3_col2" class="data row3 col2" >0.000000</td>
      <td id="T_0755b_row3_col3" class="data row3 col3" >0.853968</td>
      <td id="T_0755b_row3_col4" class="data row3 col4" >0.851376</td>
      <td id="T_0755b_row3_col5" class="data row3 col5" >0.789285</td>
    </tr>
    <tr>
      <th id="T_0755b_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_0755b_row4_col0" class="data row4 col0" >3.000000</td>
      <td id="T_0755b_row4_col1" class="data row4 col1" >3.000000</td>
      <td id="T_0755b_row4_col2" class="data row4 col2" >3.000000</td>
      <td id="T_0755b_row4_col3" class="data row4 col3" >0.999843</td>
      <td id="T_0755b_row4_col4" class="data row4 col4" >0.999844</td>
      <td id="T_0755b_row4_col5" class="data row4 col5" >0.999924</td>
    </tr>
    <tr>
      <th id="T_0755b_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_0755b_row5_col0" class="data row5 col0" >1.000000</td>
      <td id="T_0755b_row5_col1" class="data row5 col1" >1.000000</td>
      <td id="T_0755b_row5_col2" class="data row5 col2" >0.000000</td>
      <td id="T_0755b_row5_col3" class="data row5 col3" >0.605221</td>
      <td id="T_0755b_row5_col4" class="data row5 col4" >0.601936</td>
      <td id="T_0755b_row5_col5" class="data row5 col5" >0.572676</td>
    </tr>
    <tr>
      <th id="T_0755b_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_0755b_row6_col0" class="data row6 col0" >0.000000</td>
      <td id="T_0755b_row6_col1" class="data row6 col1" >0.000000</td>
      <td id="T_0755b_row6_col2" class="data row6 col2" >0.000000</td>
      <td id="T_0755b_row6_col3" class="data row6 col3" >0.998934</td>
      <td id="T_0755b_row6_col4" class="data row6 col4" >0.998941</td>
      <td id="T_0755b_row6_col5" class="data row6 col5" >0.997762</td>
    </tr>
    <tr>
      <th id="T_0755b_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_0755b_row7_col0" class="data row7 col0" >0.000000</td>
      <td id="T_0755b_row7_col1" class="data row7 col1" >0.000000</td>
      <td id="T_0755b_row7_col2" class="data row7 col2" >0.000000</td>
      <td id="T_0755b_row7_col3" class="data row7 col3" >0.986404</td>
      <td id="T_0755b_row7_col4" class="data row7 col4" >0.986593</td>
      <td id="T_0755b_row7_col5" class="data row7 col5" >0.974499</td>
    </tr>
    <tr>
      <th id="T_0755b_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_0755b_row8_col0" class="data row8 col0" >3.000000</td>
      <td id="T_0755b_row8_col1" class="data row8 col1" >3.000000</td>
      <td id="T_0755b_row8_col2" class="data row8 col2" >3.000000</td>
      <td id="T_0755b_row8_col3" class="data row8 col3" >0.999596</td>
      <td id="T_0755b_row8_col4" class="data row8 col4" >0.999602</td>
      <td id="T_0755b_row8_col5" class="data row8 col5" >0.999858</td>
    </tr>
    <tr>
      <th id="T_0755b_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_0755b_row9_col0" class="data row9 col0" >3.000000</td>
      <td id="T_0755b_row9_col1" class="data row9 col1" >3.000000</td>
      <td id="T_0755b_row9_col2" class="data row9 col2" >3.000000</td>
      <td id="T_0755b_row9_col3" class="data row9 col3" >0.999840</td>
      <td id="T_0755b_row9_col4" class="data row9 col4" >0.999841</td>
      <td id="T_0755b_row9_col5" class="data row9 col5" >0.999889</td>
    </tr>
    <tr>
      <th id="T_0755b_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_0755b_row10_col0" class="data row10 col0" >0.000000</td>
      <td id="T_0755b_row10_col1" class="data row10 col1" >0.000000</td>
      <td id="T_0755b_row10_col2" class="data row10 col2" >0.000000</td>
      <td id="T_0755b_row10_col3" class="data row10 col3" >0.999842</td>
      <td id="T_0755b_row10_col4" class="data row10 col4" >0.999843</td>
      <td id="T_0755b_row10_col5" class="data row10 col5" >0.999763</td>
    </tr>
    <tr>
      <th id="T_0755b_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_0755b_row11_col0" class="data row11 col0" >3.000000</td>
      <td id="T_0755b_row11_col1" class="data row11 col1" >3.000000</td>
      <td id="T_0755b_row11_col2" class="data row11 col2" >3.000000</td>
      <td id="T_0755b_row11_col3" class="data row11 col3" >0.999670</td>
      <td id="T_0755b_row11_col4" class="data row11 col4" >0.999673</td>
      <td id="T_0755b_row11_col5" class="data row11 col5" >0.999875</td>
    </tr>
    <tr>
      <th id="T_0755b_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_0755b_row12_col0" class="data row12 col0" >4.000000</td>
      <td id="T_0755b_row12_col1" class="data row12 col1" >4.000000</td>
      <td id="T_0755b_row12_col2" class="data row12 col2" >4.000000</td>
      <td id="T_0755b_row12_col3" class="data row12 col3" >0.960848</td>
      <td id="T_0755b_row12_col4" class="data row12 col4" >0.960663</td>
      <td id="T_0755b_row12_col5" class="data row12 col5" >0.917310</td>
    </tr>
    <tr>
      <th id="T_0755b_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_0755b_row13_col0" class="data row13 col0" >0.000000</td>
      <td id="T_0755b_row13_col1" class="data row13 col1" >0.000000</td>
      <td id="T_0755b_row13_col2" class="data row13 col2" >0.000000</td>
      <td id="T_0755b_row13_col3" class="data row13 col3" >0.999909</td>
      <td id="T_0755b_row13_col4" class="data row13 col4" >0.999910</td>
      <td id="T_0755b_row13_col5" class="data row13 col5" >0.999850</td>
    </tr>
    <tr>
      <th id="T_0755b_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_0755b_row14_col0" class="data row14 col0" >4.000000</td>
      <td id="T_0755b_row14_col1" class="data row14 col1" >4.000000</td>
      <td id="T_0755b_row14_col2" class="data row14 col2" >4.000000</td>
      <td id="T_0755b_row14_col3" class="data row14 col3" >0.663159</td>
      <td id="T_0755b_row14_col4" class="data row14 col4" >0.663904</td>
      <td id="T_0755b_row14_col5" class="data row14 col5" >0.668032</td>
    </tr>
    <tr>
      <th id="T_0755b_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_0755b_row15_col0" class="data row15 col0" >3.000000</td>
      <td id="T_0755b_row15_col1" class="data row15 col1" >3.000000</td>
      <td id="T_0755b_row15_col2" class="data row15 col2" >3.000000</td>
      <td id="T_0755b_row15_col3" class="data row15 col3" >0.998627</td>
      <td id="T_0755b_row15_col4" class="data row15 col4" >0.998643</td>
      <td id="T_0755b_row15_col5" class="data row15 col5" >0.999501</td>
    </tr>
    <tr>
      <th id="T_0755b_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_0755b_row16_col0" class="data row16 col0" >0.000000</td>
      <td id="T_0755b_row16_col1" class="data row16 col1" >0.000000</td>
      <td id="T_0755b_row16_col2" class="data row16 col2" >0.000000</td>
      <td id="T_0755b_row16_col3" class="data row16 col3" >0.929916</td>
      <td id="T_0755b_row16_col4" class="data row16 col4" >0.930613</td>
      <td id="T_0755b_row16_col5" class="data row16 col5" >0.984311</td>
    </tr>
    <tr>
      <th id="T_0755b_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_0755b_row17_col0" class="data row17 col0" >2.000000</td>
      <td id="T_0755b_row17_col1" class="data row17 col1" >2.000000</td>
      <td id="T_0755b_row17_col2" class="data row17 col2" >2.000000</td>
      <td id="T_0755b_row17_col3" class="data row17 col3" >0.999992</td>
      <td id="T_0755b_row17_col4" class="data row17 col4" >0.999992</td>
      <td id="T_0755b_row17_col5" class="data row17 col5" >0.999991</td>
    </tr>
    <tr>
      <th id="T_0755b_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_0755b_row18_col0" class="data row18 col0" >0.000000</td>
      <td id="T_0755b_row18_col1" class="data row18 col1" >0.000000</td>
      <td id="T_0755b_row18_col2" class="data row18 col2" >0.000000</td>
      <td id="T_0755b_row18_col3" class="data row18 col3" >0.987711</td>
      <td id="T_0755b_row18_col4" class="data row18 col4" >0.987820</td>
      <td id="T_0755b_row18_col5" class="data row18 col5" >0.985856</td>
    </tr>
    <tr>
      <th id="T_0755b_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_0755b_row19_col0" class="data row19 col0" >3.000000</td>
      <td id="T_0755b_row19_col1" class="data row19 col1" >3.000000</td>
      <td id="T_0755b_row19_col2" class="data row19 col2" >3.000000</td>
      <td id="T_0755b_row19_col3" class="data row19 col3" >0.999984</td>
      <td id="T_0755b_row19_col4" class="data row19 col4" >0.999984</td>
      <td id="T_0755b_row19_col5" class="data row19 col5" >0.999989</td>
    </tr>
    <tr>
      <th id="T_0755b_level0_row20" class="row_heading level0 row20" >20</th>
      <td id="T_0755b_row20_col0" class="data row20 col0" >0.000000</td>
      <td id="T_0755b_row20_col1" class="data row20 col1" >0.000000</td>
      <td id="T_0755b_row20_col2" class="data row20 col2" >0.000000</td>
      <td id="T_0755b_row20_col3" class="data row20 col3" >0.499295</td>
      <td id="T_0755b_row20_col4" class="data row20 col4" >0.504845</td>
      <td id="T_0755b_row20_col5" class="data row20 col5" >0.549568</td>
    </tr>
    <tr>
      <th id="T_0755b_level0_row21" class="row_heading level0 row21" >21</th>
      <td id="T_0755b_row21_col0" class="data row21 col0" >1.000000</td>
      <td id="T_0755b_row21_col1" class="data row21 col1" >1.000000</td>
      <td id="T_0755b_row21_col2" class="data row21 col2" >1.000000</td>
      <td id="T_0755b_row21_col3" class="data row21 col3" >0.698281</td>
      <td id="T_0755b_row21_col4" class="data row21 col4" >0.696490</td>
      <td id="T_0755b_row21_col5" class="data row21 col5" >0.872176</td>
    </tr>
    <tr>
      <th id="T_0755b_level0_row22" class="row_heading level0 row22" >22</th>
      <td id="T_0755b_row22_col0" class="data row22 col0" >1.000000</td>
      <td id="T_0755b_row22_col1" class="data row22 col1" >1.000000</td>
      <td id="T_0755b_row22_col2" class="data row22 col2" >1.000000</td>
      <td id="T_0755b_row22_col3" class="data row22 col3" >0.992177</td>
      <td id="T_0755b_row22_col4" class="data row22 col4" >0.992065</td>
      <td id="T_0755b_row22_col5" class="data row22 col5" >0.990482</td>
    </tr>
    <tr>
      <th id="T_0755b_level0_row23" class="row_heading level0 row23" >23</th>
      <td id="T_0755b_row23_col0" class="data row23 col0" >0.000000</td>
      <td id="T_0755b_row23_col1" class="data row23 col1" >0.000000</td>
      <td id="T_0755b_row23_col2" class="data row23 col2" >0.000000</td>
      <td id="T_0755b_row23_col3" class="data row23 col3" >0.699777</td>
      <td id="T_0755b_row23_col4" class="data row23 col4" >0.702576</td>
      <td id="T_0755b_row23_col5" class="data row23 col5" >0.970618</td>
    </tr>
    <tr>
      <th id="T_0755b_level0_row24" class="row_heading level0 row24" >24</th>
      <td id="T_0755b_row24_col0" class="data row24 col0" >3.000000</td>
      <td id="T_0755b_row24_col1" class="data row24 col1" >3.000000</td>
      <td id="T_0755b_row24_col2" class="data row24 col2" >3.000000</td>
      <td id="T_0755b_row24_col3" class="data row24 col3" >0.998289</td>
      <td id="T_0755b_row24_col4" class="data row24 col4" >0.998314</td>
      <td id="T_0755b_row24_col5" class="data row24 col5" >0.999619</td>
    </tr>
    <tr>
      <th id="T_0755b_level0_row25" class="row_heading level0 row25" >25</th>
      <td id="T_0755b_row25_col0" class="data row25 col0" >2.000000</td>
      <td id="T_0755b_row25_col1" class="data row25 col1" >2.000000</td>
      <td id="T_0755b_row25_col2" class="data row25 col2" >2.000000</td>
      <td id="T_0755b_row25_col3" class="data row25 col3" >0.999975</td>
      <td id="T_0755b_row25_col4" class="data row25 col4" >0.999975</td>
      <td id="T_0755b_row25_col5" class="data row25 col5" >0.999956</td>
    </tr>
    <tr>
      <th id="T_0755b_level0_row26" class="row_heading level0 row26" >26</th>
      <td id="T_0755b_row26_col0" class="data row26 col0" >3.000000</td>
      <td id="T_0755b_row26_col1" class="data row26 col1" >3.000000</td>
      <td id="T_0755b_row26_col2" class="data row26 col2" >3.000000</td>
      <td id="T_0755b_row26_col3" class="data row26 col3" >0.999725</td>
      <td id="T_0755b_row26_col4" class="data row26 col4" >0.999728</td>
      <td id="T_0755b_row26_col5" class="data row26 col5" >0.999949</td>
    </tr>
    <tr>
      <th id="T_0755b_level0_row27" class="row_heading level0 row27" >27</th>
      <td id="T_0755b_row27_col0" class="data row27 col0" >3.000000</td>
      <td id="T_0755b_row27_col1" class="data row27 col1" >3.000000</td>
      <td id="T_0755b_row27_col2" class="data row27 col2" >3.000000</td>
      <td id="T_0755b_row27_col3" class="data row27 col3" >0.999725</td>
      <td id="T_0755b_row27_col4" class="data row27 col4" >0.999728</td>
      <td id="T_0755b_row27_col5" class="data row27 col5" >0.999850</td>
    </tr>
    <tr>
      <th id="T_0755b_level0_row28" class="row_heading level0 row28" >28</th>
      <td id="T_0755b_row28_col0" class="data row28 col0" >2.000000</td>
      <td id="T_0755b_row28_col1" class="data row28 col1" >2.000000</td>
      <td id="T_0755b_row28_col2" class="data row28 col2" >2.000000</td>
      <td id="T_0755b_row28_col3" class="data row28 col3" >0.999686</td>
      <td id="T_0755b_row28_col4" class="data row28 col4" >0.999683</td>
      <td id="T_0755b_row28_col5" class="data row28 col5" >0.999490</td>
    </tr>
    <tr>
      <th id="T_0755b_level0_row29" class="row_heading level0 row29" >29</th>
      <td id="T_0755b_row29_col0" class="data row29 col0" >0.000000</td>
      <td id="T_0755b_row29_col1" class="data row29 col1" >0.000000</td>
      <td id="T_0755b_row29_col2" class="data row29 col2" >0.000000</td>
      <td id="T_0755b_row29_col3" class="data row29 col3" >0.813665</td>
      <td id="T_0755b_row29_col4" class="data row29 col4" >0.814995</td>
      <td id="T_0755b_row29_col5" class="data row29 col5" >0.825261</td>
    </tr>
    <tr>
      <th id="T_0755b_level0_row30" class="row_heading level0 row30" >30</th>
      <td id="T_0755b_row30_col0" class="data row30 col0" >3.000000</td>
      <td id="T_0755b_row30_col1" class="data row30 col1" >3.000000</td>
      <td id="T_0755b_row30_col2" class="data row30 col2" >3.000000</td>
      <td id="T_0755b_row30_col3" class="data row30 col3" >0.994274</td>
      <td id="T_0755b_row30_col4" class="data row30 col4" >0.994337</td>
      <td id="T_0755b_row30_col5" class="data row30 col5" >0.997808</td>
    </tr>
    <tr>
      <th id="T_0755b_level0_row31" class="row_heading level0 row31" >31</th>
      <td id="T_0755b_row31_col0" class="data row31 col0" >0.000000</td>
      <td id="T_0755b_row31_col1" class="data row31 col1" >0.000000</td>
      <td id="T_0755b_row31_col2" class="data row31 col2" >0.000000</td>
      <td id="T_0755b_row31_col3" class="data row31 col3" >0.999057</td>
      <td id="T_0755b_row31_col4" class="data row31 col4" >0.999065</td>
      <td id="T_0755b_row31_col5" class="data row31 col5" >0.999131</td>
    </tr>
  </tbody>
</table>




## Visualize predictions from TFLite models

At the end let's visualize predictions from TensorFlow Lite and quantized TensorFlow Lite models.

# Print images batch and labels predictions for TFLite Model

tflite_predicted_ids = np.argmax(tflite_model_predictions, axis=-1)
tflite_predicted_labels = dataset_labels[tflite_predicted_ids]
tflite_label_id = np.argmax(val_label_batch, axis=-1)

plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(val_image_batch[n])
  color = "green" if tflite_predicted_ids[n] == true_label_ids[n] else "red"
  plt.title(tflite_predicted_labels[n].title(), color=color)
  plt.axis('off')
_ = plt.suptitle("TFLite model predictions (green: correct, red: incorrect)")


```python
# Print images batch and labels predictions for TFLite Model

# TF 모델 예측

tflite_predicted_ids = np.argmax(tflite_model_predictions, axis=-1)
tflite_predicted_labels = dataset_labels[tflite_predicted_ids]
tflite_label_id = np.argmax(val_label_batch, axis=-1)

plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(val_image_batch[n])

  # 총 개수가 분모로 가고 분자에는 green이 오는 개수

  if tflite_predicted_ids[n] == true_label_ids[n]:
    color = "green"
    correct += 1

  else:
    color = "red"

print(correct / )



 # color = "green" if tflite_predicted_ids[n] == true_label_ids[n] else "red"
  plt.title(tflite_predicted_labels[n].title(), color=color)
  plt.axis('off')
_ = plt.suptitle("TFLite model predictions (green: correct, red: incorrect)")

## 이미지가 대부분 car_noise가 나오는 이유가 무엇일까
## 인식을 잘 못 하는 거 같음 왜 인식을 못하는지 해결해야할듯함
## 색이 들어가면 예측을 잘 못 하는 거 같음
```


    
![png](output_74_0.png)
    



```python

```


```python
# Print images batch and labels predictions for TFLite Model

# TF 모델 예측

tflite_predicted_ids = np.argmax(tflite_model_predictions, axis=-1)
tflite_predicted_labels = dataset_labels[tflite_predicted_ids]
tflite_label_id = np.argmax(val_label_batch, axis=-1)

plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(5000):
  # 총 개수가 분모로 가고 분자에는 green이 오는 개수
  correct = 0
  if tflite_predicted_ids[n] == true_label_ids[n]:
    color = "green"
    correct += 1
  else:
    color = "red"

print(correct / len(tflite_predicted_labels))



 # color = "green" if tflite_predicted_ids[n] == true_label_ids[n] else "red"
  plt.title(tflite_predicted_labels[n].title(), color=color)
  plt.axis('off')
_ = plt.suptitle("TFLite model predictions (green: correct, red: incorrect)")

## 이미지가 대부분 car_noise가 나오는 이유가 무엇일까
## 인식을 잘 못 하는 거 같음 왜 인식을 못하는지 해결해야할듯함
## 색이 들어가면 예측을 잘 못 하는 거 같음
```


```python
final_loss, final_accuracy = tflite_interpreter.evaluate(feature_test, label_test)
```

# Print images batch and labels predictions for TFLite Model

tflite_q_predicted_ids = np.argmax(tflite_q_model_predictions, axis=-1)
tflite_q_predicted_labels = dataset_labels[tflite_q_predicted_ids]
tflite_q_label_id = np.argmax(val_label_batch, axis=-1)

plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(val_image_batch[n])
  color = "green" if tflite_q_predicted_ids[n] == true_label_ids[n] else "red"
  plt.title(tflite_q_predicted_labels[n].title(), color=color)
  plt.axis('off')
_ = plt.suptitle("Quantized TFLite model predictions (green: correct, red: incorrect)")


```python
# Print images batch and labels predictions for TFLite Model

# 양자화된 TF 모델 예측

tflite_q_predicted_ids = np.argmax(tflite_q_model_predictions, axis=-1)
tflite_q_predicted_labels = dataset_labels[tflite_q_predicted_ids]
tflite_q_label_id = np.argmax(val_label_batch, axis=-1)

plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(val_image_batch[n])
  color = "green" if tflite_q_predicted_ids[n] == true_label_ids[n] else "red"
  plt.title(tflite_q_predicted_labels[n].title(), color=color)
  plt.axis('off')
_ = plt.suptitle("Quantized TFLite model predictions (green: correct, red: incorrect)")
```


    
![png](output_79_0.png)
    


## Export image validation batch

Export validation batch so it can be tested client side. Below we create compressed file containing all images named with the convention:

`n{}_true{}_pred{}.jpg`

where the first number is index, the second - true label index, the third - value predicted by TFLite moder generated in this notebook. Example file will look similar to this: `n0_true1_pred1.jpg`.

All images then will be put into client side testing code (res/assets in Android tests). Integration tests will run inference process on each image and then compare results with the ones saved in file names.

from PIL import Image


```python
from PIL import Image
```

VAL_BATCH_DIR = "validation_batch"


```python
VAL_BATCH_DIR = "validation_batch"
```

!mkdir {VAL_BATCH_DIR}


```python
!mkdir {VAL_BATCH_DIR}
```

    하위 디렉터리 또는 파일 validation_batch이(가) 이미 있습니다.
    

# Export batch to *.jpg files with specific naming convention.
# Make sure they are exported in the full quality, otherwise the inference
# process will return different results. 

for n in range(32):
  filename = "n{:0.0f}_true{:0.0f}_pred{:0.0f}.jpg".format(
      n,
      true_label_ids[n],
      tflite_model_pred_simplified[n][0]
  )
  img_arr = np.copy(val_image_batch[n])
  img_arr *= 255
  img_arr = img_arr.astype("uint8")
  img11 = Image.fromarray(img_arr, 'RGB')
  img11.save("{}/{}".format(VAL_BATCH_DIR, filename), "JPEG", quality=100)


```python
# Export batch to *.jpg files with specific naming convention.
# Make sure they are exported in the full quality, otherwise the inference
# process will return different results. 

for n in range(32):
  filename = "n{:0.0f}_true{:0.0f}_pred{:0.0f}.jpg".format(
      n,
      true_label_ids[n],
      tflite_model_pred_simplified[n][0]
  )
  img_arr = np.copy(val_image_batch[n])
  img_arr *= 255
  img_arr = img_arr.astype("uint8")
  img11 = Image.fromarray(img_arr, 'RGB')
  img11.save("{}/{}".format(VAL_BATCH_DIR, filename), "JPEG", quality=100)
```

!tar -zcvf {VAL_BATCH_DIR}.tar.gz {VAL_BATCH_DIR}


```python
!tar -zcvf {VAL_BATCH_DIR}.tar.gz {VAL_BATCH_DIR}
```

    a validation_batch
    a validation_batch/n0_true1_pred1.jpg
    a validation_batch/n0_true3_pred3.jpg
    a validation_batch/n0_true4_pred1.jpg
    a validation_batch/n10_true0_pred0.jpg
    a validation_batch/n10_true1_pred1.jpg
    a validation_batch/n10_true4_pred1.jpg
    a validation_batch/n11_true0_pred1.jpg
    a validation_batch/n11_true1_pred1.jpg
    a validation_batch/n11_true3_pred3.jpg
    a validation_batch/n12_true1_pred1.jpg
    a validation_batch/n12_true4_pred4.jpg
    a validation_batch/n13_true0_pred0.jpg
    a validation_batch/n13_true2_pred2.jpg
    a validation_batch/n13_true3_pred3.jpg
    a validation_batch/n14_true0_pred1.jpg
    a validation_batch/n14_true2_pred2.jpg
    a validation_batch/n14_true4_pred1.jpg
    a validation_batch/n14_true4_pred4.jpg
    a validation_batch/n15_true0_pred0.jpg
    a validation_batch/n15_true2_pred1.jpg
    a validation_batch/n15_true3_pred3.jpg
    a validation_batch/n15_true4_pred4.jpg
    a validation_batch/n16_true1_pred0.jpg
    a validation_batch/n16_true1_pred1.jpg
    a validation_batch/n16_true4_pred4.jpg
    a validation_batch/n17_true1_pred1.jpg
    a validation_batch/n17_true2_pred2.jpg
    a validation_batch/n17_true3_pred3.jpg
    a validation_batch/n18_true0_pred0.jpg
    a validation_batch/n18_true2_pred2.jpg
    a validation_batch/n18_true3_pred3.jpg
    a validation_batch/n18_true4_pred4.jpg
    a validation_batch/n19_true1_pred1.jpg
    a validation_batch/n19_true2_pred1.jpg
    a validation_batch/n19_true3_pred3.jpg
    a validation_batch/n19_true4_pred4.jpg
    a validation_batch/n1_true0_pred1.jpg
    a validation_batch/n1_true1_pred1.jpg
    a validation_batch/n1_true3_pred3.jpg
    a validation_batch/n20_true0_pred0.jpg
    a validation_batch/n20_true0_pred4.jpg
    a validation_batch/n20_true1_pred0.jpg
    a validation_batch/n20_true2_pred2.jpg
    a validation_batch/n21_true0_pred0.jpg
    a validation_batch/n21_true1_pred1.jpg
    a validation_batch/n21_true4_pred4.jpg
    a validation_batch/n22_true1_pred1.jpg
    a validation_batch/n22_true2_pred2.jpg
    a validation_batch/n22_true3_pred3.jpg
    a validation_batch/n23_true1_pred0.jpg
    a validation_batch/n23_true1_pred1.jpg
    a validation_batch/n23_true2_pred2.jpg
    a validation_batch/n23_true4_pred4.jpg
    a validation_batch/n24_true1_pred1.jpg
    a validation_batch/n24_true2_pred2.jpg
    a validation_batch/n24_true3_pred3.jpg
    a validation_batch/n25_true0_pred1.jpg
    a validation_batch/n25_true2_pred2.jpg
    a validation_batch/n25_true2_pred3.jpg
    a validation_batch/n26_true1_pred1.jpg
    a validation_batch/n26_true3_pred1.jpg
    a validation_batch/n26_true3_pred3.jpg
    a validation_batch/n27_true1_pred1.jpg
    a validation_batch/n27_true3_pred3.jpg
    a validation_batch/n27_true4_pred1.jpg
    a validation_batch/n28_true1_pred1.jpg
    a validation_batch/n28_true2_pred2.jpg
    a validation_batch/n28_true4_pred4.jpg
    a validation_batch/n29_true1_pred0.jpg
    a validation_batch/n29_true1_pred1.jpg
    a validation_batch/n29_true3_pred3.jpg
    a validation_batch/n29_true4_pred4.jpg
    a validation_batch/n2_true0_pred0.jpg
    a validation_batch/n2_true2_pred2.jpg
    a validation_batch/n2_true3_pred1.jpg
    a validation_batch/n30_true0_pred1.jpg
    a validation_batch/n30_true2_pred2.jpg
    a validation_batch/n30_true3_pred3.jpg
    a validation_batch/n31_true0_pred0.jpg
    a validation_batch/n31_true2_pred1.jpg
    a validation_batch/n31_true2_pred2.jpg
    a validation_batch/n31_true4_pred4.jpg
    a validation_batch/n3_true1_pred1.jpg
    a validation_batch/n3_true2_pred1.jpg
    a validation_batch/n3_true3_pred3.jpg
    a validation_batch/n3_true4_pred4.jpg
    a validation_batch/n4_true2_pred1.jpg
    a validation_batch/n4_true3_pred3.jpg
    a validation_batch/n5_true0_pred0.jpg
    a validation_batch/n5_true1_pred1.jpg
    a validation_batch/n5_true3_pred1.jpg
    a validation_batch/n5_true3_pred3.jpg
    a validation_batch/n6_true0_pred0.jpg
    a validation_batch/n6_true1_pred1.jpg
    a validation_batch/n7_true0_pred0.jpg
    a validation_batch/n7_true1_pred1.jpg
    a validation_batch/n7_true4_pred4.jpg
    a validation_batch/n8_true0_pred0.jpg
    a validation_batch/n8_true3_pred3.jpg
    a validation_batch/n9_true1_pred1.jpg
    a validation_batch/n9_true3_pred3.jpg
    

File `validation_batch.tar.gz` is ready to be downloaded, unpacked and put into client-side testing code.


```python
from IPython.core.display import display, HTML

display(HTML("<style>.container { width:90% !important; }</style>"))
```

    C:\Users\cupid\AppData\Local\Temp\ipykernel_6452\1939805293.py:1: DeprecationWarning: Importing display from IPython.core.display is deprecated since IPython 7.14, please import from IPython display
      from IPython.core.display import display, HTML
    


<style>.container { width:90% !important; }</style>

