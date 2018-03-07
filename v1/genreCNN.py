import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, Input, Conv1D, MaxPooling1D, Reshape, Activation
from keras import applications

# dimensions of our images.
img_width, img_height = 610, 450

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'genres/training'
validation_data_dir = 'genres/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16
img_rows, img_cols = 610, 450 
 


def train_net():
    datagen = ImageDataGenerator(
            ) #No augmentation to start out

    train_generator = datagen.flow_from_directory(
        train_data_dir,
        batch_size=batch_size,
        target_size= (6100,4500),
        #color_mode='grayscale',
        class_mode='categorical')

    validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        batch_size=batch_size,
        class_mode='categorical',
        target_size= (6100,4500)
        #color_mode='grayscale'
        )


    # build the Network
    model = Sequential()
    # model.add(Reshape(
    #     (6100,4500),
    #     input_shape=(6100,4500,1)
    #     ))
    model.add(Conv2D(256, # 30 filters
        kernel_size= (400,1),
        strides = (200,1), #1x20 filters to heavily favor patterns over time
        activation='relu',
        input_shape=(6100,4500,3)
        ))
    model.add(MaxPooling2D(
        pool_size=(2,2)
        ))
    model.add(Conv2D(256, # 30 filters
        kernel_size= (1,400),
        strides = (1,200), #1x20 filters to heavily favor patterns over time
        activation='relu',
        ))
    model.add(MaxPooling2D(
        pool_size=(2,2)
        ))

    model.add(Dense(1000,
        activation= 'relu',
        ))
    model.add(Dropout(0.2))
    model.add(Dense(200,
        activation='relu'
        )) 
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(10,
        activation='softmax'
        ))
    
    keras.utils.print_summary(model)

    adam = keras.optimizers.Adam(lr=0.001, decay=.01)
    model.compile(
        metrics=['accuracy'],
        loss=keras.losses.categorical_crossentropy,
        optimizer = adam
        )
    model.fit_generator(
        train_generator,
        epochs=200,
        validation_data=validation_generator,
        verbose=2
        )
    loss, acc = model.evaluate([tx, txq], ty,
                           batch_size=BATCH_SIZE)
    print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.array(
        [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.array(
        [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              #validation_data=(validation_data, validation_labels)
              )
    model.save_weights(top_model_weights_path)


train_net()

