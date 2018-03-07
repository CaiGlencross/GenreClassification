import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, Input, Conv1D, MaxPooling1D, Reshape, Activation, AveragePooling1D, Concatenate
from keras import applications
import sunau

# dimensions of our images.
img_width, img_height = 610, 450

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'genres/training'
validation_data_dir = 'genres/validation'
num_training = 66
num_validation = 34
data_dir = 'genres/'
num_epochs = 50
batch_size = 1
img_rows, img_cols = 610, 450 

x_train = None
y_train= None

x_validation = None
y_validation = None


def get_training_arrays(genre):
    for x in range(0, num_training):
        if (x < 10):
            ind_string = "0"+str(x)
        else:
            ind_string = str(x)
        filepath = data_dir + genre + "/" + genre + ".000" + ind_string + ".au"
        song = sunau.open(filepath, 'r')
        amp_string = song.readframes(660000)
        amp_array = np.fromstring(amp_string, dtype=np.dtype('>h'))
        #print("amplitude array (hopefully): ",amp_array.shape)
        if (x==0):
            training_array = amp_array
        else:
            training_array = np.vstack([training_array, amp_array])

    print(genre + "'s arrays are",training_array)
    return training_array

def get_validation_arrays(genre):
    for x in range(num_training, (num_training+num_validation)):
        if (x < 10):
            ind_string = "0"+str(x)
        else:
            ind_string = str(x)
        filepath = data_dir + genre + "/" + genre + ".000" + ind_string + ".au"
        song = sunau.open(filepath, 'r')
        amp_string = song.readframes(660000)
        amp_array = np.fromstring(amp_string, dtype=np.dtype('>h'))
        song.close()
        #print("amplitude array (hopefully):\n\t",amp_array)
        if (x==num_training):
            validation_array = amp_array
        else:
            validation_array = np.vstack([validation_array, amp_array])

    print(genre + "'s arrays are",validation_array)
    return validation_array

def convert_audio():

    reggae_array = get_training_arrays("reggae")
    x_train = reggae_array
    hiphop_array = get_training_arrays("hiphop")
    x_train = np.vstack([x_train,hiphop_array])
    blues_array = get_training_arrays("blues")
    x_train = np.vstack([x_train,blues_array])
    classical_array = get_training_arrays("classical")
    x_train = np.vstack([x_train,blues_array])
    country_array = get_training_arrays("country")
    x_train = np.vstack([x_train,country_array])
    disco_array = get_training_arrays("disco")
    x_train = np.vstack([x_train,disco_array])
    jazz_array = get_training_arrays("jazz")
    x_train = np.vstack([x_train,jazz_array])
    metal_array = get_training_arrays("metal")
    x_train = np.vstack([x_train,metal_array])
    pop_array = get_training_arrays("pop")
    x_train = np.vstack([x_train,pop_array])
    rock_array = get_training_arrays("rock")
    x_train = np.vstack([x_train,rock_array])

    y_train = np.array([[1,0,0,0,0,0,0,0,0,0]]*num_training+
                        [[0,1,0,0,0,0,0,0,0,0]]*num_training+
                        [[0,0,1,0,0,0,0,0,0,0]]*num_training+
                        [[0,0,0,1,0,0,0,0,0,0]]*num_training+
                        [[0,0,0,0,1,0,0,0,0,0]]*num_training+
                        [[0,0,0,0,0,1,0,0,0,0]]*num_training+
                        [[0,0,0,0,0,0,1,0,0,0]]*num_training+
                        [[0,0,0,0,0,0,0,1,0,0]]*num_training+
                        [[0,0,0,0,0,0,0,0,1,0]]*num_training+
                        [[0,0,0,0,0,0,0,0,0,1]]*num_training)


    print("x_train is: ", x_train)
    print("y_train is: ", y_train)

    reggae_validation = get_validation_arrays("reggae")
    x_validation = reggae_validation
    hiphop_validation = get_validation_arrays("hiphop")
    x_validation = np.vstack([x_validation,hiphop_validation])
    blues_validation = get_validation_arrays("blues")
    x_validation  =  np.vstack([x_validation,blues_validation])
    classical_validation = get_validation_arrays("classical")
    x_validation = np.vstack([x_validation,classical_validation])
    country_validation = get_validation_arrays("country")
    x_validation = np.vstack([x_validation,country_validation])
    disco_validation = get_validation_arrays("disco")
    x_validation = np.vstack([x_validation,disco_validation])
    jazz_validation = get_validation_arrays("jazz")
    x_validation = np.vstack([x_validation,jazz_validation])
    metal_validation = get_validation_arrays("metal")
    x_validation = np.vstack([x_validation,metal_validation])
    pop_validation = get_validation_arrays("pop")
    x_validation = np.vstack([x_validation,pop_validation])
    rock_validation = get_validation_arrays("rock")
    x_validation = np.vstack([x_validation,rock_validation])

    y_validation = np.array([[1,0,0,0,0,0,0,0,0,0]]*num_validation+
                        [[0,1,0,0,0,0,0,0,0,0]]*num_validation+
                        [[0,0,1,0,0,0,0,0,0,0]]*num_validation+
                        [[0,0,0,1,0,0,0,0,0,0]]*num_validation+
                        [[0,0,0,0,1,0,0,0,0,0]]*num_validation+
                        [[0,0,0,0,0,1,0,0,0,0]]*num_validation+
                        [[0,0,0,0,0,0,1,0,0,0]]*num_validation+
                        [[0,0,0,0,0,0,0,1,0,0]]*num_validation+
                        [[0,0,0,0,0,0,0,0,1,0]]*num_validation+
                        [[0,0,0,0,0,0,0,0,0,1]]*num_validation)
    print("x_validation is:", x_validation)
    print("y_validation is:", y_validation)


    return x_train, y_train , x_validation,y_validation














def train_net():

    train_data, train_labels , validation_data, validation_labels = convert_audio()
    train_data = np.expand_dims(train_data, axis=2)
    validation_data = np.expand_dims(validation_data, axis=2)
    print("training data shape: " , train_data.shape)
    print("training label shape: ", train_labels.shape)
    # build the Network
    input_layer = Input(shape = train_data.shape[1:])
    conv_1 = Conv1D(256, 
                kernel_size= 4,
                strides = 2
                )(input_layer)
    act_1  = Activation('relu')(conv_1)
    pool_1 = MaxPooling1D(
                pool_size=4
                )(act_1)
    conv_2 = Conv1D(256, 
                kernel_size= 4,
                strides = 2
                )(pool_1)
    act_2  = Activation('relu')(conv_2)
    pool_2 = MaxPooling1D(
                pool_size=2
                )(act_2)
    conv_3 = Conv1D(256, 
                kernel_size= 4,
                strides = 2
                )(pool_2)
    act_3  = Activation('relu')(conv_3)            
    pool_3 = MaxPooling1D(
                pool_size=2
                )(act_3)
    avgPool= AveragePooling1D(pool_size=2)(pool_3)
    maxPool= MaxPooling1D(pool_size=2)(pool_3)
    concat = keras.layers.concatenate([avgPool,maxPool], axis=-1)
    flatten= Flatten()(concat)
    dense_1= Dense(2048,
                activation= 'relu',
                )(flatten)
    dropout_1 = Dropout(0.2)(dense_1)
    dense_2  = Dense(10
                )(dropout_1)
    dropout_2= Dropout(0.2)(dense_2)
    activation_4=Activation('softmax')(dropout_2)







    # model.add(Conv1D(256, # 30 filters
    #     kernel_size= 4,
    #     strides = 2, 
    #     activation='relu',
    #     input_shape = train_data.shape[1:]
    #     ))
    # model.add(MaxPooling1D(
    #     pool_size=2
    #     ))
    # model.add(Conv1D(256,
    #     kernel_size=4,
    #     strides=2,
    #     activation='relu'
    #     ))
    # model.add(MaxPooling1D(
    #     pool_size=2
    # ))
    # model.add(Conv1D(256,
    #     kernel_size=4,
    #     strides=2,
    #     activation='relu'
    #     ))
    # model.add(MaxPooling1D(
    #     pool_size=2
    #     ))
    # model.add(Conv1D(256,
    #     kernel_size=4,
    #     strides=2,
    #     activation='relu'))
    # model.add(MaxPooling1D(
    #     pool_size=2
    #     ))



    # model.add(Flatten())
    # model.add(Dense(2048,
    #     activation= 'relu',
    #     ))
    # model.add(Dropout(0.2))
    # model.add(Dense(2048,
    #     activation='relu'
    #     )) 
    # model.add(Dropout(0.2))
    
    # model.add(Dense(10,
    #     activation='softmax'
    #     ))
    
    model = Model(inputs=input_layer, outputs = activation_4)
    keras.utils.print_summary(model)

    adam = keras.optimizers.Adam(lr=0.001, decay=.01)
    model.compile(
        metrics=['accuracy'],
        loss=keras.losses.categorical_crossentropy,
        optimizer = adam
        )
    model.fit(
        train_data, train_labels,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data = (validation_data,validation_labels),
        verbose = 2
        )
    # loss, acc = model.evaluate([tx, txq], ty,
    #                        batch_size=BATCH_SIZE)
    # print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))


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
              validation_data=(validation_data, validation_labels)
              )
    model.save_weights(top_model_weights_path)

#convert_audio()
train_net()

