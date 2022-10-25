
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, regularizers
import tensorflow_hub as hub  # type:ignore


def useBestModel(dataT, reg1=3, reg2=3, reg3=3, dropout=1, layerSize=50, activation="relu"):

    activation = activation

    eeg_model = tf.keras.Sequential([

        hub.KerasLayer(handle="saved_model/best_model",
                       input_shape=(dataT.shape[1], dataT.shape[2], 1))




    ])
    return eeg_model


def makeModel2(dataT, reg1=3, reg2=3, reg3=3, dropout=1, layerSize=50, activation="relu"):
    z = reg1
    x = reg2
    y = reg3
    drpout = dropout*0.1
    lz = layerSize
    activation = activation

    print("asdasda")
    print(dataT.shape[1])
    print(dataT.shape[2])
    eeg_model = tf.keras.Sequential([


        # layers.Flatten(input_shape = data_train.shape[1]),
        layers.LocallyConnected2D(lz, [25, 20], 20,
                                  input_shape=(dataT.shape[1], dataT.shape[2], 1), activation=activation,
                                  kernel_regularizer=regularizers.L1L2(
                                      l1=z*1e-5, l2=z*1e-4),
                                  bias_regularizer=regularizers.L2(z*1e-4),
                                  activity_regularizer=regularizers.L2(z*1e-5)

                                  ),

        layers.Dropout(drpout),
        layers.LocallyConnected2D(lz, [15, 6], 6,
                                  input_shape=(
                                      dataT.shape[1], dataT.shape[2], 1),
                                  activation=activation, kernel_regularizer=regularizers.L1L2(
                                      l1=x*1e-5, l2=x*1e-4),
                                  bias_regularizer=regularizers.L2(x*1e-4),
                                  activity_regularizer=regularizers.L2(x*1e-5)
                                  ),

        layers.Dropout(drpout+0.1),
        layers.Dense(units=lz*4, activation=activation,
                     kernel_regularizer=regularizers.L1L2(
                         l1=y*1e-5, l2=y*1e-4),
                     bias_regularizer=regularizers.L2(y*1e-4),
                     activity_regularizer=regularizers.L2(y*1e-5)

                     ),
        layers.Dropout(drpout+0.2),

        layers.Flatten(),

        layers.Dense(units=2, activation="softmax")


    ])
    return eeg_model


def makeModel(dataT, reg1=3, reg2=3, reg3=3, dropout=1, layerSize=50, activation="relu"):
    z = reg1
    x = reg2
    y = reg3
    drpout = dropout*0.1
    lz = layerSize
    activation = activation

    xK = round(dataT.shape[1] * 0.1)
    yK = round(dataT.shape[2] * 0.1)
    strid = round(1 - (xK/dataT.shape[1])) * 10

    strid2 = 5
    xK2 = round(dataT.shape[1]*0.1 - strid2)
    yK2 = round(dataT.shape[2]*0.1 - strid2)

    print(xK)
    print(yK)
    print(strid)
    print(xK2)
    print(yK2)

    # strid2 = 20
    # xK2 = (1 - strid2/20 ) * dataT.shape[1]
    # yK2 = (1 - strid2/20 ) * dataT.shape[2]
    eeg_model = tf.keras.Sequential([


        # layers.Flatten(input_shape = data_train.shape[1]),
        layers.LocallyConnected2D(lz, [xK, yK], strid,
                                  input_shape=(
                                      dataT.shape[1], dataT.shape[2], 1),
                                  activation=activation, kernel_regularizer=regularizers.L1L2(
                                      l1=z*1e-5, l2=z*1e-4),
                                  bias_regularizer=regularizers.L2(z*1e-4),
                                  activity_regularizer=regularizers.L2(z*1e-5)

                                  ),

        layers.Dropout(drpout),
        layers.LocallyConnected2D(lz, [xK2, yK2], strid2,
                                  input_shape=(
                                      dataT.shape[1], dataT.shape[2], 1),
                                  activation=activation, kernel_regularizer=regularizers.L1L2(
                                      l1=x*1e-5, l2=x*1e-4),
                                  bias_regularizer=regularizers.L2(x*1e-4),
                                  activity_regularizer=regularizers.L2(x*1e-5)
                                  ),

        layers.Dropout(drpout+0.1),
        layers.Dense(units=lz*4, activation=activation,
                     kernel_regularizer=regularizers.L1L2(
                         l1=y*1e-5, l2=y*1e-4),
                     bias_regularizer=regularizers.L2(y*1e-4),
                     activity_regularizer=regularizers.L2(y*1e-5)

                     ),
        layers.Dropout(drpout+0.2),

        layers.Flatten(),

        layers.Dense(units=2, activation="softmax")


    ])
    return eeg_model


def trainTestModel2(eeg_model, data_train, data_test,
                    labels_train, labels_test, reg, drp, lz=50, act="relu"):
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=50, restore_best_weights=True)

    eeg_model.compile(optimizer='adam',
                      loss="binary_crossentropy",
                      metrics=['accuracy'])

    # Reshaping data to fit with neural net.
    data_train_send = np.reshape(
        data_train, [data_train.shape[0], data_train.shape[1], data_train.shape[2], 1])
    data_test_send = np.reshape(
        data_test, [data_test.shape[0], data_test.shape[1], data_test.shape[2], 1])
    print(data_train_send.shape)

    # Training NN
    outputs = eeg_model.fit(data_train_send, labels_train, validation_split=0.2, callbacks=[
                            callback], epochs=1000, verbose=False)

    # Printing results
    print("Results")
    eeg_model.evaluate(data_test_send, labels_test)
    # result = eeg_model.predict(data_test_send)

    # Plotting training and validation results.
    val_loss = outputs.history["val_loss"]
    loss = outputs.history["loss"]

    plt.plot(loss, 'r', label='Training loss')
    plt.plot(val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss for reg val {} and dropout val {} and layersize val {}'.format(
        reg, drp, lz))
    plt.legend()
    plt.figure()

    val_acc = outputs.history["val_accuracy"]
    acc = outputs.history["accuracy"]
    plt.plot(acc, 'r', label='Training acc')
    plt.plot(val_acc, 'b', label='Validation acc')
    plt.title('Training and validation acc for reg val {} and dropout val {} and layersize val {}'.format(
        reg, drp, lz))
    plt.legend()
    plt.figure()
    return outputs


def trainTestModel(eeg_model, data_train, data_test,
                   labels_train, labels_test, reg, drp, lz=50,
                   act="relu", best_loss=10000, saveBest=False,
                   specificSubject=None):

    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=60, restore_best_weights=True)

    eeg_model.compile(optimizer='adam',
                      loss="binary_crossentropy",
                      metrics=['accuracy'])

    # Reshaping data to fit with neural net.
    data_train_send = np.reshape(
        data_train, [data_train.shape[0], data_train.shape[1], data_train.shape[2], 1])
    data_test_send = np.reshape(
        data_test, [data_test.shape[0], data_test.shape[1], data_test.shape[2], 1])
    print(data_train_send.shape)

    # Training NN
    outputs = eeg_model.fit(data_train_send, labels_train, validation_split=0.2, callbacks=[
                            callback], epochs=1000, verbose=False)

    # Printing results
    print("Results")
    eval = eeg_model.evaluate(data_test_send, labels_test)
    # result = eeg_model.predict(data_test_send)

    # Plotting training and validation results.
    val_loss = outputs.history["val_loss"]
    loss = outputs.history["loss"]

    if saveBest is True:
        if (np.amin(val_loss)+eval[0]) < best_loss:
            best_loss = (np.amin(val_loss)+eval[0])
            eeg_model.save('saved_model/best_model{}'.format(specificSubject))

    plt.plot(loss, 'r', label='Training loss')
    plt.plot(val_loss, 'b', label='Validation loss')
    plt.title('loss {} for reg val {} , dropout val {} ,layersize val {} and act {}'.format(
        round(eval[0], 2), reg, drp, lz, act))
    plt.legend()
    plt.figure()
    plt.pause(0.1)

    val_acc = outputs.history["val_accuracy"]
    acc = outputs.history["accuracy"]
    plt.plot(acc, 'r', label='Training acc')
    plt.plot(val_acc, 'b', label='Validation acc')
    plt.title('acc {}  for reg val {} , dropout val {} , layersize val {} and act {}'.format(
        round(eval[1], 2), reg, drp, lz, act))
    plt.legend()
    plt.figure()
    plt.pause(0.1)

    return best_loss, eval[1]
