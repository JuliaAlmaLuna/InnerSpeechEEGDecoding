import numpy as np
import tensorflow as tf

import feature_extraction as fclass
import NNMethods as nnM


# This loop works when Dataset concatenaded and normal zed using Old way.
# Set seed either randomly or manually. Needs to be same for each subject.
fClassDict = dict()
for seed in np.arange(25, 40):

    for sub in [1, 7]:  # 1 ,2 as well!
        # for sub in [1]:
        fClassDict[f"{seed},{sub}"] = fclass.featureEClass()

        specificSubject = sub
        mDataList = None
        mDataList = fClassDict[f"{seed},{sub}"].getFeatures(
            specificSubject, t_min=2, t_max=3, sampling_rate=256, twoDLabels=True
        )

        bestResultsTot = None
        ResultsPerDataSet = []
        dataset_losses = []

        first = True
        for data_train, data_test, labels_train, labels_test, name in mDataList[2:]:
            # Make model with params
            useBest = False

            print(f"\n Running dataset: {name} \n")
            # eeg_model = nnM.makeModel(data_train, reg1=3, reg2=3, reg3=3, dropout=1)
            eeg_model = nnM.makeModel(data_train, reg1=3, reg2=3, reg3=3, dropout=1)

            eeg_model.build()
            eeg_model.summary()
            best_loss = 1000000
            this_loss = 1000000

            tf.keras.utils.set_random_seed(1)

            if useBest is True:
                tf.keras.backend.clear_session()
                eeg_model = nnM.useBestModel(data_train)
                eeg_model.build()
                print("Training best model")
                best_loss, this_loss = nnM.trainTestModel(
                    eeg_model,
                    data_train,
                    data_test,
                    labels_train,
                    labels_test,
                    saveBest=False,
                    reg=1,
                    drp=1,
                    specificSubject=specificSubject,
                )

            else:
                for act in ["relu"]:  # "Leaky_relu"
                    for lz in [100]:  # ,50 10,10,25,10,,25
                        for reg in [0, 1.5, 3]:  # [0.5,1.5,3]
                            for drp in [0, 1]:  # (1,2):,3
                                tf.keras.backend.clear_session()

                                eeg_model = nnM.makeModel(
                                    data_train,
                                    reg1=reg,
                                    reg2=reg,
                                    reg3=reg,
                                    dropout=drp,
                                    layerSize=lz,
                                    activation=act,
                                )
                                eeg_model.build()
                                # eeg_model.summary()
                                print(
                                    "reg = {}, drp = {}  lz = {} , act = {}".format(
                                        reg, drp, lz, act
                                    )
                                )
                                best_loss, this_acc = nnM.trainTestModel(
                                    eeg_model,
                                    data_train,
                                    data_test,
                                    labels_train,
                                    labels_test,
                                    reg,
                                    drp,
                                    lz=lz,
                                    act=act,
                                    best_loss=best_loss,
                                    saveBest=True,
                                    specificSubject=specificSubject,
                                )

                                dataset_losses.append(
                                    [name, this_acc, act, lz, reg, drp]
                                )
            # Saving the results
            # ResultsPerDataSet.append([best_loss, name])
            ResultsPerDataSet.append(dataset_losses)
            dataset_losses = []
            if first:
                bestResultsTot = np.array([best_loss, name], dtype=object)
            if bestResultsTot[0] < best_loss:
                bestResultsTot = np.array([best_loss, name], dtype=object)
        print(bestResultsTot)

        # savearray = np.array([ResultsPerDataSet, seed, specificSubject], dtype=object)
        savearray = np.array(
            [bestResultsTot, seed, specificSubject, ResultsPerDataSet], dtype=object
        )
        from datetime import datetime

        now = datetime.now()

        # Month abbreviation, day and year
        now_string = now.strftime("D--%d-%m-%Y-T--%H-%M-%S")

        np.save(
            f"F:/PythonProjects/NietoExcercise-1/SavedResultsNN/savedBestSeed-{seed}-Subject-{specificSubject}\
                -Date-{now_string}",
            savearray,
        )

        # savearray = np.array([bestResultsPerDataSet, seed, specificSubject], dtype=object)
        # from datetime import datetime
        # now = datetime.now()
        # # Month abbreviation, day and year
        # now_string = now.strftime("D--%d-%m-%Y-T--%H-%M-%S")

        # np.save(f"F:/PythonProjects/NietoExcercise-1/SavedResults/savedBestSeed-{seed}-Date-{now_string}",savearray)

        # #np.save(f"F:/PythonProjects/NietoExcercise-1/SavedResults/savedBest-{now_string}",savearray)
