import os
import pickle
import random
import numpy as np

import keras
from keras import backend as K
from keras.layers import (Activation, Convolution2D, Dense, Dropout, Flatten,
                          Input, Lambda, MaxPooling2D, Reshape)
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, load_img
from keras.regularizers import l2
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# constants
dim = 105
epoch = 200
C_value = 0.2
data_folder = 'data'


def create_pairs_improved(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    class_num = len(digit_indices)
    for d in range(class_num):
        for i in range(len(digit_indices[d])):
            for j in range(i + 1, len(digit_indices[d])):
                z1, z2 = digit_indices[d][i], digit_indices[d][j]
                pairs += [[x[z1], x[z2]]]
                inc = random.randrange(1, class_num)
                dn = (d + inc) % class_num
                di = random.randrange(0, len(digit_indices[dn]))
                z1, z2 = digit_indices[d][i], digit_indices[dn][di]
                pairs += [[x[z1], x[z2]]]
                labels += [1, -1]
    return np.array(pairs), np.array(labels)


def load_data():
    class_num = 0
    X = []
    y = np.array([])
    folder = data_folder
    for directory in os.listdir(folder):
        if directory != '.DS_Store':
            for subdirectory in os.listdir(os.path.join(folder, directory)):
                if subdirectory != '.DS_Store':
                    for file in os.listdir(os.path.join(folder, directory, subdirectory)):
                        image_path = os.path.join(
                            folder, directory, subdirectory, file)
                        img = load_img(image_path, grayscale=True)
                        img = img.resize((dim, dim))
                        img_array = np.array(img)
                        X.append(img_array)
                        y = np.append(y, class_num)
                    class_num += 1

    X = np.array(X)
    y = np.array(y)

    X = X.reshape(X.shape[0], dim, dim, 1)

    digit_indices = [np.where(y == i)[0] for i in range(class_num)]
    random.seed(42)
    random.shuffle(digit_indices)

    train = digit_indices[:1150]
    validate = digit_indices[1150:1200]
    test = digit_indices[1200:]

    return X, train, validate, test


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    count = 0
    for i in range(len(labels)):
        if (labels[i] == -1 and predictions[i] < 0) or (labels[i] == 1 and 0 <= predictions[i]):
            count += 1
    return float(count) / len(labels)


def create_conv_network():
    seq = Sequential()
    input_shape = (dim, dim, 1)

    seq.add(Convolution2D(64, 10, 10, border_mode='valid', input_shape=input_shape))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2)))

    seq.add(Dropout(0.1))
    seq.add(Convolution2D(128, 7, 7, border_mode='valid'))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2)))

    seq.add(Dropout(0.1))
    seq.add(Convolution2D(128, 4, 4, border_mode='valid'))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2)))

    seq.add(Dropout(0.1))
    seq.add(Convolution2D(256, 4, 4, border_mode='valid'))
    seq.add(Activation('relu'))

    seq.add(Flatten())
    seq.add(Dropout(0.2))
    seq.add(Dense(4096))
    return seq


def l1_distance(vects):
    x, y = vects
    return K.abs(x - y)


def l1_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], shape1[1])


def svm_loss(y_true, y_pred):
    return C_value * K.mean(K.square(K.maximum(1 - y_pred * y_true, 0)))


def create_model():
    # network definition
    base_network = create_conv_network()
    input_a = Input(shape=(dim, dim, 1))
    input_b = Input(shape=(dim, dim, 1))

    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    feature_model = Model(input=[input_a, input_b], output=[
                          processed_a, processed_b])
    distance = Lambda(l1_distance, output_shape=l1_output_shape)(
        feature_model.output)
    # svm part
    distance = Dense(1, W_regularizer=l2(0.5))(distance)
    model = Model(input=[input_a, input_b], output=distance)
    opt = Adam()
    model.compile(loss=svm_loss, optimizer=opt)
    model.summary()
    return model, feature_model


def generator(X, indices):
    batch = 512
    pairs = []
    labels = []
    batchcounter = 0
    while 1:
        class_num = len(indices)
        for d in range(class_num):
            for i in range(len(indices[d])):
                for j in range(i + 1, len(indices[d])):
                    z1, z2 = indices[d][i], indices[d][j]
                    pairs += [[X[z1], X[z2]]]
                    inc = random.randrange(1, class_num)
                    dn = (d + inc) % class_num
                    di = random.randrange(0, len(indices[dn]))
                    z1, z2 = indices[d][i], indices[dn][di]
                    pairs += [[X[z1], X[z2]]]
                    labels += [1, -1]
                    batchcounter += 2
                    if batchcounter >= batch:
                        pairs, labels = np.array(pairs), np.array(labels)
                        yield ([pairs[:, 0], pairs[:, 1]], labels)
                        pairs = []
                        labels = []
                        batchcounter = 0


def test(n, k, X, indices, feature_model):
    # number of seen examples:k-shot
    # number of classes: n-way
    testnumber = 10
    sumacc = 0
    class_num = len(indices)

    for test in range(testnumber):
        temp_ = []
        for i in range(class_num):
            temp_.append(i)
        random.shuffle(temp_)
        choosen_classes = temp_[:n]
        X_train = []
        y_train = []
        y_test = []
        X_test = []
        for cl in choosen_classes:
            for i in range(k):
                X_train.append(X[indices[cl][i]])
                y_train.append(cl)
            for i in range(k, len(indices[cl])):
                X_test.append(X[indices[cl][i]])
                y_test.append(cl)
        X_train = np.array(X_train)
        X_train = np.reshape(X_train, (len(X_train), dim, dim, 1))
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (len(X_test), dim, dim, 1))
        y_test = np.array(y_test)

        # calculate features
        X_train_ = feature_model.predict([X_train, X_train])[0]
        X_test_ = feature_model.predict([X_test, X_test])[0]
        clf = svm.SVC(C=C_value, kernel='linear')
        clf.fit(X_train_, y_train)
        pred = clf.predict(X_test_)
        acc = accuracy_score(y_test, pred)
        sumacc += acc
    return float(sumacc) / testnumber


def log(X, test_indices, validation_indices, feature_model):
    result = []
    result.append(test(5, 1, X, test_indices, feature_model))
    result.append(test(5, 5, X, test_indices, feature_model))
    result.append(test(20, 1, X, test_indices, feature_model))
    result.append(test(20, 5, X, test_indices, feature_model))
    result.append(test(5, 1, X, validation_indices, feature_model))
    result.append(test(5, 5, X, validation_indices, feature_model))
    result.append(test(20, 1, X, validation_indices, feature_model))
    result.append(test(20, 5, X, validation_indices, feature_model))
    return result


def plot(result):
    print('Validation: 5 way 1-shot accuracy:', result[4] * 100, '%')
    print('Validation: 5 way 5-shot accuracy:', result[5] * 100, '%')
    print('Validation: 20 way 1-shot accuracy:', result[6] * 100, '%')
    print('Validation: 20 way 5-shot accuracy:', result[7] * 100, '%')
    print('Test: 5 way 1-shot accuracy:', result[0] * 100, '%')
    print('Test: 5 way 5-shot accuracy:', result[1] * 100, '%')
    print('Test: 20 way 1-shot accuracy:', result[2] * 100, '%')
    print('Test: 20 way 5-shot accuracy:', result[3] * 100, '%')


def main():
    X, train, validate, test = load_data()
    validate_pairs, validate_y = create_pairs_improved(X, validate)
    test_pairs, test_y = create_pairs_improved(X, test)

    # start experiment
    results = []
    model, feature_model = create_model()
    best_acc = 0.0
    for j in range(epoch):
        print('Epoch ', epoch)
        model.fit_generator(generator(X, train), 1150 * 380, 1)
        predicted = model.predict([validate_pairs[:, 0], validate_pairs[:, 1]])
        val_acc = compute_accuracy(predicted, validate_y)
        print('Validation accuracy:', val_acc * 100, '%')
        if val_acc > best_acc:
            results = log(X, test, validate, feature_model)
            plot(results)
            best_acc = val_acc

    plot(results)


if __name__ == "__main__":
    main()
