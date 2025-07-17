from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from keras.models import Sequential, Model
from keras.metrics import BinaryAccuracy, Precision, Recall
from tensorflow.keras.optimizers import Adam
import pandas as pd
from tensorflow.keras.losses import BinaryCrossentropy
from keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Embedding, LSTM, Dropout, Bidirectional, GRU, Flatten, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
import numpy as np
from keras.initializers import Zeros, RandomNormal, Orthogonal, GlorotUniform
from keras.initializers import glorot_normal as KaimingNormal
import tensorflow as tf

class ValidationF1ScoreCallback(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs={}):
        x_val, y_val = self.validation_data
        y_pred = np.round(self.model.predict(x_val, verbose=0))
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1_score = 2 * ((precision * recall) / (precision + recall + 1e-7))
        accuracy = accuracy_score(y_val, y_pred)
        print(f'Epoch: {epoch}, val_f1_score: {f1_score:.4f}, val_accuracy: {accuracy:.4f}')


def get_initializer(init_method):
    if init_method == 'zero':
        return Zeros()
    elif init_method == 'normal':
        return RandomNormal()
    elif init_method == 'kaiming':
        return KaimingNormal()
    elif init_method == 'ortho':
        return Orthogonal()
    else:
        return GlorotUniform()


def train_and_eval(model_type, init_method, early_stopping, data, labels):
    # 划分训练集、验证集和测试集
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

    # 设置参数初始化方法
    initializer = get_initializer(init_method)

    if model_type == 'CNN':
        # 创建模型
        model = Sequential()
        model.add(Conv1D(128, 5, activation='relu', input_shape=(50, 50), kernel_initializer=initializer))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(1, activation='sigmoid', kernel_initializer=initializer))
    if model_type == 'RNN_LSTM':
        # 创建模型
        model = Sequential()
        model.add(Bidirectional(LSTM(units=128, activation='tanh', return_sequences=True, kernel_initializer=initializer), input_shape=(11, 2)))
        model.add(Dropout(0.5))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(1, activation='sigmoid', kernel_initializer=initializer))
    if model_type == 'MLP':
        # 创建模型
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(50, 50), kernel_initializer=initializer))
        model.add(Flatten())  # 新增的Flatten层
        model.add(Dense(1, activation='sigmoid', kernel_initializer=initializer))
    if model_type == 'RNN_GRU':
        # 创建模型
        model = Sequential()
        model.add(Bidirectional(GRU(units=128, activation='tanh', return_sequences=True, kernel_initializer=initializer), input_shape=(100, 50)))
        model.add(Dropout(0.5))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(1, activation='sigmoid', kernel_initializer=initializer))

    initial_learning_rate = 0.002
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=300,
        decay_rate=0.96,
        staircase=True)
    optimizer = Adam(learning_rate=lr_schedule)
    # 创建BinaryAccuracy，Precision和Recall对象
    accuracy = BinaryAccuracy(threshold=0.5)
    precision = Precision(thresholds=0.5)
    recall = Recall(thresholds=0.5)
    # 编译模型，使用新的优化器
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[accuracy, precision, recall])

    # 创建ModelCheckpoint回调
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

    # 创建回调列表
    callbacks = [checkpoint, ValidationF1ScoreCallback((np.array(X_val), np.array(y_val)))]
    epochs = 10
    # 如果early_stopping为True，添加EarlyStopping回调
    if early_stopping == 'true':
        early_stopping_callback = EarlyStopping(monitor='val_binary_accuracy', patience=3)
        callbacks.append(early_stopping_callback)
        epochs = 50

    # 训练模型
    model.fit(np.array(X_train), np.array(y_train), validation_data=(np.array(X_val), np.array(y_val)), epochs=epochs, batch_size=64, callbacks=callbacks, verbose=0)
    # 加载最好的模型
    model.load_weights('best_model.keras')
    # 用测试集评估模型
    loss, accuracy, precision, recall = model.evaluate(np.array(X_test), np.array(y_test), verbose=0)
    f1_score = 2 * ((precision * recall) / (precision + recall + 1e-7))
    print(f'Test accuracy: {accuracy:.4f}')
    print(f'Test F1 score: {f1_score:.4f}')