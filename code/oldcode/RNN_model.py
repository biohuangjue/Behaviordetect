import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据
with open("E:\cuoshou\data.pkl", 'rb') as f:
    data = pickle.load(f)
with open("E:\cuoshou\labels.pkl", 'rb') as f:
    labels = pickle.load(f)

data = np.array(data)
labels = np.array(labels)
# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# 构建 RNN 模型
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 设置早停回调
early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

# 训练模型
history = model.fit(X_train, y_train, batch_size=32, epochs=50, callbacks=[early_stopping], validation_split=0.1)

# 评估模型
y_pred = model.predict(X_test)
y_pred = np.round(y_pred).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test accuracy: {accuracy}')

# 模型效果可视化（以准确率为例）
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# 调参示例（增加 LSTM 层的神经元数量）
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=32, epochs=50, callbacks=[early_stopping], validation_split=0.1)
y_pred = model.predict(X_test)
y_pred = np.round(y_pred).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test accuracy after tuning: {accuracy}')