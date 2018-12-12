# coding: utf-8
import numpy as np
import tensorflow as tf

import pretreatment


print(tf.__version__)
# 导入数据集
_, imgs = pretreatment.load_data()
labels = np.load('labels.npy')
# 探索数据
print(imgs.shape, labels.shape)
x, c, w, h = imgs.shape
imgs = imgs.reshape((x * c, w, h))
print(imgs.shape, labels.shape)
x, = labels.shape
t = np.zeros((x, 8))
t[:] = labels.reshape((x, 1))
labels = t.reshape(-1)
print(labels.shape)
imgs = imgs[:2 * 2000 * 8]
labels = labels[: 2 * 2000 * 8]
# 预处理数据
imgs = imgs / 255.0
# 构建模型
model = tf.keras.models.Sequential([
    # 将图像格式从二维数组转换成一维数组
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2000, activation=tf.nn.relu),
    tf.keras.layers.Dense(labels.max() + 1, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 训练模型
model.fit(imgs, labels, epochs=5)
# 评估准确率
test_loss, test_acc = model.evaluate(imgs, labels)
print('Test accuracy:', test_acc)
# 做出预测
predictions = model.predict(imgs)
print(np.argmax(predictions[0]))
