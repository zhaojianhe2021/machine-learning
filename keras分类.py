import tensorflow as tf
from tensorflow import  keras
import numpy as np

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
print(X_train_full.shape, X_train_full.dtype)

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0 # 255.0 转换为浮点数
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

class_names = ['T-shirt/top',"Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
print(class_names[y_train[0]])
'''
如果层之间没有非线性，那么即使是很深的层堆叠也等同于单个 层，这样你无法解决非常复杂的问题。相反，具有非线性激活函数的足够大的DNN理论 上可以近似任何连续函数
'''
# 建立神经网络
model = keras.models.Sequential() # 称为顺序API，Sequential
model.add(keras.layers.Flatten(input_shape=[28, 28])) # 将每个输入图像转换为一维度组，计算X.reshape（-1，1）
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu")) # activation="relu"等效于activation=keras.activations.relu
model.add(keras.layers.Dense(10, activation="softmax"))

'''
可以不用像我们刚才那样逐层添加层，而可以在创建顺序模型时传递一个层列表：
model = keras.models.Sequential([ 
    keras.layers.Flatten(input_shape=[28, 28]), 
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"), 
    keras.layers.Dense(10, activation="softmax") 
])
'''
print(model.summary())
print(model.layers)
hidden1 = model.layers[1] # 按索引获取层
print(hidden1.name) # 按名称获取

weights, biases = hidden1.get_weights() # get_weights()访问层的所有参数
# print(weights,'shape:',weights.shape)
# print(biases.shape)

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=["accuracy"])
# 训练模型
history = model.fit(X_train,y_train,epochs=30,validation_data=(X_valid, y_valid) )
import pandas as pd
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot(figsize=(8, 5))#包含在训练集和验证集上的每个轮次结束时测得的损失和额外指标的字典（history.history）,使用此字典创建pandas DataFrame
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
# plt.show()

#evaluate()用于评估已经训练过的模型.返回损失值&模型的度量值.
#在测试集上对其进行评估泛化误差

print(model.evaluate(X_test, y_test))

#使用模型进行预测
X_new = X_test[:3]
y_proba = model.predict(X_new)
print(y_proba)#round() 方法返回浮点数x的四舍五入值。

#predict_classes()方法进行预测时，返回的是类别的索引，即该样本所属的类别标签
y_pred = model.predict_classes(X_new)
print(y_pred)
print(np.array(class_names)[y_pred])























































