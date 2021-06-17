import tensorflow as tf
import matplotlib.pyplot as plt
import numpy 
import pandas as pd
from sklearn.model_selection import train_test_split
mnist = pd.read_csv('fashion-mnist_train.csv')
x = mnist.iloc[0:,1:]
y = mnist['label']
x = numpy.array(x)
y = numpy.array(y)
y = y.reshape(y.shape[0],1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
import matplotlib
for i in range(0,10):  # ploting the dataset
    data_digit=x_train[i]
    data_digit_image=data_digit.reshape(28,28)
    plt.imshow(data_digit_image,cmap=matplotlib.cm.binary,interpolation='nearest')
    plt.axis('off')
    plt.figure(i+1)
plt.show()
x_train = x_train.reshape(42000,28,28,1)
x_train = x_train/255.0
x_test = x_test.reshape(18000,28,28,1)
x_test = x_test/255.0
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(10,activation='softmax')
        
    ]
)
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics = ['accuracy'])
model.summary()
model.fit(x_train,y_train,epochs=5)
model.evaluate(x_test,y_test) # 90% accuracy in just 5 epochs