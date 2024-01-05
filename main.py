#   Import modules, packages and/or libraries
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)

#   Callback Class
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

#   Creating instance for Callback Class
callbacks = myCallback()

#   Loading Data
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

print(y_train[0])
print("\n")
print(x_train[0])
print("\n")
plt.imshow(x_train[0])
plt.show()

#   Data Engineering (Reshaping)
x_train = x_train.reshape(60000, 28, 28, 1)
x_train = x_train / 255.0
x_test = x_test.reshape(10000, 28, 28, 1)
x_test = x_test / 255.0

#   Defining Model
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', input_shape = (28, 28, 1)),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.summary()

#   Compiling Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#   Fitting Model
model.fit(x_train, y_train, epochs=10, callbacks = [callbacks])

#   Evaluation on Testing Data
model.evaluate(x_test, y_test)

#   Prediction
classifications = model.predict(x_test)

#plt.imshow(x_test[0])
print(classifications[0])
print(y_test[0])
