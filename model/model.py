# import the necessary packages
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow .keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

class Model:
	@staticmethod
	def build(input_shape, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
         model = Sequential()
         model.add(Conv2D(filters=32, kernel_size=(3, 3),kernel_initializer='he_uniform', padding='same', input_shape=input_shape, activation='relu'))
         model.add(BatchNormalization(axis=-1))
         model.add(Conv2D(filters=32, kernel_size=(3, 3),kernel_initializer='he_uniform', padding='same', activation='relu'))
         model.add(MaxPooling2D((2, 2)))
         model.add(BatchNormalization(axis=-1))
         model.add(Dropout(0.1))

         model.add(Conv2D(filters=64, kernel_size=(3, 3),kernel_initializer='he_uniform', padding='same', activation='relu'))
         model.add(BatchNormalization(axis=-1))
         model.add(Conv2D(filters=64, kernel_size=(3, 3),kernel_initializer='he_uniform', padding='same', activation='relu'))
         model.add(MaxPooling2D((2, 2)))
         model.add(BatchNormalization(axis=-1))
         model.add(Dropout(0.1))

         model.add(Conv2D(filters=128, kernel_size=(3, 3),kernel_initializer='he_uniform', padding='same', activation='relu'))
         model.add(BatchNormalization(axis=-1))
         model.add(Conv2D(filters=128, kernel_size=(3, 3),kernel_initializer='he_uniform', padding='same', activation='relu'))
         model.add(MaxPooling2D((2, 2)))
         model.add(BatchNormalization(axis=-1))
         model.add(Dropout(0.1))

         model.add(Flatten())

         model.add(Dense(128,kernel_initializer='he_uniform', activation='relu'))
         model.add(BatchNormalization())
         model.add(Dropout(0.25))

         model.add(Dense(64,kernel_initializer='he_uniform', activation='relu'))
         model.add(BatchNormalization())
         model.add(Dropout(0.5))

         model.add(Dense(classes, activation='sigmoid'))

		# return the constructed network architecture
         return model