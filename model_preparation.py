
from tensorflow import keras
from tensorflow.keras import models, layers
#from keras.datasets import mnist
from dataset import Dataset
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, Dropout, BatchNormalization
from keras.models import Sequential


'''
class Dataset:
	def __init__(self , x_train , y_train , x_test , y_test):
		(self.x_train , self.y_train), (self.x_test , self.y_test) = x_train , y_train , x_test , y_test#mnist.load_data()
		self.x_valid , self.x_train = self.x_train[:5000] , self.x_train[5000:]
		self.y_valid , self.y_train = self.y_train[:5000] , self.y_train[5000:]

	def view_data(self):
		from matplotlib import pyplot as plt
		plt.figure(figsize=(10, 10))
		for i in range(25):
			plt.subplot(5, 5, i+1)
			plt.xticks([])
			plt.yticks([])
			plt.grid(False)
			plt.imshow(self.x_train[i] , cmap=plt.cm.binary)
			plt.xlabel(self.y_train[i])
		plt.show()

	def return_data(self):
		
		return self.x_train , self.y_train , self.x_test , self.y_test
'''

class ModelPreparation:
	def __init__(self , x_train , y_train , x_test , y_test):
		test_no = int(0.1*len(x_test))
		self.x_train = x_train[test_no:]
		self.x_train = self.x_train / 255.
		self.y_train = y_train[test_no:] 
		
		self.x_test = x_test / 255.
		self.y_test = y_test
		
		self.x_valid = x_train[:test_no]
		self.x_valid = self.x_valid / 255.
		self.y_valid = y_test[:test_no]

	def create_network(self):
		self.model = Sequential()
		self.model.add(InputLayer(input_shape=(28,28,3)))

		self.model.add(Conv2D(25, (5, 5), activation="relu"))
		self.model.add(MaxPool2D((2,2)))

		self.model.add(Conv2D(25, (3,3), activation="relu"))
		self.model.add(MaxPool2D((2,2)))
		self.model.add(BatchNormalization())

		self.model.add(Conv2D(70, (3,3), activation="relu"))
		self.model.add(MaxPool2D((2,2)))
		self.model.add(BatchNormalization())

		self.model.add(Flatten())
		self.model.add(Dense(100, activation="relu"))
		self.model.add(Dense(50, activation="relu"))
		self.model.add(Dropout(0.25))

		self.model.add(Dense(3, activation="softmax"))
		print(self.model.summary())
		input()
	 
	def start_trainning(self):
	#def start_trainning(self):
		#model = self.create_network()
		#print(network.summary())
		self.create_network()
		self.model.compile(loss="sparse_categorical_crossentropy", 
						optimizer="sgd", 
						metrics=["accuracy"])

		history = self.model.fit(self.x_train, self.y_train , 
							epochs=2, 
							validation_data = (self.x_valid , self.y_valid))		

	def return_model(self):
		self.start_trainning()
		self.model.save('test_2.hd5')
		return self.model

	def evaluate_mode(self):
		loss , acc = self.model.evaluate(self.x_test, self.y_test)
		print(loss , acc)

		pass



if __name__ == '__main__':
	#x_train , y_train , x_test , y_test = d.return_data()
	d= Dataset()
	x_train, y_train, x_test, y_test = d.load_faces_images()
	m = ModelPreparation(x_train , y_train , x_test , y_test)
	#m.start_trainning()
	model = m.return_model()
	#model.start_trainning()


	