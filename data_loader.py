import os
import numpy as np 
from PIL import Image


from config import slicepath
from config import sliceSize
from config import batchSize


def getProcessedData(img,imageSize):
    img = img.resize((imageSize,imageSize), resample=Image.ANTIALIAS)
    imgData = np.asarray(img, dtype=np.uint8).reshape(imageSize,imageSize,1)
    imgData = imgData/255.
    return imgData

#Returns numpy image at size imageSize*imageSize
def getImageData(filename,imageSize):
    img = Image.open(filename)
    imgData = getProcessedData(img, imageSize)
    return imgData


class Data_Loader():
	def __init__(self):
		self.genres = os.listdir(slicepath)
		self.data = []
		self.labels = []
		for genre in self.genres:
			files = os.listdir(slicepath+genre)
			for filename in files:
				img = getImageData(slicepath+genre+'/'+filename, sliceSize)
				label = [1 if genre == g else 0 for g in self.genres]
				self.data.append(img)
				self.labels.append(label)

		self.data = np.array(self.data)
		self.labels = np.array(self.labels)

		permutations = np.random.permutation(range(len(self.data)))
		self.data = self.data[permutations]
		self.labels = self.labels[permutations]

		threshold = int(len(self.data)*0.8)
		self.train_x = self.data[:threshold]
		self.train_y = self.labels[:threshold]

		self.test_x = self.data[threshold:]
		self.test_y = self.labels[threshold:]



	def set_pointer(self):
		self.pointer = 0

	def next_batch(self):
		start = self.pointer*batchSize
		end = (self.pointer+1)*batchSize
		self.pointer+=1

		return self.train_x[start:end], self.train_y[start:end]

	def test_data(self):
		return self.test_x, self.test_y

if __name__ == '__main__':
	data_loader = Data_Loader()
