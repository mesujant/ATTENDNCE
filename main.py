#from attendence import *
from dataset import  *
from model_preparation import *
from keras.models import load_model
import cv2
import numpy as np

class Attendace:
	def __init__(self):
		self.students = ['sujan1' , 'sujan2' , "sujan3"]

	def start_attendance(self):
		model = load_model('test_2.hd5')
		face_cascade = cv2.CascadeClassifier('haarcascade_frontal_face_default.xml')
		
		cap = cv2.VideoCapture(0)
		while True:
			ret , frame = cap.read()
			faces = face_cascade.detectMultiScale(frame ,1.1, 5)
			if len(faces):
				for x , y , w , h in faces:
					roi = frame[y:y+h , x:x+w]
					img = cv2.resize(roi , (28,28), cv2.INTER_AREA)
					img2 = np.resize(img, (1, 28,28,3))
					#print(type(resize_image))
					result = np.argmax(model.predict(img2), axis=-1)
					student = (self.students[result[0]])

					#print(model.predict(np.resize(img, (1,28,28,3))))
					frame=cv2.rectangle(frame, (x, y), (x+w, y+h),(0, 255,0),2)
					frame = cv2.putText(frame, student , (x, y-10),  cv2.FONT_HERSHEY_SIMPLEX ,1, (0, 255,0),2 )

			key = cv2.waitKey(1)

			if key == ord('q'):
				break
			cv2.imshow('frame' , frame)



if __name__ == '__main__':
	attendence = Attendace()
	attendence.start_attendance()
