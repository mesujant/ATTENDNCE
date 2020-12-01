import cv2
import os 
import random
import numpy as np 
#import threading as t

class Dataset:
	def __init__(self):
		self.faces_images_path = "/home/synced/Desktop/JUST_PYTHON/AI/ATTENDNCE/IMAGES/FACES/"
		self.faces = []
		self.faces_labels = []
	def from_webcam(self, person):
		if not os.path.exists("VIDEOS"):
			os.mkdir("VIDEOS")

		cap = cv2.VideoCapture(0)
		size = int(cap.get(3)), int(cap.get(4))
		while True:
			ret, frame = cap.read()
			key = cv2.waitKey(1)
			
			if key == ord('q'):
				cap.release()
				cv2.destroyAllWindows()
				break
			if key == ord('s'):
				#out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height)
				out = cv2.VideoWriter("VIDEOS/"+person+'.mp4', cv2.VideoWriter_fourcc('m','p','4','v'), 10, size) 				
				while True:
					cv2.namedWindow(person, cv2.WINDOW_AUTOSIZE)
					ret, frame = cap.read()
					out.write(frame)
					key = cv2.waitKey(1)
					if key == ord('q'):
						cv2.destroyWindow(person)
						break
					cv2.imshow(person, frame)
					print("saving image of ", person)

			cv2.imshow("frame", frame)
			#function to convert video into images
			self.from_video(person+".mp4")


	def from_video(self, video_file):
		file_name = video_file.split(".")[0]
		cap = cv2.VideoCapture("VIDEOS/"+video_file)
		print(cap.isOpened())
		if not cap.isOpened():
			print("error while reading file")
		i = 0
		while cap.isOpened():
			ret , frame = cap.read()
			if not ret:
				break
			if not os.path.exists("IMAGES"):
				os.mkdir("IMAGES")
			if not os.path.exists("IMAGES/"+file_name):
				os.mkdir("IMAGES/"+file_name)

			cv2.imwrite("IMAGES/"+file_name+"/"+file_name+str(i)+".jpg", frame)
			i += 1
			print(file_name+str(i))
			
			cv2.imshow("frame", frame)
		self.from_image(file_name)

	def from_image(self, folder_name):
		#extract only faces of all the images
		path0 = "/home/synced/Desktop/JUST_PYTHON/AI/ATTENDNCE/IMAGES/"
		path = path0+folder_name
		face_cascade = cv2.CascadeClassifier("haarcascade_frontal_face_default.xml")
		if not os.path.exists(path0+"FACES"):
			os.mkdir(path0+"FACES")
		
		if not os.path.exists(path0+"FACES/"+folder_name):
			os.mkdir(path0+"FACES/"+folder_name)
			self.faces_images_path = path0 + "FACES/"

		i=0
		for image in os.listdir(path):
			img = cv2.imread(path+"/"+image)
			faces = face_cascade.detectMultiScale(img, 1.1, 5)
			#print(faces)

			if len(faces):
				for x, y, w, h in faces:
					roi = img[y:y+h, x:x+w]
					cv2.imwrite(path0+"FACES/"+folder_name+"/face"+str(i)+".jpg", roi)
					i += 1
		print('face extracting done for ', folder_name)
 

	def load_faces_images(self):
		persons_name = os.listdir(self.faces_images_path)
		faces = []
		labels = []
		for person in persons_name:
			face_images = os.listdir(self.faces_images_path +"/"+person)
			for face in face_images:
				read_face = cv2.imread(self.faces_images_path+"/"+person+"/"+face , 0)	
				read_face = cv2.resize(read_face , (28, 28) , cv2.INTER_AREA)
				faces.append(read_face)
				labels.append(persons_name.index(person))
			 
		
		#generating more than 70% random data for trainning 
		#around 30% for testing
		train_index = {}
		while len(train_index) < int(0.7*len(faces)):
			train_index = {random.randint(0 , len(faces)-1) for i in range(len(faces))}
		#print(faces[0])
		test_index = {i for i in range(len(faces)) if i not in train_index}
		
		faces_train = [(faces[index]) for index in train_index]
		faces_test = [(faces[index]) for index in test_index]

		labels_train = [labels[index] for index in train_index]
		labels_test = [labels[index] for index in test_index]
		return (np.asarray(faces_train) , np.asarray(labels_train) , np.asarray(faces_test) , np.asarray(labels_test))
		
			
		

if __name__ == '__main__':
	d = Dataset()
	(a1 , b1) , (a2 , b2)= d.load_faces_images()
	print("train data:" , a1 , b1)
	print("************************")
	print("test data:" , a2 , b2)
	#train_no = int(len(faces) * 0.7)