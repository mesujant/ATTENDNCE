import cv2
import os 
import threading as t

class Dataset:
	def __init__(self):
		pass

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
		i=0
		for image in os.listdir(path):
			img = cv2.imread(path+"/"+image)
			faces = face_cascade.detectMultiScale(img, 1.1, 5)
			print(faces)

			if len(faces):
				for x, y, w, h in faces:
					roi = img[y:y+h, x:x+h]
					cv2.imwrite(path0+"FACES/"+folder_name+"/face"+str(i)+".jpg", roi)
					i += 1
		print('face extracting done for ', folder_name)
 

if __name__ == '__main__':
	d = Dataset()
	#d.from_webcam("sujan")
	#d.from_video('sujan.mp4')
	d.from_image("sujan")