import cv2

video = cv2.VideoCapture(0)

i = 0

while(True):
	i = i+1

	ret, frame = video.read()

	cv2.imshow('frame', frame)

	cv2.imwrite(filename = 'sample_images/sample' + str(i) +'.jpg', img = frame)	


	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

video.release()

