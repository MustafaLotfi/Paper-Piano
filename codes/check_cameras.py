import cv2

app_running = True
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

while cap0.isOpened() and cap1.isOpened and app_running:
	ret0, frame0 = cap0.read()
	ret1, frame1 = cap1.read()

	if ret0 and ret1:
		cv2.imshow("frame", frame0)
		q = cv2.waitKey(1)
		if (q == ord('q')) or (q == ord('Q')):
			app_running = False

cap0.release()
cap1.release()
cv2.destroyAllWindows()