import vlc
import time
import cv2
import mediapipe as mp
import numpy as np
import time


# img_scale = 5
# img = cv2.imread("media/1.jpg")
# main_img_w, main_img_h = img.shape[1], img.shape[0]
# new_w, new_h = main_img_w//img_scale, main_img_h//img_scale

# img = cv2.resize(img, (new_w, new_h))

# img_mask = np.zeros((new_h, new_w, 3)).astype(np.uint8)
# # img_mask[((img[:, :, 0] < 150) & (img[:, :, 1] > 150) & (img[:, :, 2] < 150))] = 255
# img_mask[((img[:, :, 0] > 150) & (img[:, :, 1] < 150) & (img[:, :, 2] < 150))] = 255

# img_mask = img_mask.astype(np.uint8)
# # print(new_img.shape)
# # quit()
# img_clr_band = np.zeros((new_h, new_w, 3)).astype(np.uint8)
# img_clr_band[:, :, 1] = img[:, :, 1]

# cv2.imshow("img", img_mask)
# cv2.waitKey(0)
# quit()

p1 = vlc.MediaPlayer("files/A4.mp3")
p2 = vlc.MediaPlayer("files/Db4.mp3")
p3 = vlc.MediaPlayer("files/Ab4.mp3")

want2play = True
hands = mp.solutions.hands.Hands(static_image_mode=False,
	min_detection_confidence=0.5,
	min_tracking_confidence=0.5)

app_running = True
cap1 = cv2.VideoCapture(0)

while cap1.isOpened() and app_running:
	ret1, frame1 = cap1.read()

	if ret1:
		frame1 = cv2.flip(frame1, 1)
		frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

		lms = hands.process(frame1_rgb).multi_hand_landmarks
		landmarks = []
		if lms:
			for (i, lm0) in enumerate(lms):
				landmarks.append(np.array([(points.x, \
					points.y, points.z) for points in lm0.landmark]))
				if i == 1:
					break
			landmarks = np.array(landmarks)

			pos = [[None, None], [None, None]]
			dist = [None, None]
			pos[0] = (landmarks[0, 4, 0] + landmarks[0, 8, 0])/2,
			(landmarks[0, 4, 1] + landmarks[0, 8, 1])/2
			dist[0] = ((landmarks[0, 4, 0] - landmarks[0, 8, 0])**2 \
				+ (landmarks[0, 4, 1] - landmarks[0, 8, 1])**2)**0.5
			if i == 1:
				pos[1] = (landmarks[1, 4, 0] + landmarks[1, 8, 0])/2,
				(landmarks[1, 4, 1] + landmarks[1, 8, 1])/2
				dist[1] = ((landmarks[1, 4, 0] - landmarks[1, 8, 0])**2 \
					+ (landmarks[1, 4, 1] - landmarks[1, 8, 1])**2)**0.5
			
			if want2play:
				if dist[0] < 0.05:
					if pos[0][0] > 0.66:
						p1.play()
					elif 0.33 < pos[0][0] <= 0.66:
						p2.play()
					else:
						p3.play()
					want2play = False
			else:
				if not p1.is_playing() and not p2.is_playing() and not p3.is_playing():
					want2play = True
				# if dist[1] is not None and dist[1] < 0.05:

		cv2.imshow("Camera", frame1)
		q = cv2.waitKey(1)

		if (q == ord('q')) or (q == ord('Q')):
			app_running = False

cap1.release()
cv2.destroyAllWindows()
