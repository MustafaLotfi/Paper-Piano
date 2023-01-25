# Paper-Piano
## Intro
A Virtual Music Keyboard has been designed and programmed using two cameras and hand landmarks detection models. This program consist of two modes:

**Music Control (MC) mode:**
Using the hand gesture, you can start to play the music or stop it,
increase or decrese the volume or go to the next song. 

**Paper Piano (2P) mode:**
You can play your custome song by the piano keys.

This is necessary to mention that after the project idea came to my mind,
I spend only 3 days to generate the code. Therefore this is normal if
the program isn't robust or optimal. Unfortunately I don't have
enough time to improve the app now. So I welcome your request for changing
the codes if you want.


## Demo

https://user-images.githubusercontent.com/53625380/214583911-ccb8c06c-a029-486d-b24f-24f81e22ec19.mp4

I don't know why, but you should go to the second 10 or more to can watch the demo. Also don't forget to unmute the video.

## More Details:
Firstly hand landmarks in both of the images of cameras are detected.
In the MC mode, by your hand gesture or its movement you can control the app.
This part has been implemented using distances between various hand landmarks and
the their movement speeds. Information from to images are processed together in
each moment and will be used for a desired action.
In the 2P mode, the top view camera is used to find the hands are in which section
or key. The front view camera is used to determine the hands are below of a
desired threshold near the table. It shows the hand touched the table or not.

## App Cons
1. Not being robust:
For example, the program is very sensitive to hand gestures. This means
It is possible that you move your hand randomly and the app do a non-desired action.
2. Not being optimal:
Because of low time dedication to the app, absolutely the functions aren't the
best ones.
3. Weak method to detect piano key lines:
The fact is I have not implemented any method to detect lines of paino keys
at all. I considered threshold boundaries in the image. It's better to use
methods such as Canny Edge Detector to find the keys. Because of this poor method,
if the top camera change a little, the app will fail to work correctly.

## How to Run
1. Clone the repo. write in cmd this:

`git clone <repo link>`

2. Create virutal environment, activate it, install packages:

`pip install -r requirements.txt`

3. Make sure both of cameras are working correctly. You can use `check_cameras.py`
in `codes` folder.

4. run `main.py`:

`python main.py`
