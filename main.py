"""
Project name: Paper-Piano
Programmer: Mostafa Lotfi
Date: 1/1/2023
Brief Description: A Virtual Music Keyboard has been designed and programmed
using two cameras and hand landmarks detection models.
This program consist of two modes:
Mode 1: Music Control (MC):
Using the hand gesture, you can start to play the music or stop it,
increase or decrese the volume or go to the next song. 
Mode 2: Paper Piano (2P):
You can play your custome song by the piano keys.
This is necessary to mention that after the project idea came to my mind,
I spend only 3 days to generate the code. Therefore this is normal if
the program isn't robust or optimal. Unfortunately I don't have
enough time to improve the app now. So I welcome your request for changing
the codes if you want.
"""

from codes.virtual_music import VirtualMusic

music = VirtualMusic()

music.run()