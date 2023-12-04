import cv2 as cv
import cv2 as cv2
import mediapipe as mp
import time
import utils, math
import pandas as pd
from pynput.keyboard import Key,Controller
keyboard = Controller()
from PIL import ImageTk
# Fast Ai
from fastbook import *
import tkinter as tk
import tkinter.filedialog as filedialog

# variables 
frame_counter = 0
CEF_COUNTER = 0
TOTAL_BLINKS = 0
# constants
CLOSED_EYES_FRAME = 3
start = 0
end = 0
ch = 0

FONTS = cv.FONT_HERSHEY_COMPLEX

# Face bounder indices 
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
# Lips indices for landmarks
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
# Left eyes indices 
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
# Right eyes indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]  

map_face_mesh = mp.solutions.face_mesh

def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    # List of (x,y) coordinates
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw:
        [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

    # Returning the list of tuples for each landmark 
    return mesh_coord

def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

def detectFACE(img, landmarks, FACE):
    # FACE coordinates
    FACE_points = [landmarks[idx] for idx in FACE]

    # Find the minimum and maximum x and y coordinates of the FACE points
    x_values = [point[0] for point in FACE_points]
    y_values = [point[1] for point in FACE_points]
    FACE_x_min = min(x_values)
    FACE_x_max = max(x_values)
    FACE_y_min = min(y_values)
    FACE_y_max = max(y_values)

    FACE_center_x = (FACE_x_min + FACE_x_max) / 2
    FACE_center_y = (FACE_y_min + FACE_y_max) / 2
    # นำค่า x และ y ที่เป็นจุดศูนย์กลางมาเก็บในลิสต์
    FACE_center_x = int(FACE_center_x)
    FACE_center_y = int(FACE_center_y)
    cv.circle(img, (FACE_center_x, FACE_center_y), 5, utils.RED, -1)

    cam_center_x  = int(img.shape[1] / 2)
    cam_center_y = int(img.shape[0] / 2)
    cv2.circle(img, (cam_center_x, cam_center_y), 5, (0, 0, 255), -1)

    distance_x = (FACE_center_x - cam_center_x)

    # Increase width and height of the rectangle
    width_increase = 25
    height_increase = 15
    FACE_x_min -= width_increase
    FACE_x_max += width_increase
    FACE_y_min -= height_increase
    FACE_y_max += height_increase

    faceRatio = euclaideanDistance((FACE_y_max, FACE_y_min), (FACE_y_min, FACE_y_max))
    re_face = faceRatio

    # Draw rectangle around lips
    cv.rectangle(img, (FACE_x_min, FACE_y_min), (FACE_x_max, FACE_y_max), utils.GREEN, 2)

    return re_face,distance_x
 
def detecteye(img, landmarks, right_indices, left_indices):

    # Right eye
    # Horizontal line 
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # Vertical line 
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]

    # Left eye 
    # Horizontal line 
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]
    # Vertical line 
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance


    # Draw lines on eyes 
    # Right eye
    eye_right_x_min = min([landmarks[idx][0] for idx in RIGHT_EYE])
    eye_right_x_max = max([landmarks[idx][0] for idx in RIGHT_EYE])
    eye_right_y_min = min([landmarks[idx][1] for idx in RIGHT_EYE])
    eye_right_y_max = max([landmarks[idx][1] for idx in RIGHT_EYE])

    # Increase width of rectangle
    width_increase = 20
    eye_right_x_min -= width_increase
    eye_right_x_max += width_increase
    eye_right_y_min -= width_increase
    eye_right_y_max += width_increase

    # Left eye
    eye_left_x_min = min([landmarks[idx][0] for idx in LEFT_EYE])
    eye_left_x_max = max([landmarks[idx][0] for idx in LEFT_EYE])
    eye_left_y_min = min([landmarks[idx][1] for idx in LEFT_EYE])
    eye_left_y_max = max([landmarks[idx][1] for idx in LEFT_EYE])

    # Increase width of rectangle
    eye_left_x_min -= width_increase
    eye_left_x_max += width_increase
    eye_left_y_min -= width_increase
    eye_left_y_max += width_increase

    # Draw rectangle around right eye
    cv.rectangle(img, (eye_right_x_min, eye_right_y_min), (eye_right_x_max, eye_right_y_max), utils.GREEN, 2)
    cv.rectangle(img, (eye_left_x_min, eye_left_y_min), (eye_left_x_max, eye_left_y_max), utils.GREEN, 2)
    cv.polylines(img,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
    cv.polylines(img,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)

    # Left eye
    eye_left_x_min = min([landmarks[idx][0] for idx in LEFT_EYE])
    eye_left_x_max = max([landmarks[idx][0] for idx in LEFT_EYE])
    eye_left_y_min = min([landmarks[idx][1] for idx in LEFT_EYE])
    eye_left_y_max = max([landmarks[idx][1] for idx in LEFT_EYE])

    # Increase width of rectangle
    eye_left_x_min -= width_increase
    eye_left_x_max += width_increase
    eye_left_y_min -= width_increase
    eye_left_y_max += width_increase

    re_right_m = reRatio
    re_left_m = leRatio

    return(re_right_m,re_left_m)

def detectYawn(img, landmarks, LIPS):
    # Lips coordinates
    lips_points = [landmarks[idx] for idx in LIPS]

    # Find the minimum and maximum x and y coordinates of the lips points
    x_values = [point[0] for point in lips_points]
    y_values = [point[1] for point in lips_points]
    lips_x_min = min(x_values)
    lips_x_max = max(x_values)
    lips_y_min = min(y_values)
    lips_y_max = max(y_values)

    # Increase width and height of the rectangle
    width_increase = 25
    height_increase = 20
    lips_x_min -= width_increase
    lips_x_max += width_increase
    lips_y_min -= height_increase
    lips_y_max += height_increase

    cv.polylines(img,  [np.array([mesh_coords[p] for p in LIPS ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
    cv.rectangle(img, (lips_x_min, lips_y_min), (lips_x_max, lips_y_max), utils.GREEN, 2)
    yeRatio = euclaideanDistance((lips_y_max, lips_y_min), (lips_y_min, lips_y_max))

    re_yawn = yeRatio
    return re_yawn

#=================================================Start========================================================================#
# Variables for counting
blink_right_counter = 0 
blink_left_counter = 0  
blink_right_counter_n = 0 
blink_left_counter_n = 0 
yawn_counter = 0        
blink_right = 0         
blink_left = 0          
re_yawn_counter = 0     
re_yawn_counter_n = 0  
close_eye_right = 0
close_eye_right_counter = 0
close_eye_left = 0
close_eye_left_counter = 0
# variables 
frame_counter = 0 
leCEF_COUNTER = 0
leTOTAL_BLINKS = 0
# constants
leCLOSED_EYES_FRAME = 3
reCEF_COUNTER = 0
reTOTAL_BLINKS = 0
# constants
reCLOSED_EYES_FRAME = 3
key = 0
TOTAL_Yawn = 0
re_yawn2 = ""
re_right2 = ""
re_left2 = ""

aa = 0
c = 3

window = tk.Tk()
window.title("Facial exercise with DAI-NO")

video = cv2.VideoCapture(0)
with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    start_time = time.time()
    start_n = time.time()
    frame_count = 0          

    # Starting video loop
    while True:
        re_face = "face NOT found"
        frame_counter += 1 # Frame counter
        ret, frame = video.read() # Get frame from camera
        if not ret:
            break # No more frames, break
        n_current_time = time.time()
        n_elapsed_time = n_current_time - start_n
        minutes, seconds = divmod(n_elapsed_time, 60)

        # Resize frame
        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        #frame = cropped_frame(frame)
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
          mesh_coords = landmarksDetection(frame, results, False)
          reface,center = detectFACE(frame, mesh_coords, FACE_OVAL)
          reEYE,leEYE = detecteye(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
          reYawn = detectYawn(frame, mesh_coords,LIPS)

          if reface > 0 :
                re_face = "face found"
          else:
                re_face = "face NOT found"
            
          if reface <= 450 :
                frame = utils.textWithBackground(frame, f"Please move your face closer to the camera.", FONTS, 1, (75, 300), bgOpacity=0.9, textThickness=2)
          
          else :
            if reface >= 550 :
                frame = utils.textWithBackground(frame, f"Please move your face go far to the camera.", FONTS, 1, (75, 300), bgOpacity=0.9, textThickness=2)
            if reEYE > 4.7 :
                re_right = "close eye"
                re_right2 = "close eye"
                reCEF_COUNTER += 1
                key = 1
            else:
                re_right = "open eye"
                if reCEF_COUNTER > reCLOSED_EYES_FRAME:
                    reTOTAL_BLINKS += 1
                    reCEF_COUNTER = 0

            if re_right2 == "close eye" and  re_right == "open eye" :
                reTOTAL_BLINKS += 1
                re_right2 = "open eye"

            if leEYE > 4.7 :                                                      
                leCEF_COUNTER += 1
                re_left = "close eye"
                re_left2 = "close eye"
                key = 1                         
            else:
                re_left = "open eye"
                if leCEF_COUNTER > leCLOSED_EYES_FRAME:
                    leCEF_COUNTER = 0  

            if re_left2 == "close eye" and  re_left == "open eye" :
                leTOTAL_BLINKS += 1
                re_left2 = "open eye"           

            if reYawn > 150 :
                key = 2
                re_yawn2 = "open mouth"
                re_yawn = "open mouth"
                #keyboard.press(Key.down)                      
            else: 
                re_yawn = "close mouth"
                #keyboard.release(Key.down)

            if re_yawn2 == "open mouth" and  re_yawn == "close mouth" :
                TOTAL_Yawn += 1
                re_yawn2 = "close mouth"                

            if key == 2 :
                key = 0
            #elif key == 1 :
                #keyboard.press(Key.space)
                #keyboard.release(Key.space)

            if center <= -50 :
                frame = utils.textWithBackground(frame, f'Left : {center}', FONTS, 1.0, (30, 300), bgOpacity=0.9, textThickness=2)
            elif center >= 50 :
                frame = utils.textWithBackground(frame, f'Right : {center}', FONTS, 1.0, (30, 300), bgOpacity=0.9, textThickness=2)
            else :
                frame = utils.textWithBackground(frame, f'Center : {center}', FONTS, 1.0, (30, 300), bgOpacity=0.9, textThickness=2)

            frame = utils.textWithBackground(frame, f'Eye right : {re_right}', FONTS, 1.0, (30, 150), bgOpacity=0.9, textThickness=2)
            frame = utils.textWithBackground(frame, f'Eye left  : {re_left}', FONTS, 1.0, (30, 200), bgOpacity=0.9, textThickness=2)
            frame = utils.textWithBackground(frame, f'Mouth : {re_yawn}', FONTS, 1.0, (30, 250), bgOpacity=0.9, textThickness=2)
            frame = utils.textWithBackground(frame, f'Re Face : {reface}', FONTS, 0.5, (650, 50), bgOpacity=0.45, textThickness=1)
            frame = utils.textWithBackground(frame, f'Re eye right : {reEYE}', FONTS, 0.5, (650, 100), bgOpacity=0.45, textThickness=1)
            frame = utils.textWithBackground(frame, f'Re eye left : {leEYE}', FONTS, 0.5, (650, 150), bgOpacity=0.45, textThickness=1)
            frame = utils.textWithBackground(frame, f'Re Mouth : {reYawn}', FONTS, 0.5, (650, 200), bgOpacity=0.45, textThickness=1)
            frame = utils.textWithBackground(frame, f'TOTAL Re eye left : {leTOTAL_BLINKS}', FONTS, 0.5, (650, 250), bgOpacity=0.45, textThickness=1)
            frame = utils.textWithBackground(frame, f'TOTAL Re eye right : {reTOTAL_BLINKS}', FONTS, 0.5, (650, 300), bgOpacity=0.45, textThickness=1)
            frame = utils.textWithBackground(frame, f'TOTAL Re Mouth : {TOTAL_Yawn}', FONTS, 0.5, (650, 350), bgOpacity=0.45, textThickness=1)
            

        # Calculate frame per second (FPS)
        end_time = time.time() - start_time
        fps = (frame_counter / end_time)

        frame = utils.textWithBackground(frame, f'FPS : {round(fps,1)}', FONTS, 1.0, (30, 50), bgOpacity=0.9, textThickness=2)
        frame = utils.textWithBackground(frame, f'FACE : {re_face}', FONTS, 1.0, (30, 100), bgOpacity=0.9, textThickness=2)
        frame = utils.textWithBackground(frame, "Elapsed Time: {:02d}:{:02d}".format(int(minutes), int(seconds)), FONTS, 0.5, (10, 495), bgOpacity=0.45, textThickness=1)
        frame = utils.textWithBackground(frame, f"Press the 'q' or 'Q' button to close the program", FONTS, 0.5, (10, 525), bgOpacity=0.45, textThickness=1)

        cv.imshow('Facial_exercise_with_DAI-NO',frame)

        key = cv.waitKey(2)
        if key == ord('q') or key == ord('Q') :
            break

cv.destroyAllWindows()
if start == 1 :
    video.release()

def Conclude_mode(TL,TR,TY,TEM,TES) :
    image = Image.open("LOGO.jpg")
    photo = ImageTk.PhotoImage(image)
    label = tk.Label(window, image=photo)
    label.pack()
    window.geometry("800x600")
    text_label = tk.Label(window, text= " " , font=("Helvetica", 24))
    text_label.pack()
    text_label = tk.Label(window, text= f"จำนวนการกระพริบตา ซ้าย ทั้งหมด {TL} ครั้ง" , font=("Helvetica", 24), anchor="w" , padx=10)
    text_label.pack()
    text_label = tk.Label(window, text= f"จำนวนการกระพริบตา ขวา ทั้งหมด {TR} ครั้ง" , font=("Helvetica", 24), anchor="w" , padx=10)
    text_label.pack()
    text_label = tk.Label(window, text= f"จำนวนการอ้ากปากทั้งหมด {TY} ครั้ง" , font=("Helvetica", 24), anchor="w" , padx=10)
    text_label.pack()
    text_label = tk.Label(window, text= "จำนวนเวลาที่ใช้ไปทั้งหมด {:02d}:{:02d} นาที".format(TEM, TES) , font=("Helvetica", 24), anchor="w" , padx=10)
    text_label.pack()

    if start == 1:
        window.destroy()
        return video
    window.mainloop()

Conclude_mode(leTOTAL_BLINKS,reTOTAL_BLINKS,TOTAL_Yawn,int(minutes), int(seconds))