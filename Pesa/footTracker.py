import cv2
import mediapipe as mp
import numpy as np
import csv 
import time
import os



mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


counter_l = 0
stage_l = None
counter_r = 0
stage_r = None

distance_l = 0
distance_r = 0
result_distance_l = []
result_distance_r = []
show_l = "0"
show_r = "0"

check_start_r = None
lock_foot = None
check_time_r = None

# For webcam input:
# cap = cv2.VideoCapture(r'D:\projectF\Video\IMG_9414_1.mp4')
cap = cv2.VideoCapture(0)

# name_Video = input('Enter Video name:')
# size = (int(cap.get(3)), int(cap.get(4)))
# result = cv2.VideoWriter(str(name_Video)+".avi", 
#                          cv2.VideoWriter_fourcc(*'MJPG'),
#                          30, size)
# result_raw = cv2.VideoWriter(str(name_Video)+"_raw.avi", 
#                          cv2.VideoWriter_fourcc(*'MJPG'),
#                          24, size)





with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    # result_raw.write(image)
    width = int(cap.get(3))
    height = int(cap.get(4))
    

    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #make detection
    results = pose.process(image)

    # Draw the pose annotation on the image. also make image to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # cv2.line(image,(0,1025),(width,1025),(0,255,0),thickness=5)


    #แตก landmark
    try:
        landmarks = results.pose_landmarks.landmark
        
        cv2.putText(image,show_l,(20,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 145, 255),2,cv2.LINE_AA)
        cv2.putText(image,show_r,(20,80), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 145, 255),2,cv2.LINE_AA)
        
        ry_Wrist = int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y*width)
        ly_Wrist = int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y*width)
        

        if (ry_Wrist <= int(height/2)-100 or ly_Wrist <= int(height/2)-100) and check_start_r is None:
            check_start_r = "Yes"
            print("Start")
        

        #ส้นเท้าข้างซ้าย
        lx = int(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x*width)
        ly = int(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y*height)
        #แปะจุด เท้าข้างซ้าย
        cv2.circle(image,(lx,ly),10,(255,0,0),cv2.FILLED)

        #ส้นเท้าข้างขวา
        rx = int(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x*width)
        ry = int(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y*height)
        #แปะจุดเท้าข้างขวา
        cv2.circle(image,(rx,ry),10,(255,0,255),cv2.FILLED)
        
        # logic section
        
        if check_start_r == "Yes" and check_time_r == None:
            check_time_r = time.time()
            print("check_start_r = Yes")
        
        if check_start_r == "Yes" and time.time()-check_time_r >= 3 and lock_foot is None:
            lock_foot = ry
            lock_foot = lock_foot - 15
        
        if lock_foot is not None:
            cv2.line(image,(0,lock_foot),(width,lock_foot),(0,255,0),thickness=5)
            # logic เท้าซ้าย
            if ry<lock_foot:
                stage_r = 'up'
            if ry>lock_foot and stage_r =='up':
                stage_r= 'down'
                show_r = "rx: "+str(rx)
                result_distance_r.append(abs(rx-distance_r))
                distance_r = rx
                counter_r+=1
                print("R: ",counter_r)
                print("Rx: ",rx)
            
            # logic เท้าขวา
            
            if ly < lock_foot:
                stage_l = 'up'
            if ly > lock_foot and stage_l == 'up':
                stage_l = 'down'
                show_l = "lx: "+str(lx)
                result_distance_l.append(abs(lx-distance_l))
                distance_l = lx
                counter_l+=1
                print("L: ",counter_l)
                print("Lx: ",lx)
                
        
                
              
    except:
        pass
    #เอาภาพออกมา
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2,circle_radius=2),
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2,circle_radius=2)
    )
    # result.write(image)
    cv2.imshow('Mediapipe', image)

    

    if cv2.waitKey(5) & 0xFF == 27:
      break

if len(result_distance_l) != 0:
    result_distance_l.pop(0)
    print("Result L",result_distance_l)
if len(result_distance_r) != 0:
    result_distance_r.pop(0)
    print("Result R",result_distance_r)

cap.release()
# result.release()
# result_raw.release()
cv2.destroyAllWindows()

# dirname = os.path.dirname(os.path.abspath(__file__))
# csvfilename = os.path.join(dirname, 'Diagnose.csv')
# file_exists = os.path.isfile(csvfilename)
# with open(csvfilename, 'a',newline='') as csvfile: 
#     fields = ['Hz','Distance'] 
#     csvwriter = csv.DictWriter(csvfile,fieldnames=fields) 
#     if not file_exists:
#       csvwriter.writeheader()
#     if result_distance_l is not None or result_distance_r is not None:
#         if len(result_distance_l) > len(result_distance_r):
#             csvwriter.writerow({'Distance':result_distance_l})
#         elif len(result_distance_r) > len(result_distance_l):
#             csvwriter.writerow({'Distance':result_distance_r})

from sklearn import linear_model
import matplotlib.pyplot as plt
step = []
distance = []
a = 0
status = ""
def convert_1d_to_2d(l, cols):
    return [l[i:i + cols] for i in range(0, len(l), cols)]


if result_distance_l is not None or result_distance_r is not None:
    if len(result_distance_l) > len(result_distance_r):
        for i in result_distance_l:
            distance.append(i)
            step.append(a)
            a += 1   
    elif len(result_distance_r) > len(result_distance_l):
        for i in result_distance_r:
            distance.append(i)
            step.append(a)
            a += 1



twoD_step = convert_1d_to_2d(step,1)
twoD_distance = convert_1d_to_2d(distance,1)


reg = linear_model.LinearRegression()
reg.fit(twoD_step, twoD_distance)

if reg.predict([[len(distance) + 1]]) < reg.predict([[len(distance) + 2]]):
    print("Normal")
    status = "Normal"
elif reg.predict([[len(distance) + 1]]) > reg.predict([[len(distance) + 2]]):
    print("Slow gait")
    status = "Slow gait"

x = np.linspace(step[0], step[-1:],100)
y = reg.coef_ * x + reg.intercept_

plt.xlabel('step')
plt.ylabel('distance')
plt.scatter(step, distance, color='red', marker='+')
plt.plot(x, y, ':g' , label='y=mx+c')
plt.legend(loc='upper left')
plt.show()
    