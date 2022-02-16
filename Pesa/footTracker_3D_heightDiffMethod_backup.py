from turtle import distance
import cv2
import mediapipe as mp
import numpy as np
import math
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

R_up = False
L_up = False
prevR = 0
prevL = 0

distArr = []

default_H = 165

prevDist = 0
maxDist = 0
prevHeight = 0

dataAdded = False


# For webcam input:
cap = cv2.VideoCapture(r'F:\work\M.6\0.ME\ysc\code\phase 2_3D MediaPipe\Pesa\Video\pos_2\U.MOV')

# cap = cv2.VideoCapture(0)


class coordinate:
    x = 0
    y = 0
    z = 0
    na_me = ""
    
    def __init__(self, name):
        self.na_me = name
        
    def setCord(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        
    def updateCord(self, landmarks, width, height):
        self.x = (landmarks.x*width)
        self.y = int(landmarks.y*height)
        self.z = int(landmarks.z*width)
            
    def display(self):
        print(self.na_me, '->', 'x:', self.x, '/', 'y:', self.y, '/', 'z:', self.z)
        
    def getCord(self):
        return (self.x, self.y)


def getCord(landmarks, pos, width, height):
    return (int(landmarks[pos].x*width), int(landmarks[pos].y*height))

def getMid(cord1, cord2):
    return (int((cord1.x + cord2.x)/2), int((cord1.y + cord2.y)/2))


with mp_pose.Pose(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
    model_complexity=2,) as pose:
  while cap.isOpened():
    success, image = cap.read()
    # result_raw.write(image)
    width = int(cap.get(3))
    height = int(cap.get(4))
    

    

    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break

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
        #landmarks = results.pose_world_landmarks.landmark
        
        
        ry_Wrist = int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y*width)
        ly_Wrist = int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y*width)
        


        #ส้นเท้าข้างซ้าย
        LeftHeelVal = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value]
        LPos = coordinate("LPos")
        #LPos.setCord(int(LeftHeelVal.x*width), int(LeftHeelVal.y*height), int(LeftHeelVal.z*width))
        LPos.updateCord(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value], width, height)
    

        #ส้นเท้าข้างขวา
        RightHeelVal = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value]
        RPos = coordinate("RPos")
        RPos.setCord(int(RightHeelVal.x*width), int(RightHeelVal.y*height), int(RightHeelVal.z*width))
        
        
        #หาระยาทางระหว่างเท้า
        localDist = abs(RPos.z - LPos.z)
        
        
        #ขนาดตัว
        LShoulder = coordinate('LShoulder')
        LShoulder.updateCord(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], width, height)
        
        RShoulder = coordinate('RShoulder')
        RShoulder.updateCord(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], width, height)
        
        LHip = coordinate('LHip')
        LHip.updateCord(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], width, height)
        
        RHip = coordinate('RHip')
        RHip.updateCord(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value], width, height)
        
        #midShoulder = getMid(RShoulder, LShoulder)
        #midHip = getMid(RHip, LHip)
        
        localSize = round(abs(RHip.x - LHip.x),4)
        
        #put line
        # cv2.circle(image, midShoulder, 10, (255,0,255), cv2.FILLED)
        # cv2.circle(image, midHip, 10, (255,0,255), cv2.FILLED)
        # cv2.line(image, midShoulder, midHip, (0, 145, 255), 5)
        
        
        
        
        #หาความสูง
        LEye = getCord(landmarks, mp_pose.PoseLandmark.LEFT_EYE.value, width, height)
        REye = getCord(landmarks, mp_pose.PoseLandmark.RIGHT_EYE.value, width, height)
        
        midEye = (int((LEye[0] + REye[0])/2), int((LEye[1] + REye[1])/2))
        midFoot = (int((LPos.x + RPos.x)/2), int((LPos.y + RPos.y)/2))
        
        localHeight = int(abs(midFoot[1] - midEye[1]))
    
        
        #หาระยะห่างจริง
        RealDist = int((default_H * localDist) / localHeight)
        
        
  
        
        #cv2.circle(image, midFoot, 10, (255,0,255), cv2.FILLED)
        #cv2.line(image, midEye, midFoot, (0, 145, 255), 9)
        
        #put Text
        cv2.putText(image, 'L-> ' + str(LPos.z), (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 145, 255), 2)
        cv2.putText(image, 'R-> ' + str(RPos.z), (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 145, 255), 2)
        
        cv2.putText(image, 'LC-> ' + str(localSize), (20,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        #cv2.putText(image, 'D-> ' + str(localDist), (20,160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 145, 255), 2)
        
        cv2.putText(image, 'RD-> ' + str(RealDist), (20,160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 145, 255), 2)
        
        
        
        #logic section
        if prevHeight == 0:
            prevHeight = localSize
            print(prevHeight)
            
        prevDist = max(prevDist, RealDist)
        # if RealDist <= 5:
        #     if not dataAdded:
        #         dataAdded = True
        #         distArr.append( ((prevHeight - localSize) / prevHeight) * 100 )
        #         prevDist = 0
        #         prevHeight = localSize
        #         print(localSize, distArr)
        # else:
        #     dataAdded = False
        # maxDist = prevDist
        # RealDist < prevDist
        
        if RealDist < prevDist:
            #maxDist = prevDist
            
            if not dataAdded:
                dataAdded = True
                distArr.append( abs (round((((prevHeight - localSize) / prevHeight) * 100 ), 3)) )
                prevDist = 0
                prevHeight = localSize
                print(localSize, distArr[len(distArr) - 1])
                
        else:
            dataAdded = False
            
            
        
        
        
        
        
        
        
        # RPos.display()
        # LPos.display()
        # print()
        
        
        # # logic section
        # if LPos.y < RPos.y:
        #     L_up = False
        #     if not R_up:
        #         R_up = True
                
        #         if prevL == 0:
        #             prevL = LPos.x
        #         else:
        #             distArr.append(LPos.x - prevL)
        #             prevL = LPos.x
        #             print(distArr)
                      
        # if LPos.y > RPos.y:
        #     R_up = False
        #     if not L_up:
        #         L_up = True
                
        #         if prevR == 0:
        #             prevR = RPos.x
        #         else:
        #             distArr.append(RPos.x - prevR)
        #             prevR = RPos.x
        #             print(distArr)
            
    
        
                
              
    except:
        pass
    #เอาภาพออกมา
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2,circle_radius=2),
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2,circle_radius=2)
    )
    # result.write(image)
    image = cv2.resize(image, (int(width/2), int(height/2)))
    #image = cv2.resize(image, (int(width), int(height)))
    cv2.imshow('Mediapipe', image)

    

    if cv2.waitKey(5) & 0xFF == 27:
      break

'''
if len(result_distance_l) != 0:
    result_distance_l.pop(0)
    print("Result L",result_distance_l)
if len(result_distance_r) != 0:
    result_distance_r.pop(0)
    print("Result R",result_distance_r)
'''


# result.release()
# result_raw.release()
cap.release()
cv2.destroyAllWindows()




# Linear Regression
from sklearn import linear_model
import matplotlib.pyplot as plt


def convert_1d_to_2d(l, cols):
    return [l[i:i + cols] for i in range(0, len(l), cols)]

def outstand(arr):
    return arr
    hold = [arr[0]]
    for i in range(1, len(arr)):
        if hold[len(hold)-1] / arr[i] < 2 or arr[i] / hold[len(hold)-1] < 2:
            hold.append(arr[i])
    return hold
    

step = []
distance = outstand(distArr)
a = 0
avg = 0
# if result_distance_l is not None or result_distance_r is not None:
#     if len(result_distance_l) > len(result_distance_r):
#         for i in result_distance_l:
#             distance.append(i)  
#     if len(result_distance_r) > len(result_distance_l):
#         for i in result_distance_r:
#             distance.append(i)
#     if len(result_distance_r) == len(result_distance_l):
#         if sum(result_distance_r)  > sum(result_distance_l):
#             for i in result_distance_r:
#                 distance.append(i)
#         if sum(result_distance_l)  > sum(result_distance_r):
#             for i in result_distance_l:
#                 distance.append(i)



# print("Before: ",distance)  
# distance = outstand(distance)
# print("After: ",distance)
for i in range(len(distance)):
    step.append(i)



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
print("coef_ : ",reg.coef_)
y = reg.coef_ * x + reg.intercept_

plt.xlabel('step')
plt.ylabel('distance')
plt.scatter(step, distance, color='red', marker='+')
plt.plot(x, y, ':g' , label='y=mx+c')
plt.legend(loc='upper left')
plt.show()

