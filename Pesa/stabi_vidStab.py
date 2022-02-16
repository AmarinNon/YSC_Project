import os
import cv2
from vidstab import VidStab, layer_overlay



# Download test video to stabilize


# Initialize object tracker, stabilizer, and video reader
stabilizer = VidStab()
vidcap = cv2.VideoCapture(r'D:\Pesa\Video\test\U.mp4')

size = (int(vidcap.get(3)),int(vidcap.get(4)))
result = cv2.VideoWriter("Stabili_vidStab.mp4", 
                         cv2.VideoWriter_fourcc(*'MP4V'),
                         30, size)

# Initialize bounding box for drawing rectangle around tracked object

while True:
    grabbed_frame, frame = vidcap.read()

    # Pass frame to stabilizer even if frame is None
    stabilized_frame = stabilizer.stabilize_frame(input_frame=frame)

    # If stabilized_frame is None then there are no frames left to process
    if stabilized_frame is None:
        break

    # Display stabilized output
    result.write(stabilized_frame)
    cv2.imshow('Frame', stabilized_frame)

    key = cv2.waitKey(5)


    if key == 27:
        break
result.release()
vidcap.release()
cv2.destroyAllWindows()