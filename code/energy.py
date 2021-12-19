import numpy as np
import cv2

from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.applications import imagenet_utils

mobile = MobileNet()
def prepare_image(img):
    img_resize = cv2.resize(img,(224,224))
    img_array_expanded_dims = np.expand_dims(img_resize, axis=0)
    return preprocess_input(img_array_expanded_dims)

bg_frame_start = 145
bg_frame_end = 165
bg_frame_cnt = bg_frame_end-bg_frame_start+1

img_frame_start = 0
img_frame_end = 144

frame_width = 640 
frame_height = 360 

cap = cv2.VideoCapture(r"src/final2.mp4")


if not cap.isOpened():
    print("Cannot read video")
    exit()

cnt = 0
cap.set(1, bg_frame_start)

bg_avg = np.zeros((frame_height,frame_width,3), dtype="uint8")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (frame_width, frame_height))
    
    bg_avg+=(frame//bg_frame_cnt)
    
    cnt += 1
    if(cnt==bg_frame_cnt):
        break
    
cv2.imshow("Background_Avg", bg_avg)
cv2.imwrite("results/Back_Avg.png", bg_avg)
cv2.waitKey()
# cv2.destroyAllWindows()
    

cap.set(img_frame_start,0)
cnt = 0

while True:
    
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (frame_width, frame_height))

    cv2.imshow("Video_Feed", frame)
    

    subt = cv2.subtract(frame,bg_avg)
    blur = cv2.GaussianBlur(subt,(5,5),3)

    x_min = frame_width
    y_min = frame_height
    x_max = 0
    y_max = 0

    en_thresh = 10000
   
    for y in range(0, frame_height, 10):
       for x in range(0, frame_width, 10):
           en_x_y = blur[y][x][0]**2 + blur[y][x][1]**2 + blur[y][x][2]**2
           if en_x_y>en_thresh:
            x_max = max(x_max,x)
            x_min = min(x_min,x)
            y_max = max(y_max,y)
            y_min = min(y_min,y)

    # add 20 px up down left right

    inc = 20
    x_max = min(x_max + inc, 1280)
    x_min = max(x_min - inc, 0)

    y_max = min(y_max + inc, 720)
    y_min = max(y_min - inc, 0)

    if (x_max <= x_min or y_max <= y_min): # if image is empty
        continue

    preprocessed_image = prepare_image(frame[y_min:y_max, x_min:x_max])
    predictions = mobile.predict(preprocessed_image)
    results = imagenet_utils.decode_predictions(predictions)
    conf = [i[2] for i in results[0] if 'tiger' in i[1]]
    
    if conf:
        # cv2.drawContours(frame, contours, -1, (0, 0, 255), 3)
        
        cv2.rectangle(frame,(x_min, y_min),(x_max, y_max),(0,255,0),3)
        cv2.putText(frame, f"Tiger : {conf[0]*100:.2f}%", (x_min,y_min-5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,255), 2)
        if(conf[0]>0.9):
            cv2.imwrite("results/energy_bounded.png",frame)

    cv2.imshow("Background_subtracted", subt)
    # cv2.imshow("Edges", edged)

    cv2.imshow("Bounds",frame)
    
    cv2.waitKey(10)

    cnt += 1
    if(cnt==img_frame_end): 
        break
    
cap.release()
cv2.destroyAllWindows()