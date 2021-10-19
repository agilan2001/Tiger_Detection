import numpy as np
import cv2

bg_frame_start = 145
bg_frame_end = 165
bg_frame_cnt = bg_frame_end-bg_frame_start+1

img_frame_start = 0
img_frame_end = 144

frame_width = 1280
frame_height = 720

cap = cv2.VideoCapture(r"src/final2.mp4")


if not cap.isOpened():
    print("Cannot read video")
    exit()

cnt = 0
cap.set(1, bg_frame_start)

bg_avg = np.zeros((720,1280,3), dtype="uint8")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        break
    
    bg_avg+=(frame//bg_frame_cnt)
    
    cnt += 1
    if(cnt==bg_frame_cnt):
        break
    
    

cap.set(img_frame_start,0)
cnt = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        break

    # if cnt==0:
    #     cv2.imshow("Frame", frame)
    #     cv2.imwrite("results/Frame.png", frame)
    
    
    # print(cnt, frame[0][0] - [1, 2, 3])
    subt = cv2.subtract(frame,bg_avg)
    blur = cv2.GaussianBlur(subt,(9,9),2)
    edged = cv2.Canny(blur, 250, 250) 
    (contours, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = imutils.grab_contours(contours)
    # print(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:20]
    # print( contours)
    # break

    x_min = 1280
    y_min = 720
    x_max = 0
    y_max = 0
                                                                    #contours: [contour[segment[line [ point [x1,y1]] ], [] ], [], []]
    for x in contours:
        for y in x:
            for z in y:
                x_max = max(x_max,z[0])
                x_min = min(x_min,z[0])
                y_max = max(y_max,z[1])
                y_min = min(y_min,z[1])

    # add 20 px up down left right

    inc = 20
    x_max = min(x_max + inc, 1280)
    x_min = max(x_min - inc, 0)

    y_max = min(y_max + inc, 720)
    y_min = max(y_min - inc, 0)

    cv2.drawContours(frame, contours, -1, (0, 0, 255), 3)
    cv2.rectangle(frame,(x_min, y_min),(x_max, y_max),(0,255,0),3)

    # if cnt==0:
    #     cv2.imshow("Background subtracted", subt)
    #     cv2.imwrite("results/Back_subt.png", subt)

    #     cv2.imshow("Edges", edged)
    #     cv2.imwrite("results/Edged.png", edged)

    #     cv2.imshow("Bounded", frame)
    #     cv2.imwrite("results/Bounded.png", frame)


    cv2.imshow(f"frame",frame)
    
    cv2.waitKey(10)

    cnt += 1
    if(cnt==img_frame_end): 
        break
    
cap.release()
cv2.destroyAllWindows()