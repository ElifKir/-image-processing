import cv2
import time

cap = cv2.VideoCapture("MOT17-09-DPM-raw.mp4")


object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)   

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape    

    
    roi = frame[0: 720,0:1600]     


    
    mask = object_detector.apply(roi)   
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   
    
    for cnt in contours:
        
        area = cv2.contourArea(cnt)     
        if area > 2000:                  

            

            x, y, w, h = cv2.boundingRect(cnt)     
            cv2.rectangle(roi, (x, y), (x + w, y + h), (10, 250, 0), 3)    

    time.sleep(0.001)
    
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)     

    key = cv2.waitKey(30)
    if key == 27:
        break
    if cv2.waitKey(1) & 0xff == ord("q"):
    
        break
    
    

cap.release()
cv2.destroyAllWindows()
