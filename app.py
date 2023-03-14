
import imutils
import numpy as np
import cv2
  
  

pred_good=0
pred_bad=0
webcam = cv2.VideoCapture(0)
classes = ['GOOD  ','BAD  ']


while(1):
      
    persentase = 0
    _, imageFrame = webcam.read()
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
    canvas = np.zeros((250, 300, 3), dtype="uint8")

    offset_bawah = np.array([44, 2, 101], np.uint8)
    offset_atas = np.array([163, 106, 236], np.uint8)
    terong_good = cv2.inRange(hsvFrame, offset_bawah, offset_atas)
      

    kernal = np.ones((5, 5), "uint8")
      
  
      
    # deteksi terong bagus dari sampel warna
    terong_good = cv2.dilate(terong_good, kernal)
    res_blue = cv2.bitwise_and(imageFrame, imageFrame,
                               mask = terong_good)
   
    
    # track terong
    contours, hierarchy = cv2.findContours(terong_good,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                       (x + w, y + h),
                                       (255, 0, 0), 2)
            persentase = persentase+1
            cv2.putText(imageFrame, "Terong", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (255, 0, 0))
           
            if persentase > 1:
                pred_good = persentase*13
                pred_bad = 100-pred_good
            else:
                pred_good = 0
                pred_bad = 0

            
           
       
             
            
            

            
        
    cv2.rectangle(canvas, (0, (0 * 35) + 5), (pred_good*3, (0 * 35) + 35), (0, 50, 100), -1)
    cv2.rectangle(canvas, (1, (1 * 35) + 5), (pred_bad*3, (1 * 35) + 35), (0, 50, 100), -1)
    
    cv2.putText(canvas, classes[0]+ str(pred_good)+" %", (10, (0 * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(canvas, classes[1] + str(pred_bad)+" %", (10, (1 * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)   
# Program Termination
    cv2.imshow("Terong CLasification", imageFrame)
    cv2.imshow("Probabilities", canvas)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break