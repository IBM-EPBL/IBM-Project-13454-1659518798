
import os
import cv2

def create_folder(folder_name):
    if not os.path.exists('D/' + folder_name):
        os.mkdir('Dataset/Image/' + folder_name)
name=input("enter gesture name:")
create_folder(name)        

cap=cv2.VideoCapture(0)
directory='Dataset/Image/'
while True:
    _,frame=cap.read()
    count = {'c': len(os.listdir(directory+"/"+name))}
    cv2.putText(frame, "for capture press c : "+str(count['c']), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)

    row = frame.shape[1]
    col = frame.shape[0]
    cv2.rectangle(frame,(0,40),(300,400),(255,255,255),2)
    cv2.imshow("data",frame)#MithoonNS
    cv2.imshow("ROI",frame[40:400,0:300])
    frame=frame[40:400,0:300]
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == ord('c'):

        cv2.imwrite(directory+name+'/'+str(count['c'])+'.png',frame)


cap.release()
cv2.destroyAllWindows()