import cv2
import numpy as np
from dynamikontrol import Module

CONFIDENCE = 0.9   # 조건 확률
THRESHOLD = 0.3     # NMS(non max suppression
LABELS = ['Car', 'Plate']
CAR_WIDTH_TRESHOLD = 250  # 자동차의 크기가 ?px 이상일 때 > 모터 제어 (영상 크기에 따라 다름)

cap1 = cv2.VideoCapture(0)  # 첫 번째 웹캠
cap2 = cv2.VideoCapture(1)  # 두 번째 웹캠

net1 = cv2.dnn.readNetFromDarknet('yolov4-ANPR.cfg', 'yolov4-ANPR.weights') # darknet의 model을 읽어옴 (모델의 설정값, 모델의 pretrained weight)
net2 = cv2.dnn.readNetFromDarknet('yolov4-ANPR.cfg', 'yolov4-ANPR.weights')

module = Module('AD000044') # 다이나믹콘트롤 모델의 시리얼 넘버 삽입 > 이니셜라이즈

# while cap1.isOpened():       # 웹캠 열기
while True :
    
    ret1, img1 = cap1.read
    ()       # 웹캠 이미지 읽기
    ret2, img2 = cap2.read()
    if not ret1 and not ret2 :
    # if not ret1 :
        break

 
    H, W, _ = img1.shape  # 세로(height) 가로(wide) 모델 사이즈 저장
    H, W, _ = img2.shape

    blob1 = cv2.dnn.blobFromImage(img1, scalefactor=1/255., size=(416, 416), swapRB=True)     # 전처리
    blob2 = cv2.dnn.blobFromImage(img2, scalefactor=1/255., size=(416, 416), swapRB=True)
    net1.setInput(blob1)
    net2.setInput(blob2)
    
    output1 = net1.forward()
    output2 = net2.forward()

    boxes1, confidences1, class_ids1 = [], [], [] # 네모칸 , confidence, 클래스 id 저장
    boxes2, confidences2, class_ids2 = [], [], [] # 네모칸 , confidence, 클래스 id 저장

    for det1 in output1:   # 여러개의 리스트 결과를 det이라는 변수로 
        box1 = det1[:4]       #앞의 4개의 요소는 box
        scores1 = det1[5:]    #5번째부터는 scores를 나타냄
        class_id1 = np.argmax(scores1)    #scores의 argmax
        confidence1 = scores1[class_id1]   #scores의 id를 가져와서 conf구하기

        if confidence1 > CONFIDENCE:
            cx1, cy1, w1, h1 = box1 * np.array([W, H, W, H])     # 구하는 수식
            x1 = cx1 - (w1 / 2)
            y1 = cy1 - (h1 / 2)

            boxes1.append([int(x1), int(y1), int(w1), int(h1)])
            confidences1.append(float(confidence1))
            class_ids1.append(class_id1)

    for det2 in output2:   
        box2 = det2[:4]       
        scores2 = det2[5:]    
        class_id2 = np.argmax(scores2)    
        confidence2 = scores2[class_id2]   

        if confidence2 > CONFIDENCE:
            cx2, cy2, w2, h2 = box2 * np.array([W, H, W, H])     
            x2 = cx2 - (w2 / 2)
            y2 = cy2 - (h2 / 2)

            boxes2.append([int(x2), int(y2), int(w2), int(h2)])
            confidences2.append(float(confidence2))
            class_ids2.append(class_id2)

    idxs1 = cv2.dnn.NMSBoxes(boxes1, confidences1, CONFIDENCE, THRESHOLD)      #YOLO의 Non Max Suppression
    idxs2 = cv2.dnn.NMSBoxes(boxes2, confidences2, CONFIDENCE, THRESHOLD)

    if len(idxs1) > 0:     # 받은 인덱스의 NMS의 값이 있게 되면 flatten해줌
        for i in idxs1.flatten():
            x1, y1, w1, h1 = boxes1[i]

            cv2.rectangle(img1, pt1=(x1, y1), pt2=(x1 + w1, y1 + h1), color=(0, 0, 255), thickness=2)
            cv2.putText(img1, text='%s %.2f %d' % (LABELS[class_ids1[i]], confidences1[i], w1), org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
   
            if class_ids1[i] == 0:         # CAR이면서 
                if w1 > CAR_WIDTH_TRESHOLD:    # 너비가 아까 px보다 크게 되면
                    module.motor.angle(-90)    # 모터의 각도가 ?도 열기
                else:
                    module.motor.angle(0)     # 아니면 모터의 각도를 0으로 해서 닫기
    # else:
    #     module.motor.angle(0) 

    if len(idxs2) > 0:     
        for i in idxs2.flatten():
            x2, y2, w2, h2 = boxes2[i]

            cv2.rectangle(img2, pt1=(x2, y2), pt2=(x2 + w2, y2 + h2), color=(0, 0, 255), thickness=2)
            cv2.putText(img2, text='%s %.2f %d' % (LABELS[class_ids2[i]], confidences2[i], w2), org=(x2, y2 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)

            if class_ids2[i] == 0:         
                if w2 > CAR_WIDTH_TRESHOLD:    
                    module.motor.angle(-90)    
                else:
                    module.motor.angle(0)     
    else:
        module.motor.angle(0)     # 인식을 못했을 경우 > 닫기

    # cv2.imshow('result1', img1)       # 이미지를 result에 표시
    # cv2.imshow('result2', img2) 
    both_frames = cv2.hconcat([img1, img2])     # 웹캠 두개 연결
    cv2.imshow('Both Webcams', both_frames)

    if cv2.waitKey(1) == ord('q'):  # 1ms 동안 키 입력을 대기, 문자 'q'의 ASCII 코드 값을 반환 >>만약 'q'가 눌리면 while 루프를 빠져나오도록
        break

    

cap1.release()
cap2.release()
cv2.destroyAllWindows()

