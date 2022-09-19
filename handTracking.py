import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHand = mp.solutions.hands
hands = mpHand.Hands()
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=5)  # 点样式
handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=10)  # 连接线样式
pTime = 0
cTime = 0

while True:
    ret, img = cap.read()
    if ret:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)
        # print(result.multi_hand_landmarks)
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]

        if result.multi_hand_landmarks:  # 多个点出现时表现出来
            for handLms in result.multi_hand_landmarks:  # 多个手出现时表示出来
                mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS, handLmsStyle, handConStyle)
                for i, lm in enumerate(handLms.landmark):  # enumerate-用于确定是第几个点 i-表示第几个点 lm-点坐标
                    xPos = int(lm.x * imgWidth)
                    yPos = int(lm.y * imgHeight)  # 若不乘以窗口的长宽，输出的为不大于1的小数，是其在画面所占的百分比
                    # cv2.putText(img, str(i),(xPos-25,yPos+5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),2)
                    # 各个部分所对应的数字 像素往左25 向上5--字体--大小--颜色--粗细
                    if i == 4:
                        cv2.circle(img, (xPos, yPos), 10, (0, 0, 255), cv2.FILLED)
                    print(i, xPos, yPos)

        cTime = time.time()  # 每秒的帧频显示
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS :{int(fps)}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow('img', img)

    if cv2.waitKey(1) == ord('q'):
        break
