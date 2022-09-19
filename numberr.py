import mediapipe as mp
import cv2
import numpy as np


def get_angle(v1, v2):
    angle = np.dot(v1, v2) / (np.sqrt(np.sum(v1 * v1)) * np.sqrt(np.sum(v2 * v2)))
    angle = np.arccos(angle) / 3.14 * 180

    return angle


def get_str_guester(up_fingers, list_lms):
    if len(up_fingers) == 1 and up_fingers[0] == 8:

        v1 = list_lms[6] - list_lms[7]
        v2 = list_lms[8] - list_lms[7]

        angle = get_angle(v1, v2)

        if angle < 160:
            str_guester = "9"
        else:
            str_guester = "1"

    elif len(up_fingers) == 1 and up_fingers[0] == 4:
        str_guester = "Good"

    elif len(up_fingers) == 1 and up_fingers[0] == 20:
        str_guester = "Bad"

    elif len(up_fingers) == 1 and up_fingers[0] == 12:
        str_guester = "FXXX"

    elif len(up_fingers) == 2 and up_fingers[0] == 8 and up_fingers[1] == 12:
        str_guester = "2"

    elif len(up_fingers) == 2 and up_fingers[0] == 4 and up_fingers[1] == 20:
        str_guester = "6"

    elif len(up_fingers) == 2 and up_fingers[0] == 4 and up_fingers[1] == 8:
        str_guester = "8"

    elif len(up_fingers) == 3 and up_fingers[0] == 8 and up_fingers[1] == 12 and up_fingers[2] == 16:
        str_guester = "3"

    elif len(up_fingers) == 3 and up_fingers[0] == 4 and up_fingers[1] == 8 and up_fingers[2] == 12:

        dis_8_12 = list_lms[8, :] - list_lms[12, :]
        dis_8_12 = np.sqrt(np.dot(dis_8_12, dis_8_12))

        dis_4_12 = list_lms[4, :] - list_lms[12, :]
        dis_4_12 = np.sqrt(np.dot(dis_4_12, dis_4_12))

        if dis_4_12 / (dis_8_12 + 1) < 3:
            str_guester = "7"

        elif dis_4_12 / (dis_8_12 + 1) > 5:
            str_guester = "Gun"
        else:
            str_guester = "7"

    elif len(up_fingers) == 3 and up_fingers[0] == 4 and up_fingers[1] == 8 and up_fingers[2] == 20:
        str_guester = "ROCK"

    elif len(up_fingers) == 4 and up_fingers[0] == 8 and up_fingers[1] == 12 and up_fingers[2] == 16 and up_fingers[
        3] == 20:
        str_guester = "4"

    elif len(up_fingers) == 5:
        str_guester = "5"

    elif len(up_fingers) == 0:
        str_guester = "10"

    else:
        str_guester = "None"

    return str_guester


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    # 定义手 检测对象
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    while True:

        # 读取一帧图像
        success, img = cap.read()
        if not success:
            continue
        image_height, image_width, _ = np.shape(img)

        # 转换为RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 得到检测结果
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]

            mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)

            # 采集所有关键点的坐标
            list_lms = []
            for i in range(21):
                pos_x = hand.landmark[i].x * image_width
                pos_y = hand.landmark[i].y * image_height
                list_lms.append([int(pos_x), int(pos_y)])

            # 构造凸包点
            list_lms = np.array(list_lms, dtype=np.int32)
            hull_index = [0, 1, 2, 3, 6, 10, 14, 19, 18, 17, 10]
            hull = cv2.convexHull(list_lms[hull_index, :])
            # 绘制凸包
            cv2.polylines(img, [hull], True, (0, 255, 0), 2)

            # 查找外部的点数
            n_fig = -1
            ll = [4, 8, 12, 16, 20]
            up_fingers = []

            for i in ll:
                pt = (int(list_lms[i][0]), int(list_lms[i][1]))
                dist = cv2.pointPolygonTest(hull, pt, True)
                if dist < 0:
                    up_fingers.append(i)

            # print(up_fingers)
            # print(list_lms)
            # print(np.shape(list_lms))
            str_guester = get_str_guester(up_fingers, list_lms)

            cv2.putText(img, ' %s' % (str_guester), (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                        cv2.LINE_AA)

            for i in ll:
                pos_x = hand.landmark[i].x * image_width
                pos_y = hand.landmark[i].y * image_height
                # 画点
                cv2.circle(img, (int(pos_x), int(pos_y)), 3, (0, 255, 255), -1)

        cv2.imshow("hands", img)

        key = cv2.waitKey(1) & 0xFF

        # 按键 "q" 退出
        if key == ord('q'):
            break
    cap.release()