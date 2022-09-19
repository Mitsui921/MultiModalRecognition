
import mediapipe as mp
import cv2
import numpy as np
import time
import math

# cap = cv2.VideoCapture(0) # 0代表电脑自带摄像头
IMAGE_List = ['./UI_rec/img_test/ges7.jpeg']  # 图片列表
mp_drawing = mp.solutions.drawing_utils  # 点和线的样式
mp_drawing_styles = mp.solutions.drawing_styles  # 点和线的风格
mp_hands = mp.solutions.hands  # 手势识别的API
'''
  def __init__(self,
               static_image_mode=False,
               max_num_hands=2,
               model_complexity=1,  # 将其设置为更高的值可以增加解决方案的健壮性，
                            但代价是延迟更长。如果static_image_mode为true则被忽略
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
'''
gesture = [0, 1, 2, 3, 4, 5]  # 预设数字


def ges(hand_landmarks):
    flag = 0
    p0_x = hand_landmarks.landmark[0].x  # 获取关键点0的x坐标
    p0_y = hand_landmarks.landmark[0].y  # 获取关键点0的x坐标
    p5_x = hand_landmarks.landmark[5].x  # 获取食指底部关键点5的x坐标
    p5_y = hand_landmarks.landmark[5].y  # 获取食指底部关键点5的y坐标
    distance_0_5 = math.sqrt(pow(p0_x - p5_x, 2) ** 2 + pow(p0_y - p5_y, 2) ** 2)  # 计算两个观点的距离
    base = distance_0_5 / 0.6  # 人工经验将距离缩小0.6倍作为基础值

    p4_x = hand_landmarks.landmark[4].x  # 获取大拇指顶部关键点4的x坐标
    p4_y = hand_landmarks.landmark[4].y  # 获取大拇指顶部关键点4的y坐标
    distance_5_4 = math.sqrt(pow(p5_x - p4_x, 2) ** 2 + pow(p5_y - p4_y, 2) ** 2)  # 计算关键点4到关键点5的距离，判断大拇指是否处于张开状态

    p8_x = hand_landmarks.landmark[8].x  # 获取食指顶部关键字8的x坐标
    p8_y = hand_landmarks.landmark[8].y  # 获取食指顶部关键字8的y坐标
    distance_0_8 = math.sqrt(pow(p0_x - p8_x, 2) ** 2 + pow(p0_y - p8_y, 2) ** 2)  # 计算关键点0到关键点8的距离，判断食指是否处于张开状态

    p12_x = hand_landmarks.landmark[12].x  # 获取中指顶部关键点12的x坐标
    p12_y = hand_landmarks.landmark[12].y  # 获取中指顶部关键点12的y坐标
    distance_0_12 = math.sqrt(pow(p0_x - p12_x, 2) ** 2 + pow(p0_y - p12_y, 2) ** 2)  # 计算关键点0到关键点12的距离，判断中指是否处于张开状态

    p16_x = hand_landmarks.landmark[16].x  # 获取无名指关键点16的x坐标
    p16_y = hand_landmarks.landmark[16].y  # 获取无名指关键点16的y坐标
    distance_0_16 = math.sqrt(pow(p0_x - p16_x, 2) ** 2 + pow(p0_y - p16_y, 2) ** 2)  # 计算关键点0到关键点16的距离，判断无名指是否处于张开状态

    p20_x = hand_landmarks.landmark[20].x  # 获取小拇指关键点20的x坐标
    p20_y = hand_landmarks.landmark[20].y  # 获取小拇指关键点20的y坐标
    distance_0_20 = math.sqrt(pow(p0_x - p20_x, 2) ** 2 + pow(p0_y - p20_y, 2) ** 2)  # 计算关键点0到关键点20的距离，判断小拇指是否处于张开状态
    '''
    判断有几根手指处于张开状态
    '''
    if distance_0_8 > base:
        flag += 1
    if distance_0_12 > base:
        flag += 1
    if distance_0_16 > base:
        flag += 1
    if distance_0_20 > base:
        flag += 1
    if distance_5_4 > base * 0.2:  # 调低大拇指伸直阈值
        flag += 1

    return gesture[flag]  # 通过索引值返回数字


with mp_hands.Hands(
        static_image_mode=True,  # False表示为图像检测模式
        max_num_hands=2,  # 最大可检测到两只手掌
        model_complexity=0,  # 可设为0或者1，主要跟模型复杂度有关
        min_detection_confidence=0.5,  # 最大检测阈值
) as hands:
    for idx, file in enumerate(IMAGE_List):
        image = cv2.flip(cv2.imread(file), 1)  # 读取图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将其标为RGB格式
        t0 = time.time()
        results = hands.process(image)  # 使用API处理图像图像
        '''
        results.multi_handedness
        包括label和score,label是字符串"Left"或"Right",score是置信度
        results.multi_hand_landmarks
        results.multi_hand_landmrks:被检测/跟踪的手的集合
        其中每只手被表示为21个手部地标的列表,每个地标由x、y和z组成。
        x和y分别由图像的宽度和高度归一化为[0.0,1.0]。Z表示地标深度
        以手腕深度为原点，值越小，地标离相机越近。 
        z的大小与x的大小大致相同。
        '''
        t1 = time.time()
        fps = 1 / (t1 - t0)  # 实时帧率
        # print('++++++++++++++fps',fps)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 将图像变回BGR形式
        dict_handnumber = {}  # 创建一个字典。保存左右手的手势情况
        if results.multi_handedness:  # 判断是否检测到手掌
            if len(results.multi_handedness) == 2:  # 如果检测到两只手
                for i in range(len(results.multi_handedness)):
                    label = results.multi_handedness[i].classification[0].label  # 获得Label判断是哪几手
                    index = results.multi_handedness[i].classification[0].index  # 获取左右手的索引号
                    hand_landmarks = results.multi_hand_landmarks[index]  # 根据相应的索引号获取xyz值
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,  # 用于指定地标如何在图中连接。
                        mp_drawing_styles.get_default_hand_landmarks_style(),  # 如果设置为None.则不会在图上标出关键点
                        mp_drawing_styles.get_default_hand_connections_style())  # 关键点的连接风格
                    gesresult = ges(hand_landmarks)  # 传入21个关键点集合，返回数字
                    dict_handnumber[label] = gesresult  # 与对应的手进行保存为字典
            else:  # 如果仅检测到一只手
                label = results.multi_handedness[0].classification[0].label  # 获得Label判断是哪几手
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,  # 用于指定地标如何在图中连接。
                    mp_drawing_styles.get_default_hand_landmarks_style(),  # 如果设置为None.则不会在图上标出关键点
                    mp_drawing_styles.get_default_hand_connections_style())  # 关键点的连接风格
                gesresult = ges(hand_landmarks)  # 传入21个关键点集合，返回数字
                dict_handnumber[label] = gesresult  # 与对应的手进行保存为字典
        if len(dict_handnumber) == 2:  # 如果有两只手，则进入
            # print(dict_handnumber)
            leftnumber = dict_handnumber['Right']
            rightnumber = dict_handnumber['Left']
            '''
            显示实时帧率，右手值，左手值，相加值
            '''
            s = 'FPS:{0}\nRighthand Value:{1}\nLefthand Value:{2}\nAdd is:{3}'.format(int(fps), rightnumber, leftnumber,
                                                                                      str(leftnumber + rightnumber))  # 图像上的文字内容
        elif len(dict_handnumber) == 1:  # 如果仅有一只手则进入
            labelvalue = list(dict_handnumber.keys())[0]  # 判断检测到的是哪只手
            if labelvalue == 'Right':  # 左手,不知为何，模型总是将左右手搞反，则引入人工代码纠正
                number = list(dict_handnumber.values())[0]
                s = 'FPS:{0}\nRighthand Value:{1}\nLefthand Value:0\nAdd is:{2}'.format(int(fps), number, number)
            else:  # 右手
                number = list(dict_handnumber.values())[0]
                s = 'FPS:{0}\nLefthand Value:{1}\nRighthand Value:0\nAdd is:{2}'.format(int(fps), number, number)
        else:  # 如果没有检测到则只显示帧率
            s = 'FPS:{0}\n'.format(int(fps))

        y0, dy = 50, 25  # 文字放置初始坐标
        # image = cv2.flip(image,1) # 反转图像
        for i, txt in enumerate(s.split('\n')):  # 根据\n来竖向排列文字
            y = y0 + i * dy
            cv2.putText(image, txt, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow('MediaPipe Gesture Recognition', image)  # 显示图像
        cv2.imwrite('save/{0}.jpg'.format(idx), image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
