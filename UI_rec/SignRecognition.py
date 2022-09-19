
import time
from os import getcwd
import numpy as np
import cv2
import requests
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp

import image1_rc  # 载入图标文件
from SignRecognition_UI import Ui_MainWindow
from numberr import get_str_guester
import dlib
import os
import shutil
import logging
import csv
import pandas as pd
import Constant
from requests.auth import HTTPDigestAuth
import json


# Dlib 正向人脸检测器 / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# 要读取人脸图像文件的路径 / Path of cropped faces
path_images_from_camera = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/data_faces_from_camera/"

# Dlib 人脸 landmark 特征点检测器 / Get face landmarks
predictor = dlib.shape_predictor(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/data_dlib/shape_predictor_68_face_landmarks.dat")

# Dlib Resnet 人脸识别模型，提取 128D 的特征矢量 / Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# 人脸特征文件
face_features = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/features_all.csv"

class Face_Register:
    def __init__(self):
        self.path_photos_from_camera = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/data_faces_from_camera/"
        self.font = cv2.FONT_ITALIC

        self.existing_faces_cnt = 0         # 已录入的人脸计数器 / cnt for counting saved faces
        self.ss_cnt = 0                     # 录入 personX 人脸时图片计数器 / cnt for screen shots
        self.current_frame_faces_cnt = 0    # 录入人脸计数器 / cnt for counting faces in current frame

        self.save_flag = 1                  # 之后用来控制是否保存图像的 flag / The flag to control if save
        self.press_n_flag = 0               # 之后用来检查是否先按 'n' 再按 's' / The flag to check if press 'n' before 's'

        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

    # 新建保存人脸图像文件和数据 CSV 文件夹 / Mkdir for saving photos and csv
    def pre_work_mkdir(self):
        # 新建文件夹 / Create folders to save face images and csv
        if os.path.isdir(self.path_photos_from_camera):
            pass
        else:
            os.mkdir(self.path_photos_from_camera)

    # 删除之前存的人脸数据文件夹 / Delete old face folders
    def pre_work_del_old_face_folders(self):
        # 删除之前存的人脸数据文件夹, 删除 "/data_faces_from_camera/person_x/"...
        folders_rd = os.listdir(self.path_photos_from_camera)
        for i in range(len(folders_rd)):
            shutil.rmtree(self.path_photos_from_camera+folders_rd[i])
        if os.path.isfile(face_features):
            os.remove(face_features)

    # 如果有之前录入的人脸, 在之前 person_x 的序号按照 person_x+1 开始录入 / Start from person_x+1
    def check_existing_faces_cnt(self):
        if os.listdir(self.path_photos_from_camera):
            # 获取已录入的最后一个人脸序号 / Get the order of latest person
            person_list = os.listdir(self.path_photos_from_camera)
            person_num_list = []
            for person in person_list:
                person_num_list.append(int(person.split('_')[-1]))
            self.existing_faces_cnt = max(person_num_list)

        # 如果第一次存储或者没有之前录入的人脸, 按照 person_1 开始录入 / Start from person_1
        else:
            self.existing_faces_cnt = 0

    # 更新 FPS / Update FPS of Video stream
    def update_fps(self):
        now = time.time()
        # 每秒刷新 fps / Refresh fps per second
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    # 生成的 cv2 window 上面添加说明文字 / PutText on cv2 window
    def draw_note(self, img_rd):
        # 添加说明 / Add some notes
        cv2.putText(img_rd, "Face Register", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:   " + str(self.fps_show.__round__(2)), (20, 100), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Faces: " + str(self.current_frame_faces_cnt), (20, 140), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "N: Create face folder", (20, 350), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "S: Save current face", (20, 400), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    # 获取人脸 / Main process of face detection and saving
    def process(self, stream):
        # 1. 新建储存人脸图像文件目录 / Create folders to save photos
        self.pre_work_mkdir()

        # 2. 删除 "/data/data_faces_from_camera" 中已有人脸图像文件
        # / Uncomment if want to delete the saved faces and start from person_1
        # if os.path.isdir(self.path_photos_from_camera):
        #     self.pre_work_del_old_face_folders()

        # 3. 检查 "/data/data_faces_from_camera" 中已有人脸文件
        self.check_existing_faces_cnt()

        while stream.isOpened():
            flag, img_rd = stream.read()        # Get camera video stream
            kk = cv2.waitKey(1)
            faces = detector(img_rd, 0)         # Use Dlib face detector

            # 4. 按下 'n' 新建存储人脸的文件夹 / Press 'n' to create the folders for saving faces
            if kk == ord('n'):
                self.existing_faces_cnt += 1
                current_face_dir = self.path_photos_from_camera + "person_" + str(self.existing_faces_cnt)
                os.makedirs(current_face_dir)
                cv2.putText(img_rd, "Folder created successfully", (20, 300), self.font, 0.8, (0, 0, 255), 1,
                            cv2.LINE_AA)
                logging.info("\n%-40s %s", "新建的人脸文件夹 / Create folders:", current_face_dir)

                self.ss_cnt = 0                 # 将人脸计数器清零 / Clear the cnt of screen shots
                self.press_n_flag = 1           # 已经按下 'n' / Pressed 'n' already

            # 5. 检测到人脸 / Face detected
            if len(faces) != 0:
                # 矩形框 / Show the ROI of faces
                for k, d in enumerate(faces):
                    # 计算矩形框大小 / Compute the size of rectangle box
                    height = (d.bottom() - d.top())
                    width = (d.right() - d.left())
                    hh = int(height/2)
                    ww = int(width/2)

                    # 6. 判断人脸矩形框是否超出 480x640 / If the size of ROI > 480x640
                    if (d.right()+ww) > 640 or (d.bottom()+hh > 480) or (d.left()-ww < 0) or (d.top()-hh < 0):
                        cv2.putText(img_rd, "OUT OF RANGE", (20, 300), self.font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                        color_rectangle = (0, 0, 255)
                        save_flag = 0
                        if kk == ord('s'):
                            logging.warning("请调整位置 / Please adjust your position")
                    else:
                        color_rectangle = (255, 255, 255)
                        save_flag = 1

                    cv2.rectangle(img_rd,
                                  tuple([d.left() - ww, d.top() - hh]),
                                  tuple([d.right() + ww, d.bottom() + hh]),
                                  color_rectangle, 2)

                    # 7. 根据人脸大小生成空的图像 / Create blank image according to the size of face detected
                    img_blank = np.zeros((int(height*2), width*2, 3), np.uint8)

                    if save_flag:
                        # 8. 按下 's' 保存摄像头中的人脸到本地 / Press 's' to save faces into local images
                        if kk == ord('s'):
                            # 检查有没有先按'n'新建文件夹 / Check if you have pressed 'n'
                            if self.press_n_flag:
                                self.ss_cnt += 1
                                for ii in range(height*2):
                                    for jj in range(width*2):
                                        img_blank[ii][jj] = img_rd[d.top()-hh + ii][d.left()-ww + jj]
                                cv2.imwrite(current_face_dir + "/img_face_" + str(self.ss_cnt) + ".jpg", img_blank)
                                cv2.putText(img_rd, "Face img saved successfully", (20, 300), self.font, 0.8, (0, 0, 255), 1,
                                            cv2.LINE_AA)
                                logging.info("%-40s %s/img_face_%s.jpg", "写入本地 / Save into：",
                                             str(current_face_dir), str(self.ss_cnt))
                            else:
                                logging.warning("请先按 'N' 建文件夹, 按 'S' 保存当前人脸/ Please press 'N' and press 'S'")

            self.current_frame_faces_cnt = len(faces)

            # 9. 生成的窗口添加说明文字 / Add note on cv2 window
            self.draw_note(img_rd)

            # 10. 按下 'q' 键退出 / Press 'q' to exit
            if kk == ord('q'):
                break

            # 11. Update FPS
            self.update_fps()

            cv2.namedWindow("camera", 1)
            cv2.imshow("camera", img_rd)

    # 返回单张图像的 128D 特征 / Return 128D features for single image
    # Input:    path_img           <class 'str'>
    # Output:   face_descriptor    <class 'dlib.vector'>
    def return_128d_features(self, path_img):
        img_rd = cv2.imread(path_img)
        faces = detector(img_rd, 1)

        logging.info("%-40s %-20s", "检测到人脸的图像 / Image with faces detected:", path_img)

        # 因为有可能截下来的人脸再去检测，检测不出来人脸了, 所以要确保是 检测到人脸的人脸图像拿去算特征
        # For photos of faces saved, we need to make sure that we can detect faces from the cropped images
        if len(faces) != 0:
            shape = predictor(img_rd, faces[0])
            face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
        else:
            face_descriptor = 0
            logging.warning("no face")
        return face_descriptor

    # 返回 personX 的 128D 特征均值 / Return the mean value of 128D face descriptor for person X
    # Input:    path_face_personX        <class 'str'>
    # Output:   features_mean_personX    <class 'numpy.ndarray'>
    def return_features_mean_personX(self, path_face_personX):
        features_list_personX = []
        photos_list = os.listdir(path_face_personX)
        if photos_list:
            for i in range(len(photos_list)):
                # 调用 return_128d_features() 得到 128D 特征 / Get 128D features for single image of personX
                logging.info("%-40s %-20s", "正在读的人脸图像 / Reading image:", path_face_personX + "/" + photos_list[i])
                features_128d = self.return_128d_features(path_face_personX + "/" + photos_list[i])
                # 遇到没有检测出人脸的图片跳过 / Jump if no face detected from image
                if features_128d == 0:
                    i += 1
                else:
                    features_list_personX.append(features_128d)
        else:
            logging.warning("文件夹内图像文件为空 / Warning: No images in%s/", path_face_personX)

        # 计算 128D 特征的均值 / Compute the mean
        # personX 的 N 张图像 x 128D -> 1 x 128D
        if features_list_personX:
            features_mean_personX = np.array(features_list_personX, dtype=object).mean(axis=0)
        else:
            features_mean_personX = np.zeros(128, dtype=object, order='C')
        return features_mean_personX

    # 获取录入人脸的特征
    def extract(self):
        logging.basicConfig(level=logging.INFO)
        # 获取已录入的最后一个人脸序号 / Get the order of latest person
        person_list = os.listdir(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/data_faces_from_camera/")
        person_list.sort()

        with open(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/features_all.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for person in person_list:
                # Get the mean/average features of face/personX, it will be a list with a length of 128D
                logging.info("%sperson_%s", path_images_from_camera, person)
                path_face_personX = path_images_from_camera + person
                features_mean_personX = self.return_features_mean_personX(path_face_personX)

                if len(person.split('_', 2)) == 2:
                    # "person_x"
                    person_name = person
                else:
                    # "person_x_tom"
                    person_name = person.split('_', 2)[-1]
                features_mean_personX = np.insert(features_mean_personX, 0, person_name, axis=0)
                # features_mean_personX will be 129D, person name + 128 features
                writer.writerow(features_mean_personX)
                logging.info('\n')
            logging.info("所有录入人脸数据存入 / Save all the features of faces registered into: data/features_all.csv")

    def run(self):
        cap = cv2.VideoCapture(0)               # Get video stream from camera
        self.process(cap)

        cap.release()
        cv2.destroyAllWindows()

        self.extract()

class Face_Recognizer:
    def __init__(self):
        self.face_feature_known_list = []                # 用来存放所有录入人脸特征的数组 / Save the features of faces in database
        self.face_name_known_list = []                   # 存储录入人脸名字 / Save the name of faces in database

        self.current_frame_face_cnt = 0                     # 存储当前摄像头中捕获到的人脸数 / Counter for faces in current frame
        self.current_frame_face_feature_list = []           # 存储当前摄像头中捕获到的人脸特征 / Features of faces in current frame
        self.current_frame_face_name_list = []              # 存储当前摄像头中捕获到的所有人脸的名字 / Names of faces in current frame
        self.current_frame_face_name_position_list = []     # 存储当前摄像头中捕获到的所有人脸的名字坐标 / Positions of faces in current frame

        # Update FPS
        self.fps = 0                    # FPS of current frame
        self.fps_show = 0               # FPS per second
        self.frame_start_time = 0
        self.frame_cnt = 0
        self.start_time = time.time()

        self.font = cv2.FONT_ITALIC
        self.font_chinese = ImageFont.truetype("simsun.ttc", 30)

    # 从 "features_all.csv" 读取录入人脸特征 / Read known faces from "features_all.csv"
    def get_face_database(self):
        if os.path.exists(face_features):
            path_features_known_csv = face_features
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                for j in range(1, 129):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.face_feature_known_list.append(features_someone_arr)
            logging.info("Faces in Database：%d", len(self.face_feature_known_list))
            return 1
        else:
            logging.warning("'features_all.csv' not found!")
            return 0

    # 计算两个128D向量间的欧式距离 / Compute the e-distance between two 128D features
    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    # 更新 FPS / Update FPS of Video stream
    def update_fps(self):
        now = time.time()
        # 每秒刷新 fps / Refresh fps per second
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    # 生成的 cv2 window 上面添加说明文字 / PutText on cv2 window
    def draw_note(self, img_rd):
        cv2.putText(img_rd, "Face Recognizer", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Frame:  " + str(self.frame_cnt), (20, 100), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:    " + str(self.fps_show.__round__(2)), (20, 130), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Faces:  " + str(self.current_frame_face_cnt), (20, 160), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_name(self, img_rd):
        # 在人脸框下面写人脸名字 / Write names under rectangle
        img = Image.fromarray(cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        for i in range(self.current_frame_face_cnt):
            # cv2.putText(img_rd, self.current_frame_face_name_list[i], self.current_frame_face_name_position_list[i], self.font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
            draw.text(xy=self.current_frame_face_name_position_list[i], text=self.current_frame_face_name_list[i], font=self.font_chinese,
                  fill=(255, 255, 0))
            # print(self.current_frame_face_name_list[i])
            img_rd = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img_rd

    # 修改显示人名 / Show names in chinese
    def show_chinese_name(self):
        # Default known name: person_1, person_2, person_3
        if self.current_frame_face_cnt >= 1:
            # 修改录入的人脸姓名 / Modify names in face_name_known_list to chinese name
            self.face_name_known_list[0] = '张三'.encode('utf-8').decode()
            # self.face_name_known_list[1] = '张四'.encode('utf-8').decode()

    # 处理获取的视频流，进行人脸识别 / Face detection and recognition from input video stream
    def process(self, stream):
        # 1. 读取存放所有人脸特征的 csv / Read known faces from "features.all.csv"
        if self.get_face_database():
            while stream.isOpened():
                self.frame_cnt += 1
                logging.debug("Frame %d starts", self.frame_cnt)
                flag, img_rd = stream.read()
                faces = detector(img_rd, 0)
                kk = cv2.waitKey(1)
                # 按下 q 键退出 / Press 'q' to quit
                if kk == ord('q'):
                    break
                else:
                    self.draw_note(img_rd)
                    self.current_frame_face_feature_list = []
                    self.current_frame_face_cnt = 0
                    self.current_frame_face_name_position_list = []
                    self.current_frame_face_name_list = []

                    # 2. 检测到人脸 / Face detected in current frame
                    if len(faces) != 0:
                        # 3. 获取当前捕获到的图像的所有人脸的特征 / Compute the face descriptors for faces in current frame
                        for i in range(len(faces)):
                            shape = predictor(img_rd, faces[i])
                            self.current_frame_face_feature_list.append(face_reco_model.compute_face_descriptor(img_rd, shape))
                        # 4. 遍历捕获到的图像中所有的人脸 / Traversal all the faces in the database
                        for k in range(len(faces)):
                            logging.debug("For face %d in camera:", k+1)
                            # 先默认所有人不认识，是 unknown / Set the default names of faces with "unknown"
                            self.current_frame_face_name_list.append("unknown")

                            # 每个捕获人脸的名字坐标 / Positions of faces captured
                            self.current_frame_face_name_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                            # 5. 对于某张人脸，遍历所有存储的人脸特征
                            # For every faces detected, compare the faces in the database
                            current_frame_e_distance_list = []
                            for i in range(len(self.face_feature_known_list)):
                                # 如果 person_X 数据不为空
                                if str(self.face_feature_known_list[i][0]) != '0.0':
                                    e_distance_tmp = self.return_euclidean_distance(self.current_frame_face_feature_list[k],
                                                                                    self.face_feature_known_list[i])
                                    logging.debug("  With person %s, the e-distance is %f", str(i + 1), e_distance_tmp)
                                    current_frame_e_distance_list.append(e_distance_tmp)
                                else:
                                    # 空数据 person_X
                                    current_frame_e_distance_list.append(999999999)
                            # 6. 寻找出最小的欧式距离匹配 / Find the one with minimum e-distance
                            similar_person_num = current_frame_e_distance_list.index(min(current_frame_e_distance_list))
                            logging.debug("Minimum e-distance with %s: %f", self.face_name_known_list[similar_person_num], min(current_frame_e_distance_list))

                            if min(current_frame_e_distance_list) < 0.4:
                                self.current_frame_face_name_list[k] = self.face_name_known_list[similar_person_num]
                                logging.debug("Face recognition result: %s", self.face_name_known_list[similar_person_num])
                            else:
                                logging.debug("Face recognition result: Unknown person")
                            logging.debug("\n")

                            # 矩形框 / Draw rectangle
                            for kk, d in enumerate(faces):
                                # 绘制矩形框
                                cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]),
                                              (255, 255, 255), 2)

                        self.current_frame_face_cnt = len(faces)

                        # 7. 在这里更改显示的人名 / Modify name if needed
                        # self.show_chinese_name()

                        # 8. 写名字 / Draw name
                        img_with_name = self.draw_name(img_rd)
                    else:
                        img_with_name = img_rd

                logging.debug("Faces in camera now: %s", self.current_frame_face_name_list)

                cv2.imshow("camera", img_with_name)

                # 9. 更新 FPS / Update stream FPS
                self.update_fps()
                logging.debug("Frame ends\n\n")

    # OpenCV 调用摄像头并进行 process
    def run(self):
        cap = cv2.VideoCapture(0)              # Get video stream from camera
        cap.set(3, 480)                        # 640x480
        self.process(cap)

        cap.release()
        cv2.destroyAllWindows()

        return self.current_frame_face_name_list


str_guester_list = ["None"]


class Sign_MainWindow(Ui_MainWindow):
    delivery_flag = 1

    def __init__(self, MainWindow):
        self.current_image = None
        self.detInfo = []
        self.path = getcwd()  # 当前路径作为文件选择窗口路径
        self.timer_camera = QtCore.QTimer()  # 相机定时器
        self.timer_video = QtCore.QTimer()  # 视频定时器
        self.video_path = getcwd()  # 视频文件位置

        # 界面控件方法
        self.setupUi(MainWindow)
        self.retranslateUi(MainWindow)

        self.slot_init()  # 槽函数设置

        self.CAM_NUM = 0  # 摄像头标号
        self.cap = cv2.VideoCapture(self.CAM_NUM)  # 屏幕画面对象
        self.cap_video = None  # 视频画面

        # 模型对象
        self.model = None
        self.detector = None
        self.predictor = None

        self.flag_timer = ""  # 标记当前进行的任务（视频or摄像）
        self.fontC = ImageFont.truetype("./Font/楷体_GB2312.ttf", 20, 0)

        # 新建模型对象
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def slot_init(self):  # 定义槽函数
        # self.toolButton_video.clicked.connect(self.button_open_video_click)  # 绑定点击视频槽函数
        self.toolButton_camera.clicked.connect(self.button_open_camera_click)  # 绑定点击摄像头槽函数
        self.timer_camera.timeout.connect(self.show_camera)  # 摄像头定时器槽函数
        self.timer_video.timeout.connect(self.show_video)  # 视频定时器槽函数
        # self.toolButton_pic.clicked.connect(self.choose_file)  # 选择图片
        self.comboBox_select.currentIndexChanged.connect(self.select_obj)  # 下拉框槽函数
        self.comboBox_select.highlighted.connect(self.pause_run)  # 下拉框停留槽函数
        self.toolButton.clicked.connect(self.button_open_register_click)  # 绑定点击录入人脸槽函数

    def pause_run(self):
        if self.comboBox_select.count() > 1:
            if self.flag_timer == "video":
                self.timer_video.stop()
            elif self.flag_timer == "camera":
                self.timer_camera.stop()

    def choose_file(self):
        # 选择图片文件后执行此槽函数
        self.timer_camera.stop()
        self.timer_video.stop()
        if self.cap:
            self.cap.release()  # 释放摄像画面
        if self.cap_video:
            self.cap_video.release()  # 释放视频画面帧
        self.label_display.clear()

        # 重置下拉选框
        self.comboBox_select.currentIndexChanged.disconnect(self.select_obj)
        self.comboBox_select.clear()
        self.comboBox_select.addItem('所有手势')
        self.comboBox_select.currentIndexChanged.connect(self.select_obj)
        # 清除UI上的label显示
        self.label_numer_result.setText("0")
        self.label_time_result.setText('0')
        self.label_class_result.setText('None')
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_class_result.setFont(font)
        self.label_numer_score.setText("0")  # 显示置信度值
        # 清除位置坐标
        self.label_xmin_result.setText("0")
        self.label_ymin_result.setText("0")
        self.label_xmax_result.setText("0")
        self.label_ymax_result.setText("0")
        self.textEdit_camera.setText('实时摄像已关闭')
        self.textEdit_camera.setStyleSheet("background-color: transparent;\n"
                                           "border-color: rgb(0, 170, 255);\n"
                                           "color: rgb(0, 170, 255);\n"
                                           "font: regular 12pt \"华为仿宋\";")
        self.textEdit_video.setText('实时视频已关闭')
        self.textEdit_video.setStyleSheet("background-color: transparent;\n"
                                          "border-color: rgb(0, 170, 255);\n"
                                          "color: rgb(0, 170, 255);\n"
                                          "font: regular 12pt \"华为仿宋\";")
        self.label_display.clear()
        self.label_display.setStyleSheet("border-image: url(:/newPrefix/images_test/ini-image.png);")
        self.flag_timer = ""
        # 使用文件选择对话框选择图片
        fileName_choose, filetype = QFileDialog.getOpenFileName(
            self.centralwidget, "选取图片文件",
            self.path,  # 起始路径
            "图片(*.jpg;*.jpeg;*.png)")  # 文件类型
        self.path = fileName_choose  # 保存路径
        if fileName_choose != '':
            self.flag_timer = "image"
            self.textEdit_pic.setText(fileName_choose + '文件已选中')
            self.textEdit_pic.setStyleSheet("background-color: transparent;\n"
                                            "border-color: rgb(0, 170, 255);\n"
                                            "color: rgb(0, 170, 255);\n"
                                            "font: regular 12pt \"华为仿宋\";")
            self.label_display.setText('正在启动识别系统...\n\nleading')
            QtWidgets.QApplication.processEvents()
            # 生成模型对象

            image = self.cv_imread(fileName_choose)  # 读取选择的图片
            # self.current_image = image.copy()
            frame = image.copy()

            self.current_image = image.copy()

            image_height, image_width, _ = np.shape(image)
            imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB

            self.mpHands = mp.solutions.hands
            self.hands = self.mpHands.Hands()
            self.mpDraw = mp.solutions.drawing_utils

            # 得到检测结果
            time_start = time.time()  # 开始计时
            count = 0
            results = self.hands.process(imgRGB)
            if results.multi_hand_landmarks:
                self.detInfo = []
                text_select = self.comboBox_select.currentText()

                for hand in results.multi_hand_landmarks:  # 多个手出现时表示出来
                    count += 1
                    # hand = results.multi_hand_landmarks[0]

                    # 采集所有关键点的坐标
                    list_lms = []
                    for i in range(21):
                        pos_x = hand.landmark[i].x * image_width
                        pos_y = hand.landmark[i].y * image_height
                        list_lms.append([int(pos_x), int(pos_y)])

                    # 构造凸包点
                    list_lms = np.array(list_lms, dtype=np.int32)

                    # 区域位置
                    xmin = list_lms[:, 0].min() - 20
                    ymin = list_lms[:, 1].min() - 20
                    xmax = list_lms[:, 0].max() + 20
                    ymax = list_lms[:, 1].max() + 20
                    bbox = [xmin, ymin, xmax, ymax]

                    hull_index = [0, 1, 2, 3, 6, 10, 14, 19, 18, 17, 10]
                    hull = cv2.convexHull(list_lms[hull_index, :])

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
                    # self.label_numer_score.setText(str(len(up_fingers)))

                    # print(list_lms)
                    # print(np.shape(list_lms))
                    str_guester = get_str_guester(up_fingers, list_lms)
                    self.detInfo.append([str_guester, bbox])

                    # cv2.putText(image, ' %s' % (str_guester), (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                    #             cv2.LINE_AA)
                    text = "手势{}：{}".format(count + 1, str_guester)

                    if text_select != "所有手势":
                        if text_select != text:
                            continue

                    for i in ll:
                        pos_x = hand.landmark[i].x * image_width
                        pos_y = hand.landmark[i].y * image_height
                        # 画点
                        cv2.circle(image, (int(pos_x), int(pos_y)), 3, (0, 255, 255), -1)

                    cv2.polylines(image, [hull], True, (0, 255, 0), 2)  # 绘制凸包
                    self.mpDraw.draw_landmarks(image, hand, self.mpHands.HAND_CONNECTIONS)

                    # 设置检测到的人脸位置坐标显示
                    if count == 1:
                        self.label_xmin_result.setText(str(xmin))
                        self.label_xmax_result.setText(str(xmax))
                        self.label_ymin_result.setText(str(ymin))
                        self.label_ymax_result.setText(str(ymax))
                        self.label_numer_score.setText(str(len(up_fingers)))  # 伸出的手指数
                        self.label_class_result.setText(str_guester)
                    else:
                        self.label_xmin_result_2.setText(str(xmin))
                        self.label_xmax_result_2.setText(str(xmax))
                        self.label_ymin_result_2.setText(str(ymin))
                        self.label_ymax_result_2.setText(str(ymax))
                        self.label_numer_score_2.setText(str(len(up_fingers)))  # 伸出的手指数
                        self.label_class_result_2.setText(str_guester)

                    # image = self.drawRectBox(image, bbox, "手势：" + str_guester)
                    image = self.drawRectBox(image, bbox, "手势" + str(count) + "：" + str_guester)
                # 更新下拉选框
                QtWidgets.QApplication.processEvents()
                self.comboBox_select.currentIndexChanged.disconnect(self.select_obj)
                self.comboBox_select.clear()
                self.comboBox_select.addItem('所有手势')
                for i in range(len(self.detInfo)):
                    text = "手势{}：{}".format(i + 1, self.detInfo[i][0])
                    self.comboBox_select.addItem(text)
                self.comboBox_select.currentIndexChanged.connect(self.select_obj)
                self.label_numer_result.setText(str(count))  # 更新手势个数
                if count == 1:
                    self.label_xmin_result_2.setText("0")
                    self.label_xmax_result_2.setText("0")
                    self.label_ymin_result_2.setText("0")
                    self.label_ymax_result_2.setText("0")
                    self.label_numer_score_2.setText("0")  # 伸出的手指数
                    self.label_class_result_2.setText("None")
            else:
                self.comboBox_select.currentIndexChanged.disconnect(self.select_obj)
                self.comboBox_select.clear()
                self.comboBox_select.addItem('所有手势')
                # 清除UI上的label显示
                self.label_numer_result.setText("0")
                self.label_time_result.setText('0')
                self.label_class_result.setText('None')
                font = QtGui.QFont()
                font.setPointSize(16)
                self.label_class_result.setFont(font)
                self.label_numer_score.setText("0")  # 显示置信度值
                # 清除位置坐标
                self.label_xmin_result.setText("0")
                self.label_ymin_result.setText("0")
                self.label_xmax_result.setText("0")
                self.label_ymax_result.setText("0")
                self.comboBox_select.currentIndexChanged.connect(self.select_obj)

                self.label_xmin_result_2.setText("0")
                self.label_xmax_result_2.setText("0")
                self.label_ymin_result_2.setText("0")
                self.label_ymax_result_2.setText("0")
                self.label_numer_score_2.setText("0")  # 伸出的手指数
                self.label_class_result_2.setText("None")
                QtWidgets.QApplication.processEvents()

            # self.label_numer_result.setText(str(len(results.multi_hand_landmarks)))
            time_end = time.time()  # 计时结束
            self.label_time_result.setText(str(round(1 / (time_end - time_start))))  # 显示用时
            QtWidgets.QApplication.processEvents()  # 立即执行

            self.disp_img(image)

        else:
            # 选择取消，恢复界面状态
            self.flag_timer = ""
            self.textEdit_pic.setText('文件未选中')
            self.textEdit_pic.setStyleSheet("background-color: transparent;\n"
                                            "border-color: rgb(0, 170, 255);\n"
                                            "color: rgb(0, 170, 255);\n"
                                            "font: regular 12pt \"华为仿宋\";")
            self.label_display.clear()  # 清除画面
            self.label_display.setStyleSheet("border-image: url(:/newPrefix/images_test/ini-image.png);")
            self.label_class_result.setText('None')
            self.label_time_result.setText('0')
            self.label_class_result.setText("None")
            self.label_numer_score.setText("0")

    def cv_imread(self, filePath):
        # 读取图片
        # cv_img = cv2.imread(filePath)
        cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
        # imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
        # cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
        if cv_img.shape[2] > 3:
            cv_img = cv_img[:, :, :3]
        return cv_img

    def drawRectBox(self, image, rect, addText):
        # 绘制标记框
        cv2.rectangle(image, (int(round(rect[0])), int(round(rect[1]))),
                      (int(round(rect[2])), int(round(rect[3]))),
                      (0, 0, 255), 2)
        cv2.rectangle(image, (int(rect[0] - 1), int(rect[1]) - 20), (int(rect[0] + 120), int(rect[1])), (0, 0, 255),
                      -1, cv2.LINE_AA)
        img = Image.fromarray(image)
        draw = ImageDraw.Draw(img)
        draw.text((int(rect[0] + 1), int(rect[1] - 20)), addText, (255, 255, 255), font=self.fontC)
        imagex = np.array(img)
        return imagex

    def button_open_register_click(self):
        # 点击人脸录入按钮执行

        # self.label_display.setText('正在录入人脸\n\nloading')
        Face_Register_con = Face_Register()
        Face_Register_con.run()

    def button_open_camera_click(self):
        # 点击摄像头按钮执行

        # 首先清除显示
        if self.timer_video.isActive():  # 停止视频定时器
            self.timer_video.stop()
        if self.cap_video:  # 释放视频画面
            self.cap_video.release()
        # 更新界面视频文本编辑框的文字
        # self.textEdit_video.setText('实时视频未选中')
        # self.textEdit_video.setStyleSheet("background-color: transparent;\n"
        #                                   "border-color: rgb(0, 170, 255);\n"
        #                                   "color: rgb(0, 170, 255);\n"
        #                                   "font: regular 12pt \"华为仿宋\";")

        if not self.timer_camera.isActive():  # 检查定时状态
            flag = self.cap.open(self.CAM_NUM)  # 检查相机状态
            files = os.listdir("../data/data_faces_from_camera")  # 检查人脸库状态
            if not flag:  # 相机打开失败提示
                # 提示相机打开失败的对话框
                msg = QtWidgets.QMessageBox.warning(self.centralwidget, u"Warning",
                                                    u"请检测相机与电脑是否连接正确！ ",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
                self.flag_timer = ""
            elif not files:
                # 人脸库为空
                msg_box = QMessageBox(QMessageBox.Warning, 'Warning', '人脸库为空，请先进行人脸登记')
                msg_box.exec_()
            else:
                # 准备运行识别程序
                self.flag_timer = "camera"

                QtWidgets.QApplication.processEvents()  # 启动进程执行
                # 更新摄像头文本编辑框
                self.textEdit_camera.setText('实时摄像已启动')
                self.textEdit_camera.setStyleSheet("background-color: transparent;\n"
                                                   "border-color: rgb(0, 170, 255);\n"
                                                   "color: rgb(0, 0, 255);\n"
                                                   "font: regular 18pt \"华文仿宋\";")
                # 在主显示界面提示
                self.label_display.setText('正在启动识别系统...\n\nloading')

                # 先调用人脸识别，识别通过则开启手势指令，否则结束
                Face_Recognizer_con = Face_Recognizer()
                face_name_list = Face_Recognizer_con.run()

                face_name = face_name_list[0]

                # 清除UI上的label显示
                self.label_numer_result.setText("0")  # 眼部个数
                self.label_time_result.setText('0')  # 检测时间
                self.label_class_result.setText('None')  # 检测结果
                self.label_class_result_3.setText("Unknown")
                font = QtGui.QFont()
                font.setPointSize(16)
                self.label_class_result.setFont(font)

                # 清除位置坐标
                self.label_xmin_result.setText("0")
                self.label_ymin_result.setText("0")
                self.label_xmax_result.setText("0")
                self.label_ymax_result.setText("0")

                QtWidgets.QApplication.processEvents()
                if face_name == "unknown":
                    msg_box = QMessageBox(QMessageBox.Warning, 'Warning', '尚未授权，请先进行人脸登记')
                    msg_box.exec_()
                    # 若定时器未开启，界面回复初始状态
                    self.timer_camera.stop()
                    if self.cap:
                        self.cap.release()
                    self.label_display.clear()  # 清除界面显示
                    # 重置摄像头文本框的文字
                    self.textEdit_camera.setText('实时摄像已关闭')
                    self.textEdit_camera.setStyleSheet("background-color: transparent;\n"
                                                       "border-color: rgb(0, 170, 255);\n"
                                                       "color: rgb(0, 170, 255);\n"
                                                       "font: regular 12pt \"华为仿宋\";")
                    # 重置主显示
                    self.label_display.setStyleSheet("border-image: url(:/newPrefix/images_test/human.png);")
                else:
                    # 打开相机定时器
                    self.mpHands = mp.solutions.hands
                    self.hands = self.mpHands.Hands()
                    self.mpDraw = mp.solutions.drawing_utils
                    self.timer_camera.start(30)
                    self.label_class_result_3.setText(face_name)
        else:
            # 若定时器未开启，界面回复初始状态
            self.flag_timer = ""
            self.timer_camera.stop()
            if self.cap:
                self.cap.release()
            self.label_display.clear()  # 清除界面显示

            # 重置摄像头文本框的文字
            self.textEdit_camera.setText('实时摄像已关闭')
            self.textEdit_camera.setStyleSheet("background-color: transparent;\n"
                                               "border-color: rgb(0, 170, 255);\n"
                                               "color: rgb(0, 170, 255);\n"
                                               "font: regular 12pt \"华为仿宋\";")
            # 重置主显示
            self.label_display.setStyleSheet("border-image: url(:/newPrefix/images_test/human.png);")

            # 清除UI上的label显示
            self.label_numer_result.setText("0")  # 眼部个数
            self.label_time_result.setText('0')  # 时间
            self.label_class_result.setText('None')  # 检测结果
            self.label_class_result_3.setText("Unknown")
            # 设置结果字体
            font = QtGui.QFont()
            font.setPointSize(16)
            self.label_class_result.setFont(font)
            # 清除位置坐标
            self.label_xmin_result.setText("0")
            self.label_ymin_result.setText("0")
            self.label_xmax_result.setText("0")
            self.label_ymax_result.setText("0")
            QtWidgets.QApplication.processEvents()

    def show_camera(self):
        # 定时器槽函数，每隔一段时间执行

        flag, image = self.cap.read()  # 获取画面

        if flag:
            # image = cv2.flip(image, 1)  # 左右翻转
            self.current_image = image.copy()

            image_height, image_width, _ = np.shape(image)
            imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB

            # 得到检测结果
            time_start = time.time()  # 开始计时
            count = 0
            results = self.hands.process(imgRGB)
            if results.multi_hand_landmarks:
                self.detInfo = []
                text_select = self.comboBox_select.currentText()

                for hand in results.multi_hand_landmarks:  # 多个手出现时表示出来
                    count += 1
                    # hand = results.multi_hand_landmarks[0]

                    # 采集所有关键点的坐标
                    list_lms = []
                    for i in range(21):
                        pos_x = hand.landmark[i].x * image_width
                        pos_y = hand.landmark[i].y * image_height
                        list_lms.append([int(pos_x), int(pos_y)])

                    # 构造凸包点
                    list_lms = np.array(list_lms, dtype=np.int32)

                    # 区域位置
                    xmin = list_lms[:, 0].min() - 20
                    ymin = list_lms[:, 1].min() - 20
                    xmax = list_lms[:, 0].max() + 20
                    ymax = list_lms[:, 1].max() + 20
                    bbox = [xmin, ymin, xmax, ymax]

                    hull_index = [0, 1, 2, 3, 6, 10, 14, 19, 18, 17, 10]
                    hull = cv2.convexHull(list_lms[hull_index, :])

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

                    user = "Default User"
                    passwd = "robotics"

                    # 以10帧为判决区间
                    if self.gesLegal():
                    # if str_guester != str_guester_list[-1]:
                        headers = {
                            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'
                        }
                        auth = HTTPDigestAuth(user, passwd)

                        # 3 -> start
                        if str_guester == Constant.START:
                                url = "https://577p16s514.oicp.vip:443/rw/rapid/execution?action=start"
                                data = {
                                    "regain": "continue", "execmode": "continue", "cycle": "forever", "condition": "none",
                                    "stopatbp": "disabled", "alltaskbytsp": "false"
                                }
                                r = requests.post(url, headers=headers, data=data, auth=auth)
                                print("start", r.status_code)

                            # 投放物料模式
                                # 生成物料 -> 1
                                url = 'https://577p16s514.oicp.vip/rw/rapid/symbol/data/RAPID/T_ROB1/Module1/nCount?json=1'
                                response = self.request(method='get', url=url, data="", auth=auth, headers=headers)
                                if (response.status_code == 200):
                                    json_data = json.loads(response.text)["_embedded"]["_state"][0]["value"]
                                    print(json_data)
                                else:
                                    print("访问失败")

                                # 放置物料的位置 -> 2
                                # 取物料
                                url = 'https://577p16s514.oicp.vip/rw/rapid/symbol/data/RAPID/T_ROB1/Module1/PosXY?action=set&json=1'
                                json_data = [30.5827,-16.8809,0]
                                data = {
                                    "value": str(json_data)
                                }
                                response = self.request(method='post', url=url, data=data, auth=auth, headers=headers)
                                if (response.status_code == 204):
                                    print("修改数据成功：")
                                else:
                                    print("访问失败",response.status_code)

                                # 获取物料坐标 -> 3
                                url = 'https://577p16s514.oicp.vip/rw/rapid/symbol/data/RAPID/T_ROB1/Module1/PosXY?json=1'
                                response = self.request(method='get', url=url, data="", auth=auth, headers=headers)
                                if (response.status_code == 200):
                                    json_data = json.loads(response.text)["_embedded"]["_state"][0]["value"]
                                    print(json_data)
                                else:
                                    print("访问失败",response.status_code)
                            # else:


                        # 5 -> stop
                        elif str_guester == Constant.STOP:
                                url = "https://577p16s514.oicp.vip:443/rw/rapid/execution?action=stop"
                                data = {
                                    "stopmode": "stop", "usetsp": "normal"
                                }
                                r = requests.post(url, headers=headers, data=data, auth=auth)
                                print("stop", r.status_code)

                        # 1 -> jog1_1
                        elif str_guester == Constant.JOG1_1:
                                # 取物料
                                url = 'https://577p16s514.oicp.vip/rw/rapid/symbol/data/RAPID/T_ROB1/Module1/number_count?action=set&json=1'
                                data = {
                                    "value": '1'
                                }
                                response = self.request(method='post', url=url, data=data, auth=auth, headers=headers)
                                if (response.status_code == 204):

                                    print("正常")
                                else:
                                    print("访问失败")

                                # 放物料
                                url = 'https://577p16s514.oicp.vip/rw/iosystem/signals/do_test1?action=set'
                                data = {
                                    "lvalue": '1'
                                }
                                response = self.request(method='post', url=url, data=data, auth=auth, headers=headers)
                                if (response.status_code == 204):
                                    # json_data = json.loads(response.text)["_embedded"]["_state"][0]["value"]
                                    print("修改IO信号成功：")
                                else:
                                    print("访问失败",response.status_code)
                        # 2 -> jog1_2
                        elif str_guester == Constant.JOG1_2:
                                # 取物料
                                url = 'https://577p16s514.oicp.vip/rw/rapid/symbol/data/RAPID/T_ROB1/Module1/number_count?action=set&json=1'
                                data = {
                                    "value": '2'
                                }
                                response = self.request(method='post', url=url, data=data, auth=auth, headers=headers)
                                if (response.status_code == 204):

                                    print("正常")
                                else:
                                    print("访问失败")

                                # 放物料
                                url = 'https://577p16s514.oicp.vip/rw/iosystem/signals/do_test1?action=set'
                                data = {
                                    "lvalue": '1'
                                }
                                response = self.request(method='post', url=url, data=data, auth=auth, headers=headers)
                                if (response.status_code == 204):
                                    # json_data = json.loads(response.text)["_embedded"]["_state"][0]["value"]
                                    print("修改IO信号成功：")
                                else:
                                    print("访问失败")

                        # 4 -> DELIVERY
                        elif str_guester == Constant.DELIVERY:
                            # 取物料
                            url = 'https://577p16s514.oicp.vip/rw/rapid/symbol/data/RAPID/T_ROB1/Module1/number_count?action=set&json=1'
                            data = {
                                "value": '3'
                            }
                            response = self.request(method='post', url=url, data=data, auth=auth, headers=headers)
                            if (response.status_code == 204):

                                print("正常")
                            else:
                                print("访问失败")

                            # 放物料
                            url = 'https://577p16s514.oicp.vip/rw/iosystem/signals/do_test1?action=set'
                            data = {
                                "lvalue": '1'
                            }
                            response = self.request(method='post', url=url, data=data, auth=auth, headers=headers)
                            if (response.status_code == 204):
                                # json_data = json.loads(response.text)["_embedded"]["_state"][0]["value"]
                                print("修改IO信号成功：")
                            else:
                                print("访问失败")

                    str_guester_list.append(str_guester)
                    self.detInfo.append([str_guester, bbox])

                    # cv2.putText(image, ' %s' % (str_guester), (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                    #             cv2.LINE_AA)
                    text = "手势{}：{}".format(count + 1, str_guester)

                    if text_select != "所有手势":
                        if text_select != text:
                            continue

                    for i in ll:
                        pos_x = hand.landmark[i].x * image_width
                        pos_y = hand.landmark[i].y * image_height
                        # 画点
                        cv2.circle(image, (int(pos_x), int(pos_y)), 3, (0, 255, 255), -1)

                    cv2.polylines(image, [hull], True, (0, 255, 0), 2)  # 绘制凸包
                    self.mpDraw.draw_landmarks(image, hand, self.mpHands.HAND_CONNECTIONS)

                    # 设置检测到的人脸位置坐标显示
                    # if count == 1:
                    self.label_xmin_result.setText(str(xmin))
                    self.label_xmax_result.setText(str(xmax))
                    self.label_ymin_result.setText(str(ymin))
                    self.label_ymax_result.setText(str(ymax))
                    self.label_numer_score.setText(str(len(up_fingers)))  # 伸出的手指数
                    self.label_class_result.setText(str_guester)
                    # else:
                    #     self.label_xmin_result_2.setText(str(xmin))
                    #     self.label_xmax_result_2.setText(str(xmax))
                    #     self.label_ymin_result_2.setText(str(ymin))
                    #     self.label_ymax_result_2.setText(str(ymax))
                    #     self.label_numer_score_2.setText(str(len(up_fingers)))  # 伸出的手指数
                    #     self.label_class_result_2.setText(str_guester)

                    image = self.drawRectBox(image, bbox, "手势" + str(count) + "：" + str_guester)

                # 更新下拉选框
                QtWidgets.QApplication.processEvents()
                self.comboBox_select.currentIndexChanged.disconnect(self.select_obj)
                self.comboBox_select.clear()
                self.comboBox_select.addItem('所有手势')
                for i in range(len(self.detInfo)):
                    text = "手势{}：{}".format(i + 1, self.detInfo[i][0])
                    self.comboBox_select.addItem(text)
                self.comboBox_select.currentIndexChanged.connect(self.select_obj)

                self.label_numer_result.setText(str(count))
                # if count == 1:
                #     self.label_xmin_result_2.setText("0")
                #     self.label_xmax_result_2.setText("0")
                #     self.label_ymin_result_2.setText("0")
                #     self.label_ymax_result_2.setText("0")
                #     self.label_numer_score_2.setText("0")  # 伸出的手指数
                #     self.label_class_result_2.setText("None")
            else:
                self.comboBox_select.currentIndexChanged.disconnect(self.select_obj)
                self.comboBox_select.clear()
                self.comboBox_select.addItem('所有手势')
                # 清除UI上的label显示
                self.label_numer_result.setText("0")
                self.label_time_result.setText('0')
                self.label_class_result.setText('None')
                font = QtGui.QFont()
                font.setPointSize(16)
                self.label_class_result.setFont(font)
                self.label_numer_score.setText("0")  # 显示置信度值
                # 清除位置坐标
                self.label_xmin_result.setText("0")
                self.label_ymin_result.setText("0")
                self.label_xmax_result.setText("0")
                self.label_ymax_result.setText("0")
                self.comboBox_select.currentIndexChanged.connect(self.select_obj)

                # 第二只手
                # self.label_xmin_result_2.setText("0")
                # self.label_xmax_result_2.setText("0")
                # self.label_ymin_result_2.setText("0")
                # self.label_ymax_result_2.setText("0")
                # self.label_numer_score_2.setText("0")  # 伸出的手指数
                # self.label_class_result_2.setText("None")
                QtWidgets.QApplication.processEvents()
                # 显示人脸个数
            # self.label_numer_result.setText(str(len(results.multi_hand_landmarks)))
            time_end = time.time()  # 计时结束
            self.label_time_result.setText(str(round(1 / (time_end - time_start))))  # 显示用时
            QtWidgets.QApplication.processEvents()  # 立即执行

            self.disp_img(image)
        else:
            self.timer_camera.stop()  # 无画面时停止计时器

    def select_obj(self):
        QtWidgets.QApplication.processEvents()
        if self.flag_timer == "video":
            # 打开定时器
            self.timer_video.start(30)
        elif self.flag_timer == "camera":
            self.timer_camera.start(30)

        ind = self.comboBox_select.currentIndex() - 1
        ind_select = ind
        if ind <= -1:
            ind_select = 0
        # else:
        #     ind_select = len(self.detInfo) - ind - 1
        if len(self.detInfo) > 0:
            if len(self.detInfo[ind_select][0]) > 7:
                font = QtGui.QFont()
                font.setPointSize(14)
            else:
                font = QtGui.QFont()
                font.setPointSize(16)
            self.label_class_result.setFont(font)
            self.label_class_result.setText(self.detInfo[ind_select][0])  # 显示类别
            # self.label_score_result.setText(str(self.detInfo[ind_select][2]))  # 显示置信度值
            # 显示位置坐标
            self.label_xmin_result.setText(str(int(self.detInfo[ind_select][1][0])))
            self.label_ymin_result.setText(str(int(self.detInfo[ind_select][1][1])))
            self.label_xmax_result.setText(str(int(self.detInfo[ind_select][1][2])))
            self.label_ymax_result.setText(str(int(self.detInfo[ind_select][1][3])))

        image = self.current_image.copy()
        if len(self.detInfo) > 0:
            for i, box in enumerate(self.detInfo):  # 遍历所有标记框

                if ind != -1:
                    if ind != i:
                        continue
                # 在图像上标记目标框
                image = self.drawRectBox(image, box[1], "手势" + str(i + 1) + "：" + box[0])

            # 在Qt界面中显示检测完成画面
            image = cv2.resize(image, (500, 500))  # 设定图像尺寸为显示界面大小
            show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
            self.label_display.setPixmap(QtGui.QPixmap.fromImage(showImage))
            self.label_display.setScaledContents(True)

    def button_open_video_click(self):
        # 点击视频按钮时执行

        if self.timer_camera.isActive():  # 检查相机定时器状态，若开则关闭
            self.timer_camera.stop()
        if self.cap:
            self.cap.release()  # 释放相机画面
        # 更新摄像文本编辑框提示文字
        self.textEdit_camera.setText('实时摄像已关闭')
        # 设置格式
        self.textEdit_camera.setStyleSheet("background-color: transparent;\n"
                                           "border-color: rgb(0, 170, 255);\n"
                                           "color: rgb(0, 170, 255);\n"
                                           "font: regular 12pt \"华为仿宋\";")

        if not self.timer_video.isActive():  # 检查定时状态
            # 弹出文件选择框选择视频文件
            fileName_choose, filetype = QFileDialog.getOpenFileName(self.centralwidget, "选取视频文件",
                                                                    self.video_path,  # 起始路径
                                                                    "视频(*.mp4;*.avi)")  # 文件类型
            # 视频路径
            self.video_path = fileName_choose

            if fileName_choose != '':  # 若路径存在
                self.flag_timer = "video"

                # 提示启动信息
                self.label_display.setText('正在启动识别系统...\n\nloading')
                QtWidgets.QApplication.processEvents()

                try:  # 初始化视频流
                    self.cap_video = cv2.VideoCapture(fileName_choose)
                except:
                    print("[INFO] could not determine # of frames in video")
                # 准备运行识别程序

                QtWidgets.QApplication.processEvents()  # 启动进程及时更新

                # 更新文本编辑框的文字提示
                self.textEdit_camera.setText('实时摄像未启动')
                self.textEdit_camera.setStyleSheet("background-color: transparent;\n"
                                                   "border-color: rgb(0, 170, 255);\n"
                                                   "color: rgb(0, 170, 255);\n"
                                                   "font: regular 12pt \"华为仿宋\";")
                self.textEdit_video.setText(fileName_choose + '文件已选中')
                self.textEdit_video.setStyleSheet("background-color: transparent;\n"
                                                  "border-color: rgb(0, 170, 255);\n"
                                                  "color: rgb(0, 170, 255);\n"
                                                  "font: regular 12pt \"华为仿宋\";")
                # 主显示界面的提示信息
                self.label_display.setText('正在启动识别系统...\n\nleading')

                # 清除UI上的label显示
                self.label_numer_result.setText("0")
                self.label_time_result.setText('0')
                self.label_class_result.setText('None')
                font = QtGui.QFont()
                font.setPointSize(16)
                self.label_class_result.setFont(font)
                # 清除位置坐标
                self.label_xmin_result.setText("0")
                self.label_ymin_result.setText("0")
                self.label_xmax_result.setText("0")
                self.label_ymax_result.setText("0")
                QtWidgets.QApplication.processEvents()
                self.mpHands = mp.solutions.hands
                self.hands = self.mpHands.Hands()
                self.mpDraw = mp.solutions.drawing_utils
                # 打开视频定时器
                self.timer_video.start(5)

            else:
                # 选择取消，恢复界面状态
                self.flag_timer = ""
                # 提示文本框信息
                self.textEdit_camera.setText('实时摄像已关闭')
                self.textEdit_camera.setStyleSheet("background-color: transparent;\n"
                                                   "border-color: rgb(0, 170, 255);\n"
                                                   "color: rgb(0, 170, 255);\n"
                                                   "font: regular 12pt \"华为仿宋\";")
                self.textEdit_video.setText('实时视频未选中')
                self.textEdit_video.setStyleSheet("background-color: transparent;\n"
                                                  "border-color: rgb(0, 170, 255);\n"
                                                  "color: rgb(0, 170, 255);\n"
                                                  "font: regular 12pt \"华为仿宋\";")
                self.label_display.clear()  # 清除画面
                self.label_display.setStyleSheet("border-image: url(:/newPrefix/images_test/ini-image.png);")  # 重置画面
                self.label_class_result.setText('None')  # 结果显示
                self.label_time_result.setText('0')  # 时间显示

        else:
            # 定时器未开启，则界面回复初始状态
            self.flag_timer = ""
            self.timer_video.stop()  # 停止定时器
            self.cap_video.release()  # 释放视频画面
            self.label_display.clear()  # 清除显示
            # 重置文本框显示
            self.textEdit_camera.setText('实时摄像已关闭')
            self.textEdit_camera.setStyleSheet("background-color: transparent;\n"
                                               "border-color: rgb(0, 170, 255);\n"
                                               "color: rgb(0, 170, 255);\n"
                                               "font: regular 12pt \"华为仿宋\";")
            self.textEdit_video.setText('实时视频已关闭')
            self.textEdit_video.setStyleSheet("background-color: transparent;\n"
                                              "border-color: rgb(0, 170, 255);\n"
                                              "color: rgb(0, 170, 255);\n"
                                              "font: regular 12pt \"华为仿宋\";")
            # 重置主显示画面
            self.label_display.setStyleSheet("border-image: url(:/newPrefix/images_test/ini-image.png);")

            # 清除UI上的label显示
            self.label_numer_result.setText("0")
            self.label_time_result.setText('0')
            self.label_class_result.setText('None')
            font = QtGui.QFont()
            font.setPointSize(16)
            self.label_class_result.setFont(font)
            # 清除位置坐标
            self.label_xmin_result.setText("0")
            self.label_ymin_result.setText("0")
            self.label_xmax_result.setText("0")
            self.label_ymax_result.setText("0")
            QtWidgets.QApplication.processEvents()

    def show_video(self):
        # 定时器槽函数，每隔一段时间执行
        flag, image = self.cap_video.read()  # 获取画面

        if flag:
            self.current_image = image.copy()

            image_height, image_width, _ = np.shape(image)
            imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB

            # 得到检测结果
            time_start = time.time()  # 开始计时
            count = 0
            results = self.hands.process(imgRGB)
            if results.multi_hand_landmarks:
                self.detInfo = []
                text_select = self.comboBox_select.currentText()

                for hand in results.multi_hand_landmarks:  # 多个手出现时表示出来
                    count += 1
                    # hand = results.multi_hand_landmarks[0]

                    # 采集所有关键点的坐标
                    list_lms = []
                    for i in range(21):
                        pos_x = hand.landmark[i].x * image_width
                        pos_y = hand.landmark[i].y * image_height
                        list_lms.append([int(pos_x), int(pos_y)])

                    # 构造凸包点
                    list_lms = np.array(list_lms, dtype=np.int32)

                    # 区域位置
                    xmin = list_lms[:, 0].min() - 20
                    ymin = list_lms[:, 1].min() - 20
                    xmax = list_lms[:, 0].max() + 20
                    ymax = list_lms[:, 1].max() + 20
                    bbox = [xmin, ymin, xmax, ymax]

                    hull_index = [0, 1, 2, 3, 6, 10, 14, 19, 18, 17, 10]
                    hull = cv2.convexHull(list_lms[hull_index, :])

                    # 查找外部的点数
                    n_fig = -1
                    ll = [4, 8, 12, 16, 20]
                    up_fingers = []

                    for i in ll:
                        pt = (int(list_lms[i][0]), int(list_lms[i][1]))
                        dist = cv2.pointPolygonTest(hull, pt, True)
                        if dist < 0:
                            up_fingers.append(i)

                    str_guester = get_str_guester(up_fingers, list_lms)
                    self.detInfo.append([str_guester, bbox])

                    # cv2.putText(image, ' %s' % (str_guester), (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                    #             cv2.LINE_AA)
                    text = "手势{}：{}".format(count + 1, str_guester)

                    if text_select != "所有手势":
                        if text_select != text:
                            continue

                    for i in ll:
                        pos_x = hand.landmark[i].x * image_width
                        pos_y = hand.landmark[i].y * image_height
                        # 画点
                        cv2.circle(image, (int(pos_x), int(pos_y)), 3, (0, 255, 255), -1)

                    cv2.polylines(image, [hull], True, (0, 255, 0), 2)  # 绘制凸包
                    self.mpDraw.draw_landmarks(image, hand, self.mpHands.HAND_CONNECTIONS)

                    # 设置检测到的人脸位置坐标显示
                    if count == 1:
                        self.label_xmin_result.setText(str(xmin))
                        self.label_xmax_result.setText(str(xmax))
                        self.label_ymin_result.setText(str(ymin))
                        self.label_ymax_result.setText(str(ymax))
                        self.label_numer_score.setText(str(len(up_fingers)))  # 伸出的手指数
                        self.label_class_result.setText(str_guester)
                    else:
                        self.label_xmin_result_2.setText(str(xmin))
                        self.label_xmax_result_2.setText(str(xmax))
                        self.label_ymin_result_2.setText(str(ymin))
                        self.label_ymax_result_2.setText(str(ymax))
                        self.label_numer_score_2.setText(str(len(up_fingers)))  # 伸出的手指数
                        self.label_class_result_2.setText(str_guester)

                    # image = self.drawRectBox(image, bbox, "手势：" + str_guester)
                    image = self.drawRectBox(image, bbox, "手势" + str(count) + "：" + str_guester)

                # 更新下拉选框
                QtWidgets.QApplication.processEvents()
                self.comboBox_select.currentIndexChanged.disconnect(self.select_obj)
                self.comboBox_select.clear()
                self.comboBox_select.addItem('所有手势')
                for i in range(len(self.detInfo)):
                    text = "手势{}：{}".format(i + 1, self.detInfo[i][0])
                    self.comboBox_select.addItem(text)
                self.comboBox_select.currentIndexChanged.connect(self.select_obj)
                self.label_numer_result.setText(str(count))
                if count == 1:
                    self.label_xmin_result_2.setText("0")
                    self.label_xmax_result_2.setText("0")
                    self.label_ymin_result_2.setText("0")
                    self.label_ymax_result_2.setText("0")
                    self.label_numer_score_2.setText("0")  # 伸出的手指数
                    self.label_class_result_2.setText("None")
            else:
                self.comboBox_select.currentIndexChanged.disconnect(self.select_obj)
                self.comboBox_select.clear()
                self.comboBox_select.addItem('所有手势')
                # 清除UI上的label显示
                self.label_numer_result.setText("0")
                self.label_time_result.setText('0')
                self.label_class_result.setText('None')
                font = QtGui.QFont()
                font.setPointSize(16)
                self.label_class_result.setFont(font)
                self.label_numer_score.setText("0")  # 显示置信度值
                # 清除位置坐标
                self.label_xmin_result.setText("0")
                self.label_ymin_result.setText("0")
                self.label_xmax_result.setText("0")
                self.label_ymax_result.setText("0")
                self.comboBox_select.currentIndexChanged.connect(self.select_obj)
                QtWidgets.QApplication.processEvents()
                self.label_xmin_result_2.setText("0")
                self.label_xmax_result_2.setText("0")
                self.label_ymin_result_2.setText("0")
                self.label_ymax_result_2.setText("0")
                self.label_numer_score_2.setText("0")  # 伸出的手指数
                self.label_class_result_2.setText("None")
                # 显示人脸个数
            time_end = time.time()  # 计时结束
            self.label_time_result.setText(str(round(1 / (time_end - time_start))))  # 显示用时
            QtWidgets.QApplication.processEvents()  # 立即执行

            self.disp_img(image)

        else:
            # 否则关闭定时器
            self.timer_video.stop()

    def disp_img(self, image):
        # self.label_display.clear()
        image = cv2.resize(image, (500, 500))  # 设定图像尺寸为显示界面大小
        show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        a = QtGui.QPixmap.fromImage(showImage)
        self.label_display.setPixmap(a)
        self.label_display.setScaledContents(True)
        QtWidgets.QApplication.processEvents()

    def request(cls, method, url, data, auth=None, headers=None):  # 这里分别需要传人
        method = method.upper()  # 这里将传入的请求方法统一大写，然后进行判断采用什么方法
        if method == 'POST':
            return requests.post(url=url, data=data, auth=auth, headers=headers)
        elif method == 'GET':
            return requests.get(url=url, params=data, auth=auth, headers=headers)
        return f"目前没有{method}请求方法，只有POST和Get请求方法！"

    def gesLegal(self):
        length = len(str_guester_list)
        if length < 11:
            return False
        for i in range(length-10, length-1):
            if str_guester_list[i] != str_guester_list[i+1]:
                return False
        str_guester_list.clear()
        return True
