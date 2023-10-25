import math

import cvzone
import dlib
import cv2
import numpy as np
import os.path
from scipy.spatial import distance as dist
import logging
from logging.handlers import RotatingFileHandler
import traceback
import platform
from datetime import datetime

from imutils import face_utils
from ultralytics import YOLO

logger = logging.getLogger("FACE RECOGNITION LOG")
logger.setLevel(logging.ERROR)
handler = RotatingFileHandler("FR_Log.txt", maxBytes=100000, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

last_user_id = 0
user_name = " "
loading_status = ""

# indica status atual da deteção: 0 = fora de execução, 1 = em execução , 2= finalizar execução
status = 0

last_time = datetime.now()


def show_time(description):
    global last_time

    try:
        new_time = datetime.now()
        diff = new_time - last_time
        print(str(new_time) + " - " + description + " - Diff: " + str(diff.total_seconds()))
        last_time = new_time
    except Exception as e:

        logger.error(str(e))
        logger.error(traceback.format_exc())


def set_status_recognition(value):
    global status
    global last_user_id
    global user_name
    status = value
    if status == 1:
        last_user_id = 0
        user_name = ""


def get_last_user_id():
    return last_user_id


def set_last_user_id(value):
    global last_user_id
    last_user_id = value


def get_loading_status():
    global loading_status
    return loading_status


def get_status_recognition():
    return status


def check_dependencies(main_path):
    code = 1
    msg = ""

    shape_predictor = "shape_predictor_68_face_landmarks.dat"
    recognizer_model = "dlib_face_recognition_resnet_model_v1.dat"
    train_file_name = "DK_trainedFacialDescriptors.npy"
    indexes_file_name = "DK_indexes.pickle"

    if not os.path.exists(main_path + shape_predictor):
        msg += shape_predictor + " não encontrado. "
    if not os.path.exists(main_path + recognizer_model):
        msg += recognizer_model + " não encontrado. "
    if not os.path.exists(main_path + train_file_name):
        msg += train_file_name + " não encontrado. "
    if not os.path.exists(main_path + indexes_file_name):
        msg += indexes_file_name + " não encontrado. "

    if len(msg) > 1:
        code = -1

    return {"code": code, "msg": msg}


class RecognitionCamera(object):
    def __init__(self, camera, res_width, res_height, upsample_rate, main_path, threshold, recognition_rate,
                 finish_when_found, eye_threshold, eye_consec_frames):  # , window_width, window_height):

        global loading_status

        if loading_status != "B":
            print("RecognitionCamera is loading...")

            loading_status = "B"  ##BUSY

            show_time("1")
            # capturing video
            # instanciando variavel de acesso à webcam
            os_name = platform.system().lower()
            print(os_name)
            if "win" in os_name:
                self.video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
            else:  # linux
                self.video = cv2.VideoCapture(camera)

            self.video.set(cv2.CAP_PROP_FRAME_WIDTH, res_width)
            self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, res_height)
            # self.video.set(cv2.CAP_PROP_ZOOM, 220)

            shape_predictor = "shape_predictor_68_face_landmarks.dat"
            recognizer_model = "dlib_face_recognition_resnet_model_v1.dat"
            self.train_file_name = "DK_trainedFacialDescriptors.npy"
            self.indexes_file_name = "DK_indexes.pickle"

            # showTime("pre hog detector")
            # obtendo detector de face padrão do DLIB
            self.hog_detector = dlib.get_frontal_face_detector()

            # showTime("pre shape predictor")
            # obtendo detector de pontos faciais
            self.detector_points = dlib.shape_predictor(main_path + shape_predictor)

            show_time("pre recognizer")
            # obtendo reconhecedor (extraidor de descritores faciais)
            self.recognizer = dlib.face_recognition_model_v1(main_path + recognizer_model)

            self.camera = camera

            self.res_width = res_width
            self.res_height = res_height
            # self.window_width = window_width
            # self.window_height = window_height
            self.main_path = main_path
            self.upsample_rate = upsample_rate
            self.finish_when_found = finish_when_found

            self.threshold = threshold

            show_time("Pre updateFaces")

            # carregando faces treinadas
            self.indexes = np.load(main_path + self.indexes_file_name, allow_pickle=True)
            self.trained_faces = np.load(main_path + self.train_file_name, allow_pickle=True)

            show_time("POS updateFaces")

            self.bgr = (255, 255, 255)  # branco

            self.finish_camera = False
            self.frame_count = 0
            if recognition_rate == 0:
                self.recognition_rate = 1
            else:
                self.recognition_rate = recognition_rate
            self.last_jpeg = None

            self.text = ""

            global last_user_id
            global user_name

            last_user_id = 0
            user_name = ""

            # define two constants, one for the eye aspect ratio to indicate
            # blink and then a second constant for the number of consecutive
            # frames the eye must be below the threshold
            self.EYE_AR_THRESH = eye_threshold  # 0.22
            self.EYE_AR_CONSEC_FRAMES = eye_consec_frames  # 2

            # grab the indexes of the facial landmarks for the left and
            # right eye, respectively
            (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

            self.COUNTER = 0
            self.TOTAL = 0
            self.users_that_blinked = {}
            self.last_distance = " "

            self.model = YOLO("../files/l_version_1_300.pt")
            self.classNames = ["fake", "real"]
            self.confidence = 0.6
            results = self.model(None, stream=True, verbose=False)

            loading_status = "F"  # FREE
            print("RecognitionCamera loaded...")
        else:
            print("RecognitionCamera is busy...")

        # threading.Thread(target=self.start_stream,daemon=False).start()

    def __del__(self):
        # releasing camera
        self.video.release()
        print("--- camera release ---")

    def load_recognition(self, camera, res_width, res_height, upsample_rate, main_path, threshold, recognition_rate,
                         finish_when_found, eye_threshold, eye_consec_frames):
        global loading_status

        if loading_status != "B":
            print("Load recognition is loading...")

            self.video.release()

            loading_status = "B"  ##BUSY

            show_time("1")
            # capturing video
            # instanciando variavel de acesso à webcam
            os_name = platform.system().lower()
            print(os_name)
            if "win" in os_name:
                self.video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
            else:  # linux
                self.video = cv2.VideoCapture(camera)

            self.video.set(cv2.CAP_PROP_FRAME_WIDTH, res_width)
            self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, res_height)
            # self.video.set(cv2.CAP_PROP_ZOOM, 220)

            self.camera = camera

            self.res_width = res_width
            self.res_height = res_height

            self.main_path = main_path
            self.upsample_rate = upsample_rate
            self.finish_when_found = finish_when_found

            self.threshold = threshold

            show_time("Pre updateFaces")

            # carregando faces treinadas
            self.indexes = np.load(main_path + self.indexes_file_name, allow_pickle=True)
            self.trained_faces = np.load(main_path + self.train_file_name, allow_pickle=True)

            show_time("POS updateFaces")

            self.bgr = (255, 255, 255)  # branco

            self.finish_camera = False
            self.frame_count = 0
            if recognition_rate == 0:
                self.recognition_rate = 1
            else:
                self.recognition_rate = recognition_rate
            self.last_jpeg = None

            self.text = ""

            global last_user_id
            global user_name

            last_user_id = 0
            user_name = ""

            # define two constants, one for the eye aspect ratio to indicate
            # blink and then a second constant for the number of consecutive
            # frames the eye must be below the threshold
            self.EYE_AR_THRESH = eye_threshold  # 0.22
            self.EYE_AR_CONSEC_FRAMES = eye_consec_frames  # 2

            # grab the indexes of the facial landmarks for the left and
            # right eye, respectively
            (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

            self.COUNTER = 0
            self.TOTAL = 0
            self.users_that_blinked = {}
            self.last_distance = " "

            loading_status = "F"  # FREE
            print("Load recognition finished...")
        else:
            print("RecognitionCamera is busy...")

    def update_trained_faces(self):
        show_time("Pre updateFaces")
        # carregando faces treinadas
        self.indexes = np.load(self.main_path + self.indexes_file_name, allow_pickle=True)
        self.trained_faces = np.load(self.main_path + self.train_file_name, allow_pickle=True)
        show_time("POS updateFaces")

    def new_gamma_image(self, image, gamma):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table

        image_lut = cv2.LUT(image, table)

        return image_lut

    def get_frame(self):

        #     return self.last_jpeg
        #
        #
        # def start_stream(self):

        global last_user_id
        global user_name
        global status

        try:

            # while self.video.isOpened() and status != 2:

            self.frame_count += 1
            # print(self.frame_count)

            # extracting frames
            connected, image_raw = self.video.read()

            if not connected:
                logger.error(
                    "Não foi possível obter o frame da câmera {} . Verifique se a mesma foi/está desconectada.".format(
                        self.camera))
                status = 2

                return self.last_jpeg

            image_grey_1c = cv2.cvtColor(image_raw, cv2.COLOR_BGRA2GRAY)
            image_grey_3c = cv2.cvtColor(image_grey_1c, cv2.COLOR_GRAY2BGR)
            image = self.new_gamma_image(image_grey_3c, 2)

            # somente executar enquanto nenhum usuário for reconhecido
            if last_user_id == 0 or self.finish_when_found == 0 or not self.users_that_blinked.get(last_user_id) == 2:

                # The second argument indicates that we should upsample the image
                # x times.  This will make everything bigger and allow us to detect more
                faces = self.hog_detector(image, self.upsample_rate)

                faces_len = len(faces)
                # se em um frame nenhuma face ou mais de uma face for encontrada:
                if faces_len != 1:
                    # limpe a lista de usuário que piscaram
                    self.users_that_blinked.clear()
                    last_user_id = 0

                    top = self.res_height - 30

                    if faces_len == 0:
                        warning = "Nenhuma face encontrada."
                        cv2.putText(image_raw, warning, (30, top), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1,
                                    cv2.LINE_4)
                    else:
                        warning = "Mais de uma face encontrada."
                        warning2 = "Mantenha somente uma pessoa na imagem."
                        cv2.putText(image_raw, warning, (30, top), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1,
                                    cv2.LINE_4)
                        cv2.putText(image_raw, warning2, (30, top + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1,
                                    cv2.LINE_4)



                else:  # 1 face encontrada

                    show_shape = True

                    face = faces[0]

                    # obtendo pontos faciais
                    facial_points = self.detector_points(image, face)

                    shape = face_utils.shape_to_np(facial_points)

                    # obtendo pontos da imagem onde a face foi encontrada
                    left, top, right, bottom = (
                        int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))

                    # cheque piscada de olhos assim que o usuário for reconhecido
                    if last_user_id > 0 and self.users_that_blinked.get(last_user_id) is None:
                        show_shape = False
                        self.check_eyes(image_raw, facial_points)

                        # usuário reconhecido piscou?
                        blink_status = self.users_that_blinked.get(last_user_id)

                        if blink_status == 1:

                            self.bgr = (0, 140, 255)  # laranja
                            # self.bgr = (0, 255, 0)  # verde

                        else:
                            self.bgr = (0, 255, 255)  # amarelo
                    else:

                        if last_user_id > 0 and self.users_that_blinked.get(last_user_id) is not None:
                            show_shape = False

                        if self.frame_count % self.recognition_rate == 0 or self.frame_count == 1:

                            # if last_user_id > 0 and self.users_that_blinked.get(last_user_id) is not None:
                            if not show_shape:
                                cls, conf = self.check_true_face(image_raw)

                                if cls is not None and conf is not None:
                                    self.text = f'{self.classNames[cls].upper()} {int(conf * 100)}%'
                                    if conf > self.confidence:
                                        if self.classNames[cls] == 'real':
                                            self.bgr = (0, 255, 0)  # verde
                                        else:
                                            self.bgr = (0, 0, 255)  # vermelho
                                    else:
                                        self.bgr = (0, 0, 0)  # preto
                                else:
                                    self.bgr = (255,255,255)  # branco
                            else:
                                # showDiff("Depois DetectorPoints")

                                # obtendo caracteristicas principais da face
                                facial_descriptors = self.recognizer.compute_face_descriptor(image, facial_points)

                                # showDiff("Depois Recognizer")

                                # transformando caracteristicas em uma lista
                                facial_descriptors_list = [fd for fd in facial_descriptors]
                                # lista para array numpy
                                facial_descriptors_np_array = np.asarray(facial_descriptors_list, dtype=np.float64)
                                # adicionando nova "coluna" ao array numpy
                                facial_descriptors_np_array = facial_descriptors_np_array[np.newaxis, :]

                                # obtendo a distancia de TODAS AS IMAGENS TREINADAS em comparação à recebida
                                distances = np.linalg.norm(facial_descriptors_np_array - self.trained_faces, axis=1)

                                # obtendo o elemento com a MENOR distância (quanto menor mais proxima)
                                closest_element = np.argmin(distances)

                                # obtendo valor da menor distancia
                                closest_element_distance = distances[closest_element]
                                self.last_distance = str(closest_element_distance.round(3))

                                if closest_element_distance <= self.threshold:
                                    # nome padrão imagens: User.UserId.UserName.ImageId.png. Ex: User.457.Joao.5.png
                                    last_user_id = int(os.path.split(self.indexes[closest_element])[1].split(".")[1])
                                    user_name = str(os.path.split(self.indexes[closest_element])[1].split(".")[2])
                                    self.text = str(last_user_id) + " - " + user_name  # + " - " + str(
                                    #   closest_element_distance.round(3))

                                    # usuário reconhecido piscou?
                                    blink_status = self.users_that_blinked.get(last_user_id)

                                    if blink_status == 1:
                                        user = {last_user_id: 2}
                                        self.users_that_blinked.update(user)
                                        self.bgr = (0, 255, 0)  # verde
                                    elif blink_status == 2:
                                        self.bgr = (0, 255, 0)  # verde
                                    else:
                                        self.bgr = (0, 255, 255)  # amarelo


                                else:
                                    last_user_id = 0;
                                    self.text = 'Desconhecido '
                                    self.bgr = (0, 0, 255)
                                    self.users_that_blinked.clear()



                    # y = top - 15 if top - 15 > 15 else top + 15
                    # cv2.rectangle(image_raw, (left, top), (right, bottom), self.bgr, 1)
                    #
                    # cv2.putText(image_raw, self.text, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.bgr, 1, cv2.LINE_8)
                    # cv2.putText(image_raw, self.last_distance, (self.res_width - 100, self.res_height - 30),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.bgr, 1, cv2.LINE_8)

                    cvzone.cornerRect(image_raw, (left, top, right-left, bottom-top), colorC=self.bgr, colorR=self.bgr)
                    cvzone.putTextRect(image_raw, self.text, (max(0, left), max(35, top)), scale=2, thickness=4,
                                       colorR=self.bgr,
                                       colorB=self.bgr)

                    if show_shape:
                        # loop over the (x, y)-coordinates for the facial landmarks
                        # and draw them on the image
                        for (x, y) in shape:
                            cv2.circle(image_raw, (x, y), 1, (255, 255, 255), -1)


            else:

                status = 2
                # self.video.release()
                # salvando imagem inteira
                cv2.imwrite(self.main_path + "last_capture.png", image_raw)

            # encode OpenCV raw frame to jpg
            ret, jpeg = cv2.imencode('.jpg', image_raw)
            self.last_jpeg = jpeg.tobytes()

            return self.last_jpeg

        except Exception as e:
            status = 2
            logger.error(str(e))
            logger.error(traceback.format_exc())

    def check_true_face(self, img):
        cls = None
        conf = None
        # show_time("pre model yolo")
        results = self.model(img, stream=True, verbose=False)
        show_time("pos model yolo")
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])
                # if conf > self.confidence:
                #
                #     # if self.classNames[cls] == 'real':
                #     #     color = (0, 255, 0)
                #     # else:
                #     #     color = (0, 0, 255)

        # cv2.putText(img, f'{self.classNames[cls].upper()} {int(conf * 100)}%', (self.res_width - 100, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_8)

        show_time("pre return")
        return cls, conf

    def check_eyes(self, image_raw, facial_points):

        global last_user_id

        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array

        shape = face_utils.shape_to_np(facial_points)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[self.lStart:self.lEnd]
        rightEye = shape[self.rStart:self.rEnd]
        leftEAR = self.eye_aspect_ratio(leftEye)
        rightEAR = self.eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(image_raw, [leftEyeHull], -1, (255, 255, 255), 1)
        cv2.drawContours(image_raw, [rightEyeHull], -1, (255, 255, 255), 1)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < self.EYE_AR_THRESH:
            self.COUNTER += 1

        # otherwise, the eye aspect ratio is not below the blink
        # threshold
        else:
            # if the eyes were closed for a sufficient number of
            # then increment the total number of blinks
            if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                self.TOTAL += 1

                # indique que usuário piscou
                user = {last_user_id: 1}
                self.users_that_blinked.update(user)

            # reset the eye frame counter
            self.COUNTER = 0

        # draw the total number of blinks on the frame along with
        # the computed eye aspect ratio for the frame
        # cv2.putText(image_raw, "Blinks: {}".format(self.TOTAL), (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(image_raw, "{:.2f}".format(ear), (self.res_width - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_8)

    def eye_aspect_ratio(self, eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])

        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)

        # return the eye aspect ratio
        return ear
