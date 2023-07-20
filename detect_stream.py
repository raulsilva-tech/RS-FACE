import platform

from imutils import face_utils
from scipy.spatial import distance as dist
import dlib
import cv2
import numpy as np
import sys
import os.path
from datetime import datetime
import time

import logging
from logging.handlers import RotatingFileHandler
import traceback

logger = logging.getLogger("FACE RECOGNITION LOG")
logger.setLevel(logging.ERROR)
handler = RotatingFileHandler("FR_Log.txt", maxBytes=100000, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# indica status atual da deteção: 0 = fora de exeução, 1 = em execução , 2= finalizar execução
status = 0

# define se o usuário pediu para tirar foto
capture_now = False

# indica que há muitas faces na imagem
hog_too_many_faces = False

# indica que não há face na imagem
hog_zero_faces = False

# ultimo descritor de face gerado
last_descriptor = ""
last_descriptor_gamma1 = ""
last_descriptor_gamma2 = ""

last_user_id = 0


def set_capture_now():
    msg = ""

    global capture_now
    global hog_zero_faces
    global hog_too_many_faces

    capture_now = True
    hog_zero_faces = False
    hog_too_many_faces = False

    count = 0

    # loop até que capture_now receba false , que será o momento em que a imagem terá sido gerada
    while capture_now and count < 25:  # 25 iterações = 5 segundos
        count += 1
        time.sleep(0.2)

    if hog_zero_faces:
        code = -2
    elif hog_too_many_faces:
        code = -3
    elif last_user_id > 0:
        code = last_user_id
    else:
        code = 0

    return {"code": code, "msg": msg}


def get_hog_too_many_faces():
    return hog_too_many_faces


def get_hog_zero_faces():
    return hog_zero_faces


def set_status_detection(value):
    global status
    status = value


def get_status_detection():
    return status


def get_last_descriptor():
    return last_descriptor


def get_last_descriptor_gamma(number):
    if number == 1:
        return last_descriptor_gamma1
    elif number == 2:
        return last_descriptor_gamma2


class DetectionCamera(object):
    def __init__(self, camera, res_width, res_height, upsample_rate, main_path, detection_rate, recognize_threshold,
                 eye_threshold, eye_consec_frames):
        # capturing video
        # instanciando variavel de acesso à webcam
        os_name = platform.system().lower()
        print(os_name)
        if "win" in os_name:
            self.video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
        else: #se linux:
            self.video = cv2.VideoCapture(camera)

        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, res_width)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, res_height)
        self.video.set(cv2.CAP_PROP_ZOOM, 220)

        shape_predictor = "shape_predictor_68_face_landmarks.dat"
        recognizer_model = "dlib_face_recognition_resnet_model_v1.dat"

        # obtendo detector de face padrão do DLIB
        self.hog_detector = dlib.get_frontal_face_detector()
        # obtendo detector de pontos faciais
        self.detector_points = dlib.shape_predictor(main_path + shape_predictor)
        # obtendo reconhecedor (extraidor de descritores faciais)
        self.recognizer = dlib.face_recognition_model_v1(main_path + recognizer_model)

        self.camera = camera
        self.res_width = res_width
        self.res_height = res_height
        self.main_path = main_path
        self.upsample_rate = upsample_rate
        if detection_rate == 0:
            self.detection_rate = 1
        else:
            self.detection_rate = detection_rate
        self.frame_count = 0
        self.last_jpeg = None

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

        train_file_name = "DK_trainedFacialDescriptors.npy"
        indexes_file_name = "DK_indexes.pickle"
        if os.path.exists(main_path + indexes_file_name) and os.path.exists(main_path + train_file_name):
            self.indexes = np.load(main_path + indexes_file_name, allow_pickle=True)
            self.trained_faces = np.load(main_path + train_file_name, allow_pickle=True)
            self.threshold = recognize_threshold
            self.necessary_files_exist = True
        else:
            self.necessary_files_exist = False

    def __del__(self):
        print("--- camera release ---")
        self.video.release()

    def get_descriptor_from_this_image(self, image):

        # The second argument indicates that we should upsample the image
        # x times.  This will make everything bigger and allow us to detect more
        faces = self.hog_detector(image, self.upsample_rate)

        faces_length = len(faces)

        if faces_length == 0:

            return ""

        else:

            # print(faces)
            for face in faces:
                # obtendo pontos faciais
                facial_points = self.detector_points(image, face)
                # descritor facial composto por 128 descritores resultantes da analise do algoritmo
                facial_descriptors = self.recognizer.compute_face_descriptor(image, facial_points)

                # convertendo descritor DLib para uma lista
                facial_descriptors_list = [df for df in facial_descriptors]

                descriptors = str(facial_descriptors_list)

            return descriptors

    def new_gamma_image(self, image, gamma):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table

        image_lut = cv2.LUT(image, table)

        return image_lut

    def generate_gamma_images(self, main_image):

        global last_descriptor_gamma1
        global last_descriptor_gamma2

        # gerando nova imagem com gamma = 1.5
        image_lut1 = self.new_gamma_image(main_image, 1.5)
        # salvando arquivo
        cv2.imwrite(self.main_path + "last_capture_gamma1.png", image_lut1)
        # gerando descritores da nova imagem
        last_descriptor_gamma1 = self.get_descriptor_from_this_image(image_lut1)

        # gerando nova imagem com gamma = 0.5
        image_lut2 = self.new_gamma_image(main_image, 0.5)
        # salvando arquivo
        cv2.imwrite(self.main_path + "last_capture_gamma2.png", image_lut2)
        # gerando descritores da nova imagem
        last_descriptor_gamma2 = self.get_descriptor_from_this_image(image_lut2)

    def recognize_this_face(self, facial_descriptors_list):

        if self.necessary_files_exist:

            global last_user_id
            last_user_id = 0

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

            if closest_element_distance <= self.threshold:
                # nome padrão imagens: User.UserId.UserName.ImageId.png. Ex: User.457.Joao.5.png
                last_user_id = int(os.path.split(self.indexes[closest_element])[1].split(".")[1])

    def get_frame(self):

        try:

            self.frame_count += 1

            global capture_now
            global hog_too_many_faces
            global hog_zero_faces
            global last_descriptor
            global status

            # if not self.video.isOpened():
            #     logger.error(
            #         "Não foi possível obter o frame da câmera {} . Verifique se a mesma foi/está desconectada.".format(
            #             self.camera))
            #     status = 2
            #     return None

            # extracting frames
            connected, image= self.video.read()

            if not connected:
                logger.error(
                    "Não foi possível obter o frame da câmera {} . Verifique se a mesma foi/está desconectada.".format(
                        self.camera))

                status = 2

                return self.last_jpeg.tobytes()

            # if self.frame_count % self.detection_rate == 0 or self.frame_count == 1:

            # The second argument indicates that we should upsample the image
            # x times.  This will make everything bigger and allow us to detect more
            faces = self.hog_detector(image, self.upsample_rate)

            faces_length = len(faces)

            if faces_length == 1:
                # cor retangulo = branco
                bgr = (255, 255, 255)
            else:
                # cor retangulo = vermelho
                bgr = (0, 0, 255)

            if faces_length == 0:

                # usuário selecionou TIRAR FOTO no disp
                if capture_now:
                    hog_too_many_faces = False
                    hog_zero_faces = True
                    capture_now = False

            else:

                # print(faces)
                for face in faces:

                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

                    self.checkEyes(gray_image, face)

                    # obtendo pontos da imagem onde a face foi encontrada
                    left, top, right, bottom = (
                        int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))

                    # usuário selecionou TIRAR FOTO no disp
                    if capture_now:

                        # há somente uma face na imagem?
                        if faces_length == 1:

                            hog_zero_faces = False
                            hog_too_many_faces = False

                            # obtendo pontos faciais
                            facial_points = self.detector_points(gray_image, face)
                            # descritor facial composto por 128 descritores resultantes da analise do algoritmo
                            facial_descriptors = self.recognizer.compute_face_descriptor(gray_image, facial_points)

                            # convertendo descritor DLib para uma lista
                            facial_descriptors_list = [df for df in facial_descriptors]
                            ## crie arquivo que armazena os descritores faciais
                            # with open(last_descriptor, 'w') as filehandle:
                            #    filehandle.write(str(facial_descriptors_list))

                            last_descriptor = str(facial_descriptors_list)

                            # verificando se usuário já existe na base
                            self.recognize_this_face(facial_descriptors_list)

                            # salvando imagem inteira
                            cv2.imwrite(self.main_path + "last_capture.png", gray_image)
                            # gerando imagens mais escuras e seus respectivos descritores
                            # self.generate_gamma_images(gray_image)

                            print("Foto e descritores obtidos com sucesso.")


                        else:

                            hog_zero_faces = False
                            hog_too_many_faces = True

                        capture_now = False

                    cv2.rectangle(image, (left, top), (right, bottom), bgr, 2)

            # resizedFrame = cv2.resize(image, (self.res_width, self.res_height), interpolation=cv2.INTER_AREA)

            # encode OpenCV raw frame to jpg and displaying it
            ret, jpeg = cv2.imencode('.jpg', image)
            self.last_jpeg = jpeg

            return jpeg.tobytes()

            # else:
            #     if self.last_jpeg is not None:
            #         return self.last_jpeg.tobytes()
            #     else:
            #         return None

        except Exception as e:
            status = 2
            logger.error(str(e))
            logger.error(traceback.format_exc())
            return None

    def checkEyes(self, gray, face):

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = self.detector_points(gray, face)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[self.lStart:self.lEnd]
        rightEye = shape[self.rStart:self.rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        # leftEyeHull = cv2.convexHull(leftEye)
        # rightEyeHull = cv2.convexHull(rightEye)
        # cv2.drawContours(frame, [leftEyeHull], -1, (255, 255, 255), 1)
        # cv2.drawContours(frame, [rightEyeHull], -1, (255, 255, 255), 1)

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

            # reset the eye frame counter
            self.COUNTER = 0

        # draw the total number of blinks on the frame along with
        # the computed eye aspect ratio for the frame
        # cv2.putText(frame, str(self.TOTAL), (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        # cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def eye_aspect_ratio(eye):
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
