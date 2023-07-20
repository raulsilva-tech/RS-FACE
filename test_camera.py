import platform

import cv2
import logging
from logging.handlers import RotatingFileHandler
import traceback


# estrutura para registro de Logs e Erros que podem ocorrer na execução do script
logger = logging.getLogger("FACE RECOGNITION LOG")
logger.setLevel(logging.ERROR)
handler = RotatingFileHandler("FR_Log.txt", maxBytes=10000, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def test(camera_id):
    try:
        # Legenda Id Retornado
        # -3 : Camera não disponível (deste o inicio da execução)
        # -2 : Camera desconectada durante a execução
        # -1 : Erro na execução (exceção disparada)
        #  1 : Sucesso
        test_id = 1
        message = "Sucesso"

        # instanciando variavel de acesso à webcam

        os_name = platform.system().lower()
        print(os_name)
        if "win" in os_name:
            capture = cv2.VideoCapture(int(camera_id), cv2.CAP_DSHOW)
        else:  # se linux:
            capture = cv2.VideoCapture(int(camera_id))

        if capture.isOpened():

            connected, image = capture.read()

            if not connected:
                test_id = -2
                message = "Não foi possível obter o frame da câmera {} . Verifique se a mesma foi/está desconectada.".format(
                        camera_id)
                logger.error(message)


        else:
            test_id = -3
            message = "O streaming da câmera {} não está aberto. Verifique se a mesma está conectada.".format(camera_id)
            logger.error(message)


        capture.release()

        return {"code": test_id, "msg": message}

    except Exception as e:

        logger.error(str(e))
        logger.error(traceback.format_exc())
        return {"code": -1, "msg": "Exceção disparada na execução do teste de câmera. Cheque o log do sistema de Reconhecimento Facial."}


def watch(camera, width, height):

    try:
        # Legenda Id Retornado
        # -3 : Camera não disponível (deste o inicio da execução)
        # -2 : Camera desconectada durante a execução
        # -1 : Erro na execução (exceção disparada)
        #  1 : Sucesso
        test_id = 1

        # instanciando variavel de acesso à webcam
        os_name = platform.system().lower()
        print(os_name)
        # se windows..
        if "win" in os_name:
            capture = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
        else: # se linux:
            capture = cv2.VideoCapture(camera)

        # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # nome da janela que será aberta para exibir a msg
        windowName = "window"
        cv2.namedWindow(windowName)
        # definindo fullscreen a fim de que ela fique na posição x: 0 e y: 0 e sem
        cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while capture.isOpened():

            connected, image = capture.read()

            if not connected:
                test_id = -2
                logger.error(
                    "Não foi possível obter o frame da câmera {} . Verifique se a mesma foi/está desconectada.".format(
                        camera))
                break
            else:
                resizedFrame = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                cv2.imshow(windowName, resizedFrame)

                # USUÁRIO CANCELOU VISUALIZAÇÃO
                if cv2.waitKey(1) == ord("q"):
                    break

        else:
            test_id = -3
            logger.error(
                "O streaming da câmera {} não está aberto. Verifique se a mesma está conectada.".format(camera))

        capture.release()
        cv2.destroyAllWindows()

        return test_id

    except Exception as e:

        logger.error(str(e))
        logger.error(traceback.format_exc())
        return -1

