import time

from flask import Flask, request, Response, send_file

import logging
from logging.handlers import RotatingFileHandler
import traceback

from test_camera import test
from add_descriptor import add, deactivate, reset_list
from detect_stream import DetectionCamera, set_capture_now, get_last_descriptor, \
    set_status_detection, get_status_detection, get_last_descriptor_gamma
from recognize_stream import RecognitionCamera, set_status_recognition, get_last_user_id, \
    get_status_recognition, check_dependencies, get_loading_status

from waitress import serve

# iniciando a api rest
app = Flask(__name__)

files_dir = "files/"

logger = logging.getLogger("FACE RECOGNITION LOG")
logger.setLevel(logging.ERROR)
handler = RotatingFileHandler("FR_Log.txt", maxBytes=100000, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# INICIANDO VARIAVEIS GLOBAIS COM SEUS VALORES PADRÕES
rec_camera = None
g_camera_id = 0
g_width = 640
g_height = 480
g_upsample_rate = 0
g_rec_frame_rate = 20
g_threshold = 0.4
g_finish_when_found = 1
g_eye_threshold = 0.21
g_eye_consec_frames = 2


@app.route('/get_image/<string:file_name>')
def app_get_image(file_name):
    full_path = files_dir + file_name
    return send_file(full_path, mimetype='image/png')

@app.route('/check_dependencies')
def app_check_dependencies():
    return check_dependencies(files_dir)

@app.route('/capture_now')
def app_capture():
    result = set_capture_now()
    return result, 200

@app.route('/stop_detection')
def app_stop_detection():
    set_status_detection(2)
    # global rec_camera
    # rec_camera = None
    return {"code": 1, "msg": ""}, 200


@app.route('/stop_recognition')
def app_stop_recognition():
    set_status_recognition(2)
    # global rec_camera
    # rec_camera = None
    return {"code": 1, "msg": ""}, 200

@app.route('/update_faces')
def app_update_faces():
    if rec_camera is not None:
        rec_camera.update_trained_faces()
        return {"code": 1, "msg": ""}, 200
    else:
        return {"code": 0, "msg": "Instancia ainda não iniciada."}, 200


@app.route("/get_last_descriptor")
def app_get_last_descriptor():
    return get_last_descriptor()


@app.route("/get_last_descriptor_gamma/<int:number>")
def app_get_last_descriptor_gamma(number):
    return get_last_descriptor_gamma(number)


@app.route("/test/<int:camera_id>")
def app_test_camera(camera_id):
    if rec_camera is None:
        result_json = test(camera_id)
        return result_json, 200
    else:
        return {"code": 1, "msg": "Instância em execução."}, 200


@app.route("/add_descriptor", methods=['POST'])
def app_add_descriptor():
    descriptors_list = request.json
    return add(descriptors_list, files_dir), 200
    # return {"code": result, "msg": ""}, 200

@app.route("/deactivate_descriptor", methods=['POST'])
def app_deactivate_descriptor():
    descriptors_list = request.json
    return deactivate(descriptors_list, files_dir), 200

def gen_detection(camera):
    set_status_detection(1)

    while get_status_detection() < 2:

        # get camera frame
        frame = camera.get_frame()

        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type:image/jpeg\r\n'
                   b'Content-Length: ' + f"{len(frame)}".encode() + b'\r\n'
                                                                    b'\r\n' + frame + b'\r\n')


@app.route('/video_detection/<int:camera_id>/<int:width>/<int:height>/<int:upsample_rate>/<int:detect_frame_rate>/<float:threshold>/<float:eye_threshold>/<int:eye_consec_frames>')
def app_detect(camera_id, width, height, upsample_rate, detect_frame_rate, threshold, eye_threshold,eye_consec_frames):
    global rec_camera
    rec_camera = None

    # parando reconhecimento por caso tiver algum em exceução
    if get_status_recognition() == 1:
        set_status_recognition(2)
        time.sleep(2)

    try:
        # obtendo status da detecção
        status = get_status_detection()

        # 1 = em execução, então set para 2 a fim de parar a ultima execução
        if status == 1:
            set_status_detection(2)
            time.sleep(2)

        camera = DetectionCamera(camera_id, width, height, upsample_rate, files_dir, detect_frame_rate,threshold,eye_threshold,eye_consec_frames)
        return Response(gen_detection(camera),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    except Exception as e:

        logger.error(str(e))
        logger.error(traceback.format_exc())


def gen_recognition(camera):
    set_status_recognition(1)

    while get_status_recognition() < 2:

        # print("to rodando")
        # get camera frame
        frame = camera.get_frame()

        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type:image/jpeg\r\n'
                   b'Content-Length: ' + f"{len(frame)}".encode() + b'\r\n'
                                                                    b'\r\n' + frame + b'\r\n')



@app.route('/watch_recognition')
def app_watch_recognition():

    global rec_camera
    global g_camera_id
    global g_width
    global g_height
    global g_upsample_rate
    global g_rec_frame_rate
    global g_threshold
    global g_finish_when_found
    global g_eye_threshold
    global g_eye_consec_frames
    global files_dir

    if get_loading_status() != "B":
        if rec_camera is None:
            print("rec_camera vazio")
            rec_camera = RecognitionCamera(g_camera_id, g_width, g_height, g_upsample_rate, files_dir, g_threshold, g_rec_frame_rate,
                                           g_finish_when_found, g_eye_threshold,
                                           g_eye_consec_frames)

    count = 1
    while (get_loading_status() == "B" and count <=10):
        time.sleep(1)
        print("Waiting rec_camera to be loaded: {}s".format(count))
        count+=1

    if count == 11:
        return {"code": -1, "msg": "Erro no carregamento de RecognitionCamera."}, 200


        # parando detecção por caso tiver algum em execução
    if get_status_detection() == 1:
        set_status_detection(2)
        time.sleep(2)

    try:

        return Response(gen_recognition(rec_camera),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    except Exception as e:

        logger.error(str(e))
        logger.error(traceback.format_exc())

@app.route(
    '/load_recognition/<int:camera_id>/<int:width>/<int:height>/<int:upsample_rate>/<int:rec_frame_rate>/<float:threshold>/<int:finish_when_found>/<float:eye_threshold>/<int:eye_consec_frames>')  # /<int:window_width>/<int:window_height>')
def app_load_recognition(camera_id, width, height, upsample_rate, rec_frame_rate, threshold, finish_when_found, eye_threshold,
                  eye_consec_frames):  # , window_width, window_height):


    global g_camera_id
    global g_width
    global g_height
    global g_upsample_rate
    global g_rec_frame_rate
    global g_threshold
    global g_finish_when_found
    global g_eye_threshold
    global g_eye_consec_frames
    global files_dir

    g_camera_id = camera_id
    g_width = width
    g_height = height
    g_upsample_rate = upsample_rate
    g_rec_frame_rate = rec_frame_rate
    g_threshold = threshold
    g_finish_when_found = finish_when_found
    g_eye_threshold = eye_threshold
    g_eye_consec_frames = eye_consec_frames

    try:
        global rec_camera
        # rec_camera = None
        if rec_camera is None:
            rec_camera = RecognitionCamera(camera_id, width, height, upsample_rate, files_dir, threshold, rec_frame_rate,
                                           finish_when_found, eye_threshold,
                                           eye_consec_frames)  # , window_width,window_height)),
        else:
            rec_camera.load_recognition(camera_id, width, height, upsample_rate, files_dir, threshold, rec_frame_rate,
                                           finish_when_found, eye_threshold,
                                           eye_consec_frames)
        return {"code": 1, "msg": ""}, 200
    except Exception as e:

        logger.error(str(e))
        logger.error(traceback.format_exc())
        return {"code": 0, "msg": str(e)}, 200


@app.route('/get_last_user_id')
def app_get_last_user_id():
    return {"code": get_last_user_id(), "msg": ""}, 200

@app.route('/reset_list')
def app_reset_list():
    result = reset_list(files_dir)
    return result, 200


# iniciando em modo debug
if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000)

# serve(app, host='0.0.0.0', port=5000) #WAITRESS!
