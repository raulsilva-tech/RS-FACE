import json
import os
import _pickle as cPickle
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
import traceback
from datetime import datetime
import time

# estrutura para registro de Logs e Erros que podem ocorrer na execução do script
logger = logging.getLogger("FACE RECOGNITION LOG")
logger.setLevel(logging.ERROR)
handler = RotatingFileHandler("FR_Log.txt", maxBytes=100000, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

lastMs = int(round(time.time() * 1000))


def showDiff(msg):
    global lastMs
    currentMs = int(round(time.time() * 1000))
    diff = currentMs - lastMs
    print(datetime.utcnow().strftime('%H:%M:%S.%f')[:-3] + " : " + msg + " " + str(diff) + " ms")

    lastMs = currentMs


def add(user_ufd_list, files_dir):
    try:

        train_file_name = "DK_trainedFacialDescriptors.npy"
        indexes_file_name = "DK_indexes.pickle"

        # checando existencia do arquivo de treinamento
        training_path = files_dir + train_file_name
        if os.path.exists(training_path):
            # obtendo descritores incluídos nele
            processed_faces = np.load(training_path, allow_pickle=True)
        else:
            processed_faces = None

        # checando existencia do arquivo indexador
        indexes_path = files_dir + indexes_file_name
        if os.path.exists(indexes_path):
            # obtendo indexadores incluidos nele
            with open(indexes_path, 'rb') as filehandle:
                # read the data as binary data stream
                indexes = cPickle.load(filehandle)
                idx = len(indexes)
        else:
            indexes = {}
            idx = 0

        # # se lista não estiver vazia, verifique os indexes setados nos UFDs do usuário e zere-os
        # if processed_faces is not None:
        #
        #     print("Tamanho: {}".format(len(user_ufd_list)))
        #     # percorrendo cada descritor do usuário para atualiza-lo no arquivo de treinamento npy
        #     for ufd in user_ufd_list:
        #         ufd_index = ufd.get("index")
        #
        #         #valorando o descritor "antigo" com um valor alto a fim de que nunca seja retornado no reocnhecimento
        #         processed_faces[ufd_index] = [100] * 128 # 128 = num obrigatorio de itens no array
        #         indexes[ufd_index] = None

        # percorrendo UFDs do usuário para adiciona-los no final do array
        for ufd in user_ufd_list:

            image_name = ufd.get("name")
            #obtendo primeiro nome do usuário excluindo caracteres especiais
            #image_name = ''.join(e for e in ufd.get("name") if e.isalnum())

            facial_descriptors_list = json.loads(ufd.get("descriptors"))

            # converter lista em um array numpy
            np_array_facial_descriptors = np.asarray(facial_descriptors_list, dtype=np.float64)
            # adicionando uma nova "coluna"/"dimensão no array (que agora é um vetor)
            np_array_facial_descriptors = np_array_facial_descriptors[np.newaxis, :]

            # adicionando descritores faciais da face atual à lista externa que recebe todos os descritores
            if processed_faces is None:
                processed_faces = np_array_facial_descriptors
            else:
                processed_faces = np.concatenate((processed_faces, np_array_facial_descriptors), axis=0)

            indexes[idx] = image_name
            ufd["index"] = idx
            idx += 1

        # print(indexes)

        # print(np.argmax(processed_faces))

        # for p in processed_faces:
        #     print(p)
        #
        # for user in user_ufd_list:
        #     print(user)

        np.save(training_path, processed_faces)
        with open(indexes_path, 'wb') as f:
            cPickle.dump(indexes, f)

        # user_descriptor_path = users_dir + "UFD_"+str(user_id)+".json"
        # # atualizando UFD com os ids indexados no arquivo de treinamento
        # new_ufd_file = open(user_descriptor_path, 'w')
        # new_ufd_file.write(json.dumps(user_ufd_list))
        # new_ufd_file.close()

        # indicando que a execução foi realizada com sucesso
        return json.dumps(user_ufd_list)

    except Exception as e:
        logger.error(str(e))
        logger.error(traceback.format_exc())
        return {"code": -1,"msg": str(e)}


def deactivate(user_ufd_list, files_dir):

    try:

        train_file_name = "DK_trainedFacialDescriptors.npy"
        indexes_file_name = "DK_indexes.pickle"

        # checando existencia do arquivo de treinamento
        training_path = files_dir + train_file_name
        if os.path.exists(training_path):
            # obtendo descritores incluídos nele
            processed_faces = np.load(training_path, allow_pickle=True)
        else:
            processed_faces = None

        # checando existencia do arquivo indexador
        indexes_path = files_dir + indexes_file_name
        if os.path.exists(indexes_path):
            # obtendo indexadores incluidos nele
            with open(indexes_path, 'rb') as filehandle:
                # read the data as binary data stream
                indexes = cPickle.load(filehandle)
                idx = len(indexes)
        else:
            indexes = {}
            idx = 0
        """"
        # # se lista não estiver vazia, verifique os indexes setados nos UFDs do usuário e zere-os
        if processed_faces is not None:

            # print("Tamanho: {}".format(len(user_ufd_list)))
            # percorrendo cada descritor do usuário para atualiza-lo no arquivo de treinamento npy
            for ufd in user_ufd_list:
                ufd_index = ufd.get("index")
                #elemento existe?
                if len(processed_faces) >= ufd_index:
                    # valorando o descritor "antigo" com um valor alto a fim de que nunca seja retornado no reocnhecimento
                    processed_faces[ufd_index] = [100] * 128  # 128 = num obrigatorio de itens no array
                    indexes[ufd_index] = None
        """
        if processed_faces is not None and len(indexes) > 0:
            # for p in processed_faces:
            #     print(p)
            np.save(training_path, processed_faces)
            with open(indexes_path, 'wb') as f:
                cPickle.dump(indexes, f)

            # indicando que a execução foi realizada com sucesso
            return json.dumps(user_ufd_list)

        else:
            # retorna NF quando os arquivos de treinamento ainda não existem
            return {"code": 0,"msg": "Arquivo de treinamento não existe"}

    except Exception as e:
        logger.error(str(e))
        logger.error(traceback.format_exc())
        return {"code": -1,"msg": str(e)}


def reset_list(files_dir):
    try:

        train_file_name = "DK_trainedFacialDescriptors.npy"
        indexes_file_name = "DK_indexes.pickle"

        message = "Sucesso"

        # checando existencia do arquivo de treinamento para renomea-lo
        training_path = files_dir + train_file_name
        if os.path.exists(training_path):
            if os.path.exists(training_path+".old"):
                os.remove(training_path+".old")

            os.rename(training_path,training_path+".old")

        else:
            message = " Arquivo não existe: "+ training_path

        # checando existencia do arquivo indexador para renomea-lo
        indexes_path = files_dir + indexes_file_name
        if os.path.exists(indexes_path):
            if os.path.exists(indexes_path+".old"):
                os.remove(indexes_path+".old")

            os.rename(indexes_path,indexes_path+".old")
        else:
            message += " | Arquivo não existe: " + indexes_path

        return {"code": 1, "msg": message}

    except Exception as e:
        logger.error(str(e))
        logger.error(traceback.format_exc())
        return {"code": -1,"msg": str(e)}