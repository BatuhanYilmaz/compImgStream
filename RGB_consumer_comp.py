
#       RGB STREAM      #

import queue
import cv2
import pika
import sys
import threading
import numpy as np
import zstd
import time
import pickle
from argparse import ArgumentParser


from ZSTDCoder import ZSTDCoder
from LZ4Coder import LZ4Coder
#       MACRO DEFINITIONS       #

RGB_QUEUE = 'RGBQueue'
READ_QUEUE = queue.Queue(maxsize=10)

WINDOW_TITLE = 'RGB Stream Consumer'
HOST_NAME = 'localhost'
# HOST_NAME = '167.99.246.227'
# HOST_NAME = '192.168.1.61'

HEARTBEAT_VAL = 0
QUEUE_ARG_NUMBER = 10
STOP_FLAG = False
FACE_DETECTION_FLAG = False
MESSAGE_COUNTER = 0
MESSAGE_COUNTER_PREV = 0

# Construct the argument parse and parse the arguments
ap = ArgumentParser()

ap.add_argument("-c", "--compressor", type=str, default="0", required=True,
	help="Type of compression algorithm (0 for ZSTD, 1 for LZ4)")
ap.add_argument("-r", "--ratecmp", type=int, default=-3, required=False,
	help="Compression rate (ZSTD: btw. -100 and 22) (LZ4: btw. 0 and 16)")
ap.add_argument("-t", "--threads", type=int, default=4, required=False,
	help="Number of threads")

args = vars(ap.parse_args())

# Choosing which compression algorithm will be in use
myCoder_comps = {"0": ZSTDCoder(), "1": LZ4Coder()}
myCoder = myCoder_comps[args["compressor"]]

#       PARAMETER CHECK      #

# def streamthreadfunc():

#     cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)

#     cv2.imshow(WINDOW_TITLE, rgb_color_image)
#     key = cv2.waitKey(1)
#     if key == 27:
#         pipeline.stop()
#         cv2.destroyAllWindows()
#         STOP_FLAG = True
#         exit(1)


def master_callback(ch, method, properties, body):
    global MESSAGE_COUNTER
    MESSAGE_COUNTER += 1
    rgb_color_bytes = np.frombuffer(body, dtype=np.uint8)
    READ_QUEUE.put(item=rgb_color_bytes, block=True)

def master_listener():
    global STOP_FLAG
    try:
        # credentials = pika.PlainCredentials('TaricRasp', 'TaricRaspPass')
        #credentials = pika.PlainCredentials('Taric', 'TaricPass')

        parameters = pika.ConnectionParameters(host=HOST_NAME,
                                               #port=5672,
                                               #virtual_host='/',
                                               heartbeat=HEARTBEAT_VAL,
                                               #credentials=credentials,
                                               )

        master_connection = pika.BlockingConnection(parameters)

        master_channel = master_connection.channel()
        master_channel.queue_declare(queue=RGB_QUEUE, arguments={"x-max-length": QUEUE_ARG_NUMBER})
        master_channel.queue_purge(queue=RGB_QUEUE)
        master_channel.basic_consume(queue=RGB_QUEUE, on_message_callback=master_callback, auto_ack=True)
        master_channel.start_consuming()

    except KeyboardInterrupt:
        print('Interrupted')
        STOP_FLAG = True
        exit(1)


def rgb_data_read_from_python_queue():


    while True:
        recv_msg = READ_QUEUE.get(block=True)

        # Unpack the consumed package: recv_msg_loads = [frame_shape, time_comp, comp_bytes] 
        recv_msg_loads = pickle.loads(recv_msg)
        
        # --------------------------- DECOMPRESSION ---------------------------
        rgb_data_reshaped, time_decomp, size_retr_data = myCoder.decompress_bytes(recv_msg_loads[2],
                                                                                    frame_shape=recv_msg_loads[0],
                                                                                    data_type=np.uint8)
        # ---------------------------------------------------------------------

        #rgb_data_reshaped = np.reshape(rgb_frame, RX_FRAME_SHAPE)

        # --------------- WRITE ELAPSED TIMES ON DISPLAY IMAGE ----------------
        retr_img_display = myCoder.display_results(recv_msg_loads[2], 
                                                    rgb_data_reshaped,
                                                    time_comp=recv_msg_loads[1],
                                                    time_decomp=time_decomp,
                                                    size_retr_data=size_retr_data)
        # ---------------------------------------------------------------------

        # ---------------------------- SHOW IMAGE -----------------------------
        cv2.imshow(WINDOW_TITLE, retr_img_display)
        # ---------------------------------------------------------------------

        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            exit(1)

try:
    print("*** RGB STREAM ***")
    threading.Thread(target=rgb_data_read_from_python_queue, daemon=True).start()
    threading.Thread(target=master_listener, daemon=True).start()
    TIME_PREV = time.time()

    while True:
        print(round(((MESSAGE_COUNTER - MESSAGE_COUNTER_PREV) / (time.time() - TIME_PREV)), 3))
        MESSAGE_COUNTER_PREV = MESSAGE_COUNTER
        TIME_PREV = time.time()
        if STOP_FLAG:
            exit(1)
        time.sleep(1)

except RuntimeError as err:
    print("Hata Runtime : ", err)
    STOP_FLAG = True
    exit(1)

except KeyboardInterrupt:
    print('Interrupted')
    STOP_FLAG = True
    exit(1)

except Exception as err:
    print("Hata Exception : ", err)
    STOP_FLAG = True
    exit(1)

