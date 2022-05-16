#!/usr/bin/env python3
import numpy as np
import cv2
import pika
import time
import pickle
from argparse import ArgumentParser

from ZSTDCoder import ZSTDCoder
from LZ4Coder import LZ4Coder

# Construct the argument parse and parse the arguments
ap = ArgumentParser()
ap.add_argument("-i", "--input", type=int, default=0, required=True,
	help="Port number of camera (0 for built-in webcam)")
ap.add_argument("-d", "--display", action='store_true', required=False,
	help="Display option")
ap.add_argument("-c", "--compressor", type=str, default="0", required=True,
	help="Type of compression algorithm (0 for ZSTD, 1 for LZ4)")
ap.add_argument("-r", "--ratecmp", type=int, default=-3, required=False,
	help="Compression rate (btw. -100 and 22)")
ap.add_argument("-t", "--threads", type=int, default=4, required=False,
	help="Number of threads")

args = vars(ap.parse_args())

#HOST_NAME = '192.168.8.114'
HOST_NAME = 'localhost'
HEARTBEAT_VAL = 0
QUEUE_ARG_NUMBER = 10
RGB_QUEUE = 'RGBQueue'

#credentials = pika.PlainCredentials(username='Taric', password='TaricPass')

parameters = pika.ConnectionParameters(host=HOST_NAME,heartbeat=HEARTBEAT_VAL)# port=5672, virtual_host='/', credentials=credentials, )

rgb_connection = pika.BlockingConnection(parameters=parameters)
        
rgb_channel = rgb_connection.channel()

rgb_channel.queue_declare(queue=RGB_QUEUE, arguments={"x-max-length": QUEUE_ARG_NUMBER})
    
cap = cv2.VideoCapture(args["input"])
time.sleep(3)

# Choosing which compression algorithm will be in use
myCoder_comps = {"0": ZSTDCoder(), "1": LZ4Coder()}
myCoder = myCoder_comps[args["compressor"]]

while(cap.isOpened()):

    ret, frame = cap.read()
    # Compress frames and get elapsed time for it
    comp_bytes, time_comp = myCoder.compress_frame(frame, comp_rate=args["ratecmp"], n_threads=args["threads"])
    
    # Pack frame shape as bytestream w/ pickle
    frame_shape = [frame.shape[0], frame.shape[1], frame.shape[2]]
    pub_msg =  [frame_shape, time_comp, comp_bytes]
    pub_msg_dumps = pickle.dumps(pub_msg)

    # Publish the message
    rgb_channel.basic_publish(exchange="", routing_key=RGB_QUEUE, body=pub_msg_dumps)

    if args["display"]:
        cv2.imshow("Rasp RGB Stream", frame)
        
        if cv2.waitKey(1) == 27:
            break
    
        
cap.release()

cv2.destroyAllWindows()

