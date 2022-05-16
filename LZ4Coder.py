import lz4.frame
import cv2
import sys
import numpy as np
from time import time

class LZ4Coder:

    def compress_frame(self, frame, comp_rate=0, n_threads=1):
        
        time_comp_start = time()

        # Converting BGR to YUV colorspace
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        byte_frame = bytes(frame)

        # ---------------- COMPRESSION ----------------
        # in, [-100 - 22], [1, (12)]
        comp_bytes = lz4.frame.compress(byte_frame,
                                        compression_level=comp_rate,
                                        #block_size=lz4.frame.BLOCKSIZE_MAX4MB
                                        )
        time_comp = time()-time_comp_start
        # ---------------------------------------------------------------------
        return comp_bytes, time_comp


    # Decompresses the compressed frame data to retrieve the original frame
    def decompress_bytes(self, compressed_data_bytes, frame_shape, data_type):

        # --------------- DECOMPRESSION ---------------
        time_decomp_start = time()
        retr_data = lz4.frame.decompress(compressed_data_bytes)

        decoded = np.frombuffer(retr_data, dtype=data_type)
        # ----------------------------------------------

        decoded_frame = decoded.reshape((frame_shape))
        decoded_frame = cv2.cvtColor(decoded_frame, cv2.COLOR_YUV2BGR)

        time_decomp= time()-time_decomp_start

        # Return the retrieved image, elapsed time for decompression and size of the uncompressed data
        return(decoded_frame, time_decomp, sys.getsizeof(retr_data))
        
    # Returns dispay image with elapsed decompression time and compressed data information
    def display_results(self, comp_bytes, retr_frame, time_comp, time_decomp, size_retr_data):

        # --------------- WRITE ELAPSED TIMES ON DISPLAY IMAGE ---------------
        
        retr_img_gui = cv2.putText(retr_frame, 
                                    "Compression time: " + str(round(time_comp*1000, 2))+ "ms", 
                                    org=(10,30),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=1,
                                    color=(0,255,0),
                                    thickness=2)

        retr_img_gui = cv2.putText(retr_frame, 
                                    "Decompression time: " + str(round(time_decomp*1000, 2))+ "ms", 
                                    org=(10,60),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=1,
                                    color=(0,255,0),
                                    thickness=2)
        # ---------------------------------------------------------------------

        # --------------- WRITE DATA SIZE ON DISPLAY IMAGE ---------------

        retr_img_gui = cv2.putText(retr_frame, 
                                    "Compressed data size: " + str(round(sys.getsizeof(comp_bytes)/1e6, 2))+ "MB", 
                                    org=(10,90),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=1,
                                    color=(255,255,0),
                                    thickness=2)

        retr_img_gui = cv2.putText(retr_frame, 
                                    "Retrieved data size: " + str(round(size_retr_data/1e6, 2))+ "MB", 
                                    org=(10,120),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=1,
                                    color=(255,255,0),
                                    thickness=2)

        # ---------------------------------------------------------------------
        return retr_img_gui