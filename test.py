import os, cv2, numpy as np

CODEC_MAP = {
    'mp4': 'mp4v', 'avi': 'XVID', 'wmv': 'WMV2',
    }

# CODEC_MAP = {
#     'mp4': 'avc1', 'avi': 'X264', 'wmv': 'WMV2',
#     }

CODEC_MAP = {
    'mp4': 'X264', 'avi': 'XVID', 'wmv': 'WMV2',
    }

for _ in range(80):
    # ret, frame = cap.read()
    # if not ret: break
    h,w = 112,160
    # h,w = 480,640
    # h,w = 720,1280
    # h,w = 720,1280
    # h,w = 1080,1920
    # h,w = 1440,2560
    # h,w = 2160,3840
    # h,w = 4320,7680
    frame = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

    if 'outvid' not in locals():
        outvid_file = 'output_cv2.mp4'
        # outvid_file = 'output.avi'
        # Define the codec and create VideoWriter object
        # 'XVID' is a popular codec, you can also use other codecs like 'MJPG', 'X264', etc.
        file_ext = os.path.splitext(outvid_file)[1][1:].lower()
        codec = CODEC_MAP.get(file_ext,'XVID') # Default to 'XVID' if extension is not in the map
        fourcc = cv2.VideoWriter_fourcc(*codec)
        # Specify the output file name, codec, frames per second (fps), and frame size
        h,w,c = frame.shape
        outvid = cv2.VideoWriter(outvid_file, fourcc, 20.0, (w, h))   

    # Write the frame to the video
    outvid.write(frame)

    # Display the frame (optional)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if 'cap' in locals():
    cap.release()
outvid.release()
del outvid
cv2.destroyAllWindows()