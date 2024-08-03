import datetime
import time
import cv2, json,sys, math
# sys.path.insert(0,'D:\Projects\AISoftArt\CoVi\Object_Detection\yolov8-face')
for nm in ['show','Select ROI']:
    cv2.namedWindow(nm, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(nm, 800, 600)  # Set your desired width and height

def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Get the current size of the image
    (h, w) = image.shape[:2]

    # If both width and height are None, return the original image
    if width is None and height is None:
        return image

    # Calculate aspect ratio
    aspect_ratio = w / float(h)

    # If width is provided, resize the image based on width while maintaining aspect ratio
    if width is not None:
        new_height = int(width / aspect_ratio)
        resized_image = cv2.resize(image, (width, new_height), interpolation=inter)
    
    # If height is provided, resize the image based on height while maintaining aspect ratio
    else:
        new_width = int(height * aspect_ratio)
        resized_image = cv2.resize(image, (new_width, height), interpolation=inter)

    return resized_image

from drowsy_detection import get_mediapipe_app, calculate_avg_ear, plot_eye_landmarks, plot_text
facemesh_model = get_mediapipe_app()
thresholds = {
    "EAR_THRESH": 0.18,
    "WAIT_TIME": 1.0,
}
eye_idxs = {
    "left": [362, 385, 387, 263, 373, 380],
    "right": [33, 160, 158, 133, 153, 144],
}
RED = (0, 0, 255)  # BGR
GREEN = (0, 255, 0)  # BGR




# Function to capture video and perform object detection with pose estimation
def detect_from_video(video_path, output_path):
    global results
    # Open the video capture
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Select ROI
    sucess, frame = cap.read()
    plot_text(frame, 'Select ROI', (10,30), 250, fs=1, frm_i=10)
    roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    roi_x, roi_y, roi_w, roi_h = roi  # Extract ROI coordinates
    cv2.destroyWindow("Select ROI")

    state_tracker = {
        "start_time": time.perf_counter(),
        "DROWSY_TIME": 0.0,  # Holds the amount of time passed with EAR < EAR_THRESH
        "COLOR": GREEN,
        "play_alarm": False,
    }
    frame_number = 0
    repeat=0
    out = None  # Initialize out as None
    while True:
        start_time = time.time()
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read a frame from the video
        sucess, frame = cap.read()
        if not sucess : 
            if repeat==0: break
            repeat+=1
            frame_number = 0
            continue
        # frame = resize_with_aspect_ratio(frame, width=640)

        if out is None:
            fps = cap.get(cv2.CAP_PROP_FPS)
            # frm_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            # frm_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frm_h, frm_w, _ = frame.shape

            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') #XVID
            out = cv2.VideoWriter(output_path, fourcc, fps, (frm_w, frm_h))

        EAR_txt_pos = (10, int(frm_h // 2 * 1.55))
        DROWSY_TIME_txt_pos = (10, int(frm_h // 2 * 1.7))
        ALM_txt_pos = (10, int(frm_h // 2 * 1.85))
        frame_ROI = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        frm_roi_h,frm_roi_w,_=frame_ROI.shape
        results = facemesh_model.process(frame_ROI)
        fs = 0.6
        if results.multi_face_landmarks:
            for rmf in results.multi_face_landmarks:
                landmarks = rmf.landmark
                EAR, coordinates = calculate_avg_ear(landmarks, eye_idxs["left"], eye_idxs["right"], 
                                                        frm_roi_w, frm_roi_h)
                # print(f'{coordinates=}')
                coordinates = (
                    [(x + roi_x, y + roi_y) for (x, y) in coordinates[0]],  # Left eye
                    [(x + roi_x, y + roi_y) for (x, y) in coordinates[1]]   # Right eye
                )
                if math.sin(math.radians(frame_number*40))>0:
                    frame = plot_eye_landmarks(frame, coordinates[0], coordinates[1], state_tracker["COLOR"], fs=2)
                else:
                    frame = plot_eye_landmarks(frame, coordinates[0], coordinates[1], state_tracker["COLOR"], transp=1)

                if EAR < thresholds["EAR_THRESH"]:

                    # Increase DROWSY_TIME to track the time period with EAR less than the threshold
                    # and reset the start_time for the next iteration.
                    end_time = time.perf_counter()

                    state_tracker["DROWSY_TIME"] += end_time - state_tracker["start_time"]
                    state_tracker["start_time"] = end_time
                    state_tracker["COLOR"] = RED

                    if state_tracker["DROWSY_TIME"] >= thresholds["WAIT_TIME"]:
                        state_tracker["play_alarm"] = True
                        plot_text(frame, "WAKE UP! WAKE UP", ALM_txt_pos, state_tracker["COLOR"], fs=fs, frm_i=frame_number)

                else:
                    state_tracker["start_time"] = time.perf_counter()
                    state_tracker["DROWSY_TIME"] = 0.0
                    state_tracker["COLOR"] = GREEN
                    state_tracker["play_alarm"] = False

                EAR_txt = f"EAR: {round(EAR, 2)}"
                DROWSY_TIME_txt = f"Drowsy: {round(state_tracker['DROWSY_TIME'], 2)} Secs"
                plot_text(frame, EAR_txt, EAR_txt_pos, state_tracker["COLOR"], fs=fs, frm_i=frame_number+5)
                plot_text(frame, DROWSY_TIME_txt, DROWSY_TIME_txt_pos, state_tracker["COLOR"], fs=fs,frm_i=frame_number+10)

        else:
            state_tracker["start_time"] = time.perf_counter()
            state_tracker["DROWSY_TIME"] = 0.0
            state_tracker["COLOR"] = GREEN
            state_tracker["play_alarm"] = False

            # Flip the frame horizontally for a selfie-view display.
            # frame = cv2.flip(frame, 1)



        # End time for object detection
        end_time = time.time()

        # Calculate time taken for object detection
        detection_time = end_time - start_time

        # Calculate fps
        fps = 1 / detection_time
        # frame_step = 10 # fast for play video
        # frame_step = 2 # fast for play video
        frame_step = 1
        frame_number += frame_step        
        fps_status = f"Frame step: {frame_step}, FPS: {int(fps)}"
        
        # cv2.putText(frame, fps_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fs, 0, 5)
        # cv2.putText(frame, fps_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fs, (175, 175, 175), 2)
        plot_text(frame, fps_status, (10, 30), (175, 175, 175), fs=fs, frm_i=frame_number+10)
        # Display the frame with detections
        out.write(frame)
        cv2.imshow('show', frame)

        # Exit loop (press 'q' key)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # Release capture and close windows
    cap.release()
    if out is not None:  # Release out only if it's not None
        out.release()    
    cv2.destroyAllWindows()

# Replace 'path/to/your/video.mp4' with your video path
video_path = rf"D:\Projects\AISoftArt\Datasets\Shopping-WvhYuDvH17I.mp4"
vp=r"C:\Users\osama\Downloads\Video\GFF.mp4"
vp='https://www.youtube.com/watch?v=Z96lHQg0CLc'
vp=r"C:\Users\osama\Downloads\Video\corrosion.mp4"
vp=r"C:\Users\osama\Downloads\3.jpg"
vp=r"C:\Users\osama\Downloads\Video\A Worker Transferring Goods In A Forklift · Free Stock Video.mp4"
vp=r"C:\Users\osama\Downloads\The_driver_appears_drowsy_or_i (1).mp4"
vp=r"C:\Users\osama\Downloads\02-driver-driver-drowsiness-detection-Driver-drowsing.gif"
vp=r"C:\Users\osama\Downloads\giphy.gif"
vp=r"C:\Users\osama\Downloads\giphy-ezgif.com-gif-maker.mp4"
vp=r"C:\Users\osama\Downloads\The_driver_appears_drowsy_or_i (1).mp4"
vp=r"C:\Users\osama\Downloads\drowsy.mp4"
current_datetime = datetime.datetime.now()
date_time_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
output_path = date_time_str+'.mp4'
detect_from_video(vp, output_path)