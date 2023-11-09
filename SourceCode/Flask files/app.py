#Necessary imports
from flask import Flask, render_template, request, send_from_directory, send_file, Response
import os
from urllib.parse import urlparse
from ultralytics import YOLO
import cv2                                        
import time                                            
import math
import winsound

app = Flask(__name__)

@app.route("/")
def index():
  return render_template("index.html")

@app.route("/aboutus.html")
def about_us():
  return render_template("aboutus.html")


@app.route('/webcam_func')
def video_feed():
  return webcam_func()

#alarm
def play_alarm():
    duration = 1000  
    freq = 440  
    winsound.Beep(freq, duration)

@app.route("/display/<path:filename>")
def display(filename):
    parsed_url = urlparse(request.url)
    url_path = parsed_url.path
    directory, filename = url_path.split("/display/")[1].rsplit("/", 1)
    input_path = os.path.join(directory, filename).replace("\\", "/")
    print(input_path)
    return send_file(input_path, as_attachment=False)

@app.route('/video/<path:filename>')
def serve_video(filename):
    video_path = filename 
    def generate_frames():
        cap = cv2.VideoCapture(video_path)
        while True:
            success, frame = cap.read()
            if not success:
                break
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1)
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
     
@app.route("/", methods=["GET", "POST"])
def application():
  input_path = None
  alarm_activated = False

  if request.method == "POST":
    
    if "file_name" in request.files:
      f = request.files["file_name"]
      basepath = os.path.dirname(__file__)
      filepath = os.path.join(basepath, "uploads", f.filename)
      print("Upload folder is ", filepath)
      file_extension = f.filename.rsplit(".", 1)[1]
      #Image detection Section
      if file_extension in ("jpg", "png"):
        img=cv2.imread(filepath)
        model=YOLO('best.pt')
        model.predict(source = img, save_txt=True,save=True)
        folder_path = 'runs/detect'
        subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
        latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
        input_path = folder_path+'/'+latest_subfolder+'/'+"image0.jpg" 
        #alarm if weapon detected.
        with open(os.path.join(folder_path, latest_subfolder, "labels/image0.txt"), "r") as f:
            for line in f:
                if line.strip():
                    alarm_activated = True
                    break
        if alarm_activated:
            print("Weapon detected")
            play_alarm()
        return render_template('index.html', input_path= input_path,video_format=None)
      
      #Video Section
      elif file_extension in ("mp4"):
            model=YOLO('best.pt')
            model.predict(source = filepath,save_txt=True,save=True)
            folder_path = 'runs/detect'
            subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
            latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x))) 
            input_path = folder_path+'/'+latest_subfolder+'/'+f.filename.replace(".mp4",".avi")

            #alarm if weapon detected.
            for filename in os.listdir(os.path.join(folder_path, latest_subfolder, "labels/")):
                if filename.endswith(".txt"):
                    filepath = os.path.join(folder_path, latest_subfolder, "labels/", filename)
                    with open(filepath, "r") as f:
                        # Check for weapon detection
                        for line in f:
                            if line.strip():
                                alarm_activated = True
                                break
                    # Play alarm if weapon detected
                    if alarm_activated:
                        play_alarm()
                        print("Weapon detected")
                        break  # Break out of the loop once alarm is activated
            if not alarm_activated:
                # No alarm was activated
                print("No weapons detected.")
            return render_template('index.html', input_path= input_path,video_format="mp4")

#WebCam section  
classNames = ['handgun','knife']
@app.route('/webcam_func')
def webcam_func():
  model=YOLO("best.pt")
  def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
      success, frame = cap.read()
      results = model(frame, stream=True)
      for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(frame, classNames[cls], org, font, fontScale, color, thickness)
            play_alarm()
      if not success:
        break
      ret, buffer = cv2.imencode('.jpg', frame)
      if not ret:
        continue
      frame_bytes = buffer.tobytes()
      yield (b'--frame\r\n'
             b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
      time.sleep(0.1)
  return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

   
@app.route('/webcam.html')
def webcam():
  return render_template('webcam.html')

if __name__ == "__main__":
  app.run(debug=True)
