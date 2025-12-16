import cv2
import mediapipe as mp
import numpy as np
import os
import time
import pickle
import json
import random
from datetime import datetime
from collections import defaultdict, deque
from flask import Flask, render_template, Response, request, jsonify

# ------------------------------
# FLASK CONFIG
# ------------------------------
app = Flask(__name__)

# ------------------------------
# CONFIG & MATH CONSTANTS
# ------------------------------
KNOWN_DIR = "known"
SECURITY_LOG_DIR = os.path.join("data", "security_log")
DB_PATH = os.path.join("data", "faces_db.pkl")
LOG_PATH = os.path.join("data", "events_log.csv")

# Global Tuning Variables (Mutable)
tuning = {
    "HAMMING_DISTANCE": 75,
    "SMOOTH_FRAMES": 8,
    "INTRUDER_ALERT": True
}


THRESHOLD_MATCHES = 12       
BLINK_THRESHOLD = 0.22
CAPTURE_EVERY_N_FRAMES = 2
RECOGNITION_INTERVAL = 4     
JPEG_QUALITY = 70            

# Global State
state = {
    "enroll_mode": False,
    "is_recording": False,
    "enroll_name": "",
    "frame_counter": 0,
    "session_samples": 0,
    "recent_names": deque(maxlen=tuning["SMOOTH_FRAMES"]),
    "camera_active": False,
    # Security State
    "intruder_timer": 0,
    "liveness_state": "WAITING", # WAITING, CHALLENGE, VERIFIED
    "liveness_target": "",       # BLINK, SMILE
    "liveness_timer": 0,
    "latest_alert": None         # Queue for UI messages
}


# ------------------------------
# 1. MEDIAPIPE SETUP
# ------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ------------------------------
# 2. UTILITIES (ORB & DB)
# ------------------------------
orb = cv2.ORB_create(nfeatures=1500) 
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def ensure_dirs():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    os.makedirs(KNOWN_DIR, exist_ok=True)
    os.makedirs(SECURITY_LOG_DIR, exist_ok=True)

def load_db():
    if not os.path.exists(DB_PATH): return {}
    try:
        with open(DB_PATH, "rb") as f: return pickle.load(f)
    except: return {}

def save_db(db):
    try:
        with open(DB_PATH, "wb") as f: pickle.dump(db, f)
    except Exception as e:
        print(f"[!] Error saving DB: {e}")

ensure_dirs()
db = load_db()
print(f"[i] Database loaded: {list(db.keys())}")

def compute_orb(img_bgr):
    if img_bgr is None or img_bgr.size == 0: return None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    kps, des = orb.detectAndCompute(gray, None)
    return des

def match_descriptors(des_query, db):
    if des_query is None: return "Unknown", 0
    best_name, best_score = "Unknown", 0
    
    try:
        db_items = list(db.items())
    except RuntimeError:
        return "Unknown", 0

    for name, saved_des_list in db_items:
        max_matches_for_person = 0
        for saved_des in saved_des_list:
            if saved_des is None: continue
            matches = bf.match(des_query, saved_des)
            # Use mutable global tuning param
            good_matches = [m for m in matches if m.distance < tuning["HAMMING_DISTANCE"]]
            count = len(good_matches)
            if count > max_matches_for_person:
                max_matches_for_person = count
        
        if max_matches_for_person > best_score:
            best_score = max_matches_for_person
            best_name = name
                
    return best_name, best_score

# ------------------------------
# 3. MATH HELPERS
# ------------------------------
def calculate_ear(landmarks, indices, w, h):
    try:
        coords = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in indices]
        v1 = np.linalg.norm(coords[1] - coords[5])
        v2 = np.linalg.norm(coords[2] - coords[4])
        hor = np.linalg.norm(coords[0] - coords[3])
        return (v1 + v2) / (2.0 * hor + 1e-6)
    except: return 0.3

def detect_emotion(landmarks):
    try:
        left_corner = landmarks[61]
        right_corner = landmarks[291]
        upper_lip_center = landmarks[0]
        corners_avg_y = (left_corner.y + right_corner.y) / 2
        smile_intensity = (upper_lip_center.y - corners_avg_y) * 100
        
        top_lip = landmarks[13]
        bottom_lip = landmarks[14]
        mouth_open = (bottom_lip.y - top_lip.y) * 100

        if mouth_open > 5.0: return "SURPRISED"
        elif smile_intensity > 2.0: return "HAPPY"
        else: return "NEUTRAL"
    except: return "NEUTRAL"

def get_head_pose(landmarks, w, h):
    face_3d = np.array([
        (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
    ], dtype=np.float64)

    face_2d = np.array([[landmarks[i].x * w, landmarks[i].y * h] 
                        for i in [1, 152, 263, 33, 291, 61]], dtype=np.float64)

    focal_length = 1 * w
    cam_matrix = np.array([[focal_length, 0, h/2], [0, focal_length, w/2], [0, 0, 1]])
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return angles[0] * 360, angles[1] * 360, angles[2] * 360

def draw_recording_overlay(frame, is_recording, sample_count):
    h, w, _ = frame.shape
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    if is_recording:
        cv2.circle(frame, (60, 60), 20, (0, 0, 255), -1)
        cv2.putText(frame, "RECORDING...", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(frame, f"Samples: {sample_count}", (100, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(frame, "Make Expressions (Smile, Frown, Surprise)", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "READY TO RECORD", (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

def get_stable_roi(lms, w, h):
    # 234: Left Ear/Cheek, 454: Right Ear/Cheek, 1: Nose Tip
    left_x = lms[234].x * w
    right_x = lms[454].x * w
    nose_x = lms[1].x * w
    nose_y = lms[1].y * h
    
    face_width = right_x - left_x
    box_w = face_width * 1.8
    box_h = box_w * 1.3
    
    x1 = int(nose_x - (box_w / 2))
    y1 = int(nose_y - (box_h / 2))
    y1 = int(y1 - (box_h * 0.1))
    
    x2 = int(x1 + box_w)
    y2 = int(y1 + box_h)
    
    return (max(0, x1), max(0, y1), min(w, x2) - max(0, x1), min(h, y2) - max(0, y1))

# ------------------------------
# 4. VIDEO GENERATOR LOOP
# ------------------------------
def generate_frames():
    cap = cv2.VideoCapture(0)
    # Optional: Lower resolution to 320x240 for even more speed if needed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    cached_name = "Unknown"
    cached_score = 0
    cached_color = (0, 0, 255)
    
    # AI Skipping State
    frame_process_counter = 0
    process_stride = 2   # Process 1 frame, skip 1 frame (Increase for more speed)
    last_results = None
    
    # Caching text for skipped frames
    cached_pose = ""
    cached_blink = ""
    cached_emotion = ""
    cached_liveness_msg = "" # Message to show user

    while True:
        if not state["camera_active"]:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "CAMERA PAUSED", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', blank, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1) 
            continue

        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        frame_process_counter += 1
        should_process_ai = (frame_process_counter % process_stride == 0)

        # AI Processing
        if should_process_ai:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last_results = face_mesh.process(rgb)
        
        # Drawing Logic (Uses last_results)
        if last_results and last_results.multi_face_landmarks:
            for face_landmarks in last_results.multi_face_landmarks:
                lms = face_landmarks.landmark
                
                # Re-calculate lightweight math or reuse? 
                # Calculating geometry is fast, MediaPipe is slow. 
                # We can re-calc pose/ear on old landmarks mapped to new frame (approx)
                # Or just cache the text strings. Let's re-calc to keep logic simple, 
                # landmarks are static relative to head but head moves. 
                # Actually, if we skip MediaPipe, lms are from previous frame.
                
                if should_process_ai:
                    try: 
                        p, y, r = get_head_pose(lms, w, h)
                        cached_pose = f"P:{int(p)} Y:{int(y)}"
                    except: pass
                    
                    l_ear = calculate_ear(lms, LEFT_EYE, w, h)
                    r_ear = calculate_ear(lms, RIGHT_EYE, w, h)
                    is_blinking = ((l_ear + r_ear) / 2.0) < BLINK_THRESHOLD
                    cached_blink = "BLINKING" if is_blinking else "Eyes Open"
                    cached_emotion = detect_emotion(lms)
                
                # Draw using cached or fresh data
                x, y, bw, bh = get_stable_roi(lms, w, h)
                cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 255), 2)
                
                roi = frame[y:y+bh, x:x+bw]
                
                if state["enroll_mode"]:
                    if state["is_recording"] and roi.size > 0:
                        # Only capture on processed frames to ensure quality
                        if should_process_ai:
                            state["frame_counter"] += 1
                            if state["frame_counter"] % CAPTURE_EVERY_N_FRAMES == 0:
                                des = compute_orb(roi)
                                if des is not None:
                                    nm = state["enroll_name"]
                                    if nm not in db: db[nm] = []
                                    db[nm].append(des)
                                    state["session_samples"] += 1
                                    if state["session_samples"] == 1:
                                        cv2.imwrite(f"{KNOWN_DIR}/{nm}_thumb.jpg", roi)
                
                elif roi.size > 0:
                    # Recognition Logic
                    if should_process_ai and (frame_process_counter % (RECOGNITION_INTERVAL * process_stride) == 0):
                        des = compute_orb(roi)
                        name, score = match_descriptors(des, db)
                        state["recent_names"].append(name)
                        if state["recent_names"]:
                            final_name = max(set(state["recent_names"]), key=state["recent_names"].count)
                        else: final_name = "Unknown"
                        
                        if score < THRESHOLD_MATCHES: final_name = "Unknown"
                        
                        # --- SECURITY LOGIC ---
                        # 1. Intruder Alert
                        if final_name == "Unknown":
                            state["liveness_state"] = "WAITING" # Reset liveness
                            cached_name = "Unknown"
                            cached_color = (0, 0, 255)
                            cached_liveness_msg = ""
                            
                            if tuning["INTRUDER_ALERT"]:
                                state["intruder_timer"] += 1
                                # Logic runs every (RECOGNITION_INTERVAL * process_stride) frames.
                                # Interval=4, Stride=2 => Runs every 8 frames.
                                # 30 FPS / 8 = 3.75 checks per second.
                                # To wait 5 seconds: 5 * 3.75 = ~18-20 ticks.
                                if state["intruder_timer"] > 20: 
                                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    path = os.path.join(SECURITY_LOG_DIR, f"intruder_{ts}.jpg")
                                    cv2.imwrite(path, frame)
                                    print(f"[!] Intruder Snapshot Saved: {path}")
                                    state["intruder_timer"] = 0 # Reset to avoid spam
                                    cached_liveness_msg = "ALERT: INTRUDER LOGGED"
                                    state["latest_alert"] = f"Intruder Detected! ({datetime.now().strftime('%H:%M:%S')})"
                        
                        # 2. Liveness Challenge
                        else:
                            state["intruder_timer"] = 0 # Reset intruder
                            cached_name = final_name
                            
                            if state["liveness_state"] == "VERIFIED":
                                cached_color = (0, 255, 0)
                                cached_liveness_msg = "VERIFIED"
                            
                            elif state["liveness_state"] == "WAITING":
                                state["liveness_state"] = "CHALLENGE"
                                state["liveness_target"] = random.choice(["BLINK", "SMILE", "SURPRISE"])
                                state["liveness_timer"] = 0
                                cached_color = (0, 255, 255) # Yellow/Orange
                                cached_liveness_msg = f"ACTION REQUIRED: {state['liveness_target']}"
                            
                            elif state["liveness_state"] == "CHALLENGE":
                                state["liveness_timer"] += 1
                                current_action = None
                                if "BLINKING" in cached_blink: current_action = "BLINK"
                                elif "HAPPY" in cached_emotion: current_action = "SMILE"
                                elif "SURPRISED" in cached_emotion: current_action = "SURPRISE"
                                
                                if current_action == state["liveness_target"]:
                                    state["liveness_state"] = "VERIFIED"
                                    cached_color = (0, 255, 0)
                                    cached_liveness_msg = "VERIFIED: ACCESS GRANTED"
                                
                                if state["liveness_timer"] > 90: # ~3 seconds timeout
                                    state["liveness_state"] = "WAITING" # Retry
                                    cached_liveness_msg = "FAILED: RETRYING..."
                                else:
                                    cached_liveness_msg = f"ACTION REQUIRED: {state['liveness_target']}"
                                    
                        cached_score = score
                        
                    
                    cv2.rectangle(frame, (x, y), (x+bw, y+bh), cached_color, 2)
                    cv2.putText(frame, f"{cached_name} ({int(cached_score)})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cached_color, 2)
                    # Show Liveness/Security Message
                    if cached_liveness_msg:
                        col = (0, 255, 0) if "VERIFIED" in cached_liveness_msg else ((0, 0, 255) if "ALERT" in cached_liveness_msg else (0, 255, 255))
                        cv2.putText(frame, cached_liveness_msg, (x, y+bh+50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
                    elif cached_emotion:
                         cv2.putText(frame, f"{cached_emotion}", (x, y+bh+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if state["enroll_mode"]:
            draw_recording_overlay(frame, state["is_recording"], state["session_samples"])

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


# ------------------------------
# 5. FLASK ROUTES
# ------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/command', methods=['POST'])
@app.route('/api/control', methods=['POST'])
def command():
    cmd = request.json.get('command')
    val = request.json.get('value', '')
    
    if cmd == 'start_enroll':
        if val:
            state["enroll_name"] = "".join(x for x in val if x.isalnum())
            state["enroll_mode"] = True
            state["is_recording"] = False
            state["session_samples"] = 0
            state["recent_names"].clear()
            return jsonify({"status": "Enrollment Mode Started", "name": state["enroll_name"]})
    
    elif cmd == 'toggle_record':
        if state["enroll_mode"]:
            state["is_recording"] = not state["is_recording"]
            if not state["is_recording"]:
                save_db(db)
                state["enroll_mode"] = False
                return jsonify({"status": "Saved & Finished", "samples": state["session_samples"]})
            return jsonify({"status": "Recording..."})
        else:
             return jsonify({"status": "Error: Not in Enrollment Mode"})
    
    elif cmd == 'clear_db':
        db.clear()
        save_db(db)
        state["recent_names"].clear()
        for f in os.listdir(KNOWN_DIR):
            fp = os.path.join(KNOWN_DIR, f)
            if os.path.isfile(fp):
                os.remove(fp)
        return jsonify({"status": "Database Cleared"})
    
    elif cmd == 'toggle_camera':
        state['camera_active'] = not state['camera_active']
        status = "Camera Started" if state['camera_active'] else "Camera Paused"
        return jsonify({"status": status, "active": state['camera_active']})

    # NEW: Update Settings Logic
    elif cmd == 'update_settings':
        try:
            settings = json.loads(val)
            tuning["HAMMING_DISTANCE"] = int(settings.get("sensitivity", 75))
            tuning["SMOOTH_FRAMES"] = int(settings.get("smoothing", 8))
            tuning["INTRUDER_ALERT"] = bool(settings.get("intruder_alert", tuning["INTRUDER_ALERT"]))
            state["recent_names"] = deque(maxlen=tuning["SMOOTH_FRAMES"])
            return jsonify({"status": "Settings Updated"})
        except Exception as e:
            return jsonify({"status": f"Error updating settings: {str(e)}"})
            
    elif cmd == 'toggle_intruder':
        tuning["INTRUDER_ALERT"] = not tuning["INTRUDER_ALERT"]
        status = "ON" if tuning["INTRUDER_ALERT"] else "OFF"
        return jsonify({"status": f"Intruder Alert {status}"})

    return jsonify({"status": "Unknown Command"})

@app.route('/api/faces', methods=['GET'])
def get_faces():
    keys_snapshot = list(db.keys())
    counts = {name: len(db[name]) for name in keys_snapshot}
    counts = {name: len(db[name]) for name in keys_snapshot}
    return jsonify({"faces": keys_snapshot, "counts": counts})

@app.route('/api/status_updates')
def status_updates():
    alert = state.get("latest_alert")
    if alert:
        state["latest_alert"] = None # Clear after reading
        return jsonify({"alert": alert})
    return jsonify({})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)