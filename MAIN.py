import cv2
import mediapipe as mp
import numpy as np
import os
import time
import pickle
import json
import random
import threading
import queue
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
    "HAMMING_DISTANCE": 55,
    "SMOOTH_FRAMES": 8,
    "INTRUDER_ALERT": True
}


THRESHOLD_MATCHES = 200     
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
    "latest_alert": None,        # Queue for UI messages
    "privacy_mode": False        # True if user is looking away
}


# ------------------------------
# 1. MEDIAPIPE SETUP
# ------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=False, # Optimization: Turn off Iris tracking
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


def prune_db(db, max_samples=50):
    for name in db:
        if len(db[name]) > max_samples:
            # Keep random 50 samples to maintain variety
            db[name] = random.sample(db[name], max_samples)
    return db

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
        # Landmarks: 
        # 0: Upper Lip Top, 13: Upper Lip Inner, 14: Lower Lip Inner
        # 61: Left Mouth Corner, 291: Right Mouth Corner
        
        left_corner = landmarks[61]
        right_corner = landmarks[291]
        upper_lip = landmarks[0]
        
        # Calculate Mouth Width (Normalizer)
        # Use Euclidean distance for better accuracy even if head is turned
        # Since landmarks are normalized (0-1), this works fine.
        mouth_width = np.hypot(left_corner.x - right_corner.x, left_corner.y - right_corner.y)
        if mouth_width < 0.001: mouth_width = 0.001 # Prevent div by zero

        # Calculate Y-coordinates rel to lip center
        corners_avg_y = (left_corner.y + right_corner.y) / 2.0
        lip_y = upper_lip.y
        
        # Smile: Corners are HIGHER (smaller Y) than lip
        # Frown: Corners are LOWER (larger Y) than lip
        # Normalized Curvature Ratio
        curvature_ratio = (lip_y - corners_avg_y) / mouth_width
        
        # Mouth Open Ratio
        top_lip_inner = landmarks[13]
        bottom_lip_inner = landmarks[14]
        open_ratio = (bottom_lip_inner.y - top_lip_inner.y) / mouth_width

        # Debug Prints (visible in console for tuning)
        # print(f"DEBUG: Curves: {curvature_ratio:.3f}, Open: {open_ratio:.3f}")

        # Tuning Thresholds (Normalized)
        # Typical Values from observation:
        # Neutral: Curvature ~0.0 to -0.1
        # Smile: Curvature > 0.15
        # Frown: Curvature < -0.2
        # Open: > 0.3
        
        expert_override = "" # Return this along with emotion for debug if needed; simplified here.

        if open_ratio > 0.25: return "SURPRISED" 
        elif curvature_ratio > 0.12: return "HAPPY" 
        elif curvature_ratio < -0.20: return "FROWN" 
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
    return angles[0], angles[1], angles[2]

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

    return (max(0, x1), max(0, y1), min(w, x2) - max(0, x1), min(h, y2) - max(0, y1))

# ------------------------------
# 4A. ASYNC RECOGNITION WORKER
# ------------------------------
class RecognitionWorker:
    def __init__(self, db_ref):
        self.db = db_ref
        self.input_queue = queue.Queue(maxsize=4) # Buffer a few frames
        self.latest_results = {} # Key: face_index, Value: (name, score)
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()
    
    def _worker_loop(self):
        while self.running:
            try:
                # Wait for a new ROI to process
                # Item is (roi, face_index)
                item = self.input_queue.get(timeout=1)
                roi, idx = item
                
                # Heavy lifting happens here (OFF main thread)
                des = compute_orb(roi)
                name, score = match_descriptors(des, self.db)
                
                with self.lock:
                    self.latest_results[idx] = (name, score)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[!] Worker Error: {e}")

    def process(self, roi, idx):
        # Non-blocking put (drop frame if busy)
        if not self.input_queue.full():
            self.input_queue.put((roi, idx))

    def get_latest_results(self):
        with self.lock:
            return self.latest_results.copy()

    def update_db(self, new_db):
        self.db = new_db

# Initialize Worker
rec_worker = RecognitionWorker(db)

# ------------------------------
# 4. VIDEO GENERATOR LOOP
# ------------------------------
def generate_frames():
    # Use DirectShow on Windows for better stability
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # Optional: Lower resolution to 320x240 for even more speed if needed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("[*] Camera Initialized")
    if not cap.isOpened():
        print("[!] Error: Could not open camera completely.")

    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    # Per-face caching (Key: face_index)
    cache_data = defaultdict(lambda: {
        "name": "Unknown", "score": 0, "color": (0, 0, 255),
        "pose": "", "blink": "", "emotion": "", "liveness_msg": ""
    })

    # AI Skipping State
    frame_process_counter = 0
    process_stride = 2   # Process 1 frame, skip 1 frame (Increase for more speed)
    last_results = None
    
    consecutive_errors = 0

    try:
        while True:
            try:
                if not state["camera_active"]:
                    blank = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank, "CAMERA PAUSED", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    ret, buffer = cv2.imencode('.jpg', blank, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                    frame = buffer.tobytes()
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    time.sleep(0.1) 
                    continue
    
                success, frame = cap.read()
                if not success: 
                    consecutive_errors += 1
                    print(f"[!] Error: Failed to read frame (Attempt {consecutive_errors}/10). Retrying...")
                    
                    if consecutive_errors > 10:
                        print("[!] Critical: Camera connection lost. Stopping stream.")
                        break
                        
                    time.sleep(0.2)
                    continue
                
                consecutive_errors = 0 # Reset on success
            
                frame = cv2.flip(frame, 1)
                # Force frame to standard size to ensure consistent processing
                frame = cv2.resize(frame, (640, 480))
                h, w, _ = frame.shape
            
                frame_process_counter += 1
                should_process_ai = (frame_process_counter % process_stride == 0)

                # AI Processing
                if should_process_ai:
                    # Optimization: Resize to 50% for faster inference (320x240)
                    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    last_results = face_mesh.process(rgb)
            
                # Drawing Logic (Uses last_results)
                if last_results and last_results.multi_face_landmarks:
                    for idx, face_landmarks in enumerate(last_results.multi_face_landmarks):
                        lms = face_landmarks.landmark
                        d = cache_data[idx] # Point to this face's data dictionary

                        if should_process_ai:
                            try: 
                                p, y, r = get_head_pose(lms, w, h)
                                d["pose"] = f"P:{int(p)} Y:{int(y)}"
                            
                                # --- ATTENTION GUARD LOGIC ---
                                if abs(y) > 50 or abs(p) > 35:
                                    if idx == 0: state["privacy_mode"] = True
                                else:
                                    if idx == 0: state["privacy_mode"] = False
                                
                            except: 
                                if idx == 0: state["privacy_mode"] = False
                                pass
                        
                            l_ear = calculate_ear(lms, LEFT_EYE, w, h)
                            r_ear = calculate_ear(lms, RIGHT_EYE, w, h)
                            is_blinking = ((l_ear + r_ear) / 2.0) < BLINK_THRESHOLD
                            d["blink"] = "BLINKING" if is_blinking else "Eyes Open"
                            d["emotion"] = detect_emotion(lms)
                    
                        # Draw using cached or fresh data
                        x, y, bw, bh = get_stable_roi(lms, w, h)
                    
                        # Only use Face 0 for Enrollment
                        if state["enroll_mode"] and idx == 0:
                            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 255), 2)
                            roi = frame[y:y+bh, x:x+bw]

                            if state["is_recording"] and roi.size > 0:
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
                    
                        else: 
                            # Recognition Mode
                            roi = frame[y:y+bh, x:x+bw]
                            if roi.size > 0:
                                if should_process_ai and (frame_process_counter % (RECOGNITION_INTERVAL * process_stride) == 0):
                                    rec_worker.process(roi, idx)
                                
                                # Get latest results for ALL faces
                                all_results = rec_worker.get_latest_results()
                                if idx in all_results:
                                    final_name, score = all_results[idx]
                                
                                    if score < THRESHOLD_MATCHES: final_name = "Unknown"
                                
                                    d["name"] = final_name
                                    d["score"] = score
                                
                                    # --- SECURITY LOGIC ---
                                    if final_name == "Unknown":
                                        d["color"] = (0, 0, 255)
                                        d["liveness_msg"] = ""
                                    
                                        if tuning["INTRUDER_ALERT"]:
                                            state["intruder_timer"] += 1
                                            if state["intruder_timer"] > 150: 
                                                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                                                path = os.path.join(SECURITY_LOG_DIR, f"intruder_{ts}.jpg")
                                                cv2.imwrite(path, frame)
                                                print(f"[!] Intruder Snapshot Saved: {path}")
                                                state["intruder_timer"] = 0
                                                state["latest_alert"] = f"Intruder Detected! ({datetime.now().strftime('%H:%M:%S')})"
                                                d["liveness_msg"] = "ALERT: INTRUDER LOGGED"
                                    else:
                                        state["intruder_timer"] = 0 
                                        d["color"] = (0, 255, 0)
                                        if idx == 0:
                                            if state["liveness_state"] == "VERIFIED":
                                                d["color"] = (0, 255, 0)
                                                d["liveness_msg"] = "VERIFIED"
                                        
                                            elif state["liveness_state"] == "WAITING":
                                                state["liveness_state"] = "CHALLENGE"
                                                state["liveness_target"] = random.choice(["BLINK", "SMILE", "FROWN", "SURPRISE"])
                                                state["liveness_timer"] = 0
                                                d["color"] = (0, 255, 255)
                                                d["liveness_msg"] = f"ACTION: {state['liveness_target']}"
                                        
                                            elif state["liveness_state"] == "CHALLENGE":
                                                state["liveness_timer"] += 1
                                                current_action = None
                                                if "BLINKING" in d["blink"]: current_action = "BLINK"
                                                elif "HAPPY" in d["emotion"]: current_action = "SMILE"
                                                elif "FROWN" in d["emotion"]: current_action = "FROWN"
                                                elif "SURPRISED" in d["emotion"]: current_action = "SURPRISE"
                                            
                                                if current_action == state["liveness_target"]:
                                                    state["liveness_state"] = "VERIFIED"
                                                    d["color"] = (0, 255, 0)
                                                    d["liveness_msg"] = "VERIFIED: AUTHORIZED"
                                            
                                                if state["liveness_timer"] > 90:
                                                    state["liveness_state"] = "WAITING"
                                                    d["liveness_msg"] = "FAILED: RETRY..."
                                                else:
                                                    d["liveness_msg"] = f"ACTION: {state['liveness_target']}"
                            
                            cv2.rectangle(frame, (x, y), (x+bw, y+bh), d["color"], 2)
                            cv2.putText(frame, f"{d['name']} ({int(d['score'])})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, d["color"], 2)
                        
                            if d["liveness_msg"]:
                                 col = d["color"]
                                 cv2.putText(frame, d["liveness_msg"], (x, y+bh+50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
                            elif d["emotion"]:
                                 cv2.putText(frame, f"{d['emotion']}", (x, y+bh+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                            # DEBUG VISUALIZATION (Temporary: Remove for Prod)
                            # Helps user see why detection fails
                            if idx == 0 and should_process_ai:
                                 # Re-calc for display (inefficient but safe for debug)
                                 try:
                                     lms = face_landmarks.landmark
                                     lc = lms[61]; rc = lms[291]; ul = lms[0]
                                     mw = np.hypot(lc.x - rc.x, lc.y - rc.y) or 0.001
                                     curve = (ul.y - (lc.y + rc.y)/2) / mw
                                     opn = (lms[14].y - lms[13].y) / mw
                                     cv2.putText(frame, f"C:{curve:.2f} O:{opn:.2f}", (x, y+bh+80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                                 except: pass

                if state["enroll_mode"]:
                    draw_recording_overlay(frame, state["is_recording"], state["session_samples"])

                ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                frame = buffer.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
            except Exception as e:
                print(f"[!] Critical Error in Video Loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)
            
    finally:
        cap.release()
        print("[*] Camera Resources Released Cleanly")


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
                
                # Prune DB after new enrollment
                prune_db(db)
                save_db(db) # Save again after pruning
                rec_worker.update_db(db) # Update worker with new data
                
                return jsonify({"status": "Saved & Finished", "samples": state["session_samples"]})
            return jsonify({"status": "Recording..."})
        else:
             return jsonify({"status": "Error: Not in Enrollment Mode"})
    
    elif cmd == 'clear_db':
        # 1. Clear In-Memory DB and State
        db.clear()
        state["recent_names"].clear()
        state["liveness_state"] = "WAITING"
        state["enroll_mode"] = False
        state["is_recording"] = False
        state["session_samples"] = 0
        state["enroll_name"] = ""

        # 2. Clear known faces
        for f in os.listdir(KNOWN_DIR):
            fp = os.path.join(KNOWN_DIR, f)
            if os.path.isfile(fp):
                os.remove(fp)
        
        # 3. Clear Security Logs (Intruder snapshots)
        if os.path.exists(SECURITY_LOG_DIR):
            for f in os.listdir(SECURITY_LOG_DIR):
                fp = os.path.join(SECURITY_LOG_DIR, f)
                if os.path.isfile(fp):
                    os.remove(fp)

        # 4. Clear Event Log
        if os.path.exists(LOG_PATH):
            os.remove(LOG_PATH)

        # 5. Save empty DB state to disk
        save_db(db)
        
        # 6. Update worker
        rec_worker.update_db(db)
        
        print("[!] SYSTEM RESET: All data cleared.")
        return jsonify({"status": "All Data & Logs Cleared"})
    
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
    privacy = state.get("privacy_mode", False)
    
    response = {"privacy": privacy}
    
    if alert:
        state["latest_alert"] = None # Clear after reading
        response["alert"] = alert
        
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)