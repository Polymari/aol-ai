from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import cv2
import numpy as np
import time
import os

# Import only the functions we need, NOT db
from MAIN import (
    compute_orb_descriptors, 
    extract_face_roi, 
    load_db, 
    save_db, 
    ensure_dirs,
    MIN_FACE_SIZE
)

app = FastAPI()

# Add CORS middleware to allow browser requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize our own db variable at module level
ensure_dirs()
db = load_db()  # This creates our own db dictionary
if not db:
    db = {}  # Start with empty dict if no saved db exists
    print("[i] No existing database, starting fresh")
else:
    print(f"[i] Loaded {len(db)} faces from database")

# Initialize cascade for face detection
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Serve the enrollment HTML at root
@app.get("/")
async def read_root():
    return FileResponse("index.html")

# Serve the recognition page
@app.get("/recognize")
async def recognize_page():
    return FileResponse("recognize.html")

@app.post("/enroll")
async def enroll(name: str, file: UploadFile):
    global db  # We need to modify the module-level db
    
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    
    if img is None:
        return {"status": "failed", "reason": "invalid image"}

    # Detect face first
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=MIN_FACE_SIZE)
    
    if len(rects) == 0:
        return {"status": "failed", "reason": "no face detected"}
    
    # Use largest face
    rects = sorted(rects, key=lambda r: r[2]*r[3], reverse=True)
    roi = extract_face_roi(img, rects[0])

    # Save image to known/ folder
    path = f"known/{name}_{int(time.time())}.jpg"
    cv2.imwrite(path, roi)

    # Extract descriptors from face ROI
    des = compute_orb_descriptors(roi)
    if des is None:
        return {"status": "failed", "reason": "no descriptors extracted"}

    # Add to database
    if name not in db:
        db[name] = []
    db[name].append(des)
    
    # Save to disk
    save_db(db)
    
    print(f"[+] Enrolled {name} - Total samples: {len(db[name])}")

    return {"status": "ok", "name": name, "total_samples": len(db[name])}

@app.get("/faces")
async def list_faces():
    """List all enrolled faces"""
    return {
        "faces": list(db.keys()),
        "count": len(db),
        "samples": {name: len(samples) for name, samples in db.items()}
    }

@app.post("/recognize")
async def recognize_face(file: UploadFile):
    """Recognize a face from an uploaded image"""
    from MAIN import match_descriptors
    
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    
    if img is None:
        return {"status": "failed", "reason": "invalid image"}

    # Detect face
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=MIN_FACE_SIZE)
    
    if len(rects) == 0:
        return {"status": "failed", "reason": "no face detected", "name": "Unknown", "confidence": 0}
    
    # Use largest face
    rects = sorted(rects, key=lambda r: r[2]*r[3], reverse=True)
    roi = extract_face_roi(img, rects[0])

    # Extract descriptors
    des = compute_orb_descriptors(roi)
    if des is None:
        return {"status": "failed", "reason": "no descriptors extracted", "name": "Unknown", "confidence": 0}

    # Match against database
    name, score = match_descriptors(des, db)
    
    # Apply threshold (same as MAIN.py default)
    threshold = 6
    if score < threshold:
        name = "Unknown"
    
    return {
        "status": "ok",
        "name": name,
        "confidence": float(score),
        "threshold": threshold,
        "faces_detected": len(rects)
    }

if __name__ == "__main__":
    import uvicorn
    import logging
    
    # Reduce noisy warnings
    logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")