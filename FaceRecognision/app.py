from datetime import datetime

import cv2

from config import (
	DATASET_DIR,
	DEEPFACE_DETECTOR_BACKEND,
	DEEPFACE_DISTANCE_METRIC,
	DEEPFACE_MODEL_NAME,
	FRAME_HEIGHT,
	FRAME_WIDTH,
	LOCAL_LOG_FILE,
	MATCH_DISTANCE_THRESHOLD,
	RECOGNITION_EVERY_N_FRAMES,
	SUPABASE_KEY,
	SUPABASE_LOGS_TABLE,
	SUPABASE_URL,
	WEBCAM_INDEX,
)
from utils.camera import CameraStream
from utils.database import CloudDatabase
from utils.logger import LocalCSVLogger
from utils.recognition import FaceRecognizer


def build_log_row(detected_name: str, confidence: float, granted: bool, reason: str) -> dict:
	return {
		"timestamp": datetime.utcnow().isoformat(),
		"detected_name": detected_name,
		"confidence": round(float(confidence), 4),
		"granted_or_denied": "GRANTED" if granted else "DENIED",
		"reason": reason,
	}


def main() -> None:
	camera = CameraStream(camera_index=WEBCAM_INDEX, width=FRAME_WIDTH, height=FRAME_HEIGHT)
	recognizer = FaceRecognizer(
		dataset_path=str(DATASET_DIR),
		threshold=MATCH_DISTANCE_THRESHOLD,
		model_name=DEEPFACE_MODEL_NAME,
		detector_backend=DEEPFACE_DETECTOR_BACKEND,
		distance_metric=DEEPFACE_DISTANCE_METRIC,
	)
	cloud_db = CloudDatabase(
		supabase_url=SUPABASE_URL,
		supabase_key=SUPABASE_KEY,
		logs_table=SUPABASE_LOGS_TABLE,
	)
	local_logger = LocalCSVLogger(str(LOCAL_LOG_FILE))

	if not camera.open():
		print("Error: Unable to open webcam.")
		return

	print("Premium Guest Face-Recognition Entry started.")
	print("Press 'q' to quit.")

	frame_counter = 0
	latest_status = {
		"name": "Unknown",
		"confidence": 0.0,
		"status": "WAITING",
		"reason": "No scan yet",
	}

	while True:
		ok, frame = camera.read()
		if not ok or frame is None:
			print("Warning: failed to read frame from webcam.")
			break

		frame_counter += 1
		bbox = camera.detect_largest_face(frame)

		if bbox is not None:
			x, y, w, h = bbox
			cv2.rectangle(frame, (x, y), (x + w, y + h), (70, 200, 255), 2)

			if frame_counter % RECOGNITION_EVERY_N_FRAMES == 0:
				face_crop = camera.crop_face(frame, bbox)
				rec = recognizer.recognize(face_crop)

				detected_name = rec["matched_name"]
				confidence = float(rec["confidence"])
				granted = bool(rec["verified"])
				reason = "Face found in dataset" if granted else "Face not found in dataset"

				latest_status = {
					"name": detected_name,
					"confidence": confidence,
					"status": "ACCESS GIVEN" if granted else "ACCESS DENIED",
					"reason": reason,
				}

				log_row = build_log_row(detected_name, confidence, granted, reason)
				local_logger.log_attempt(log_row)
				cloud_db.log_access_attempt(log_row)
		else:
			cv2.putText(
				frame,
				"No face detected",
				(20, 40),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.8,
				(100, 100, 255),
				2,
			)

		status_color = (80, 220, 120) if latest_status["status"] == "ACCESS GIVEN" else (40, 80, 255)
		cv2.putText(frame, f"Name: {latest_status['name']}", (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
		cv2.putText(frame, f"Confidence: {latest_status['confidence']:.2f}", (20, 490), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
		cv2.putText(frame, f"Reason: {latest_status['reason']}", (20, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
		cv2.putText(frame, latest_status["status"], (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)

		cv2.imshow("Premium Guest Face-Recognition Entry", frame)
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

	camera.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()

