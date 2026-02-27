from pathlib import Path
from typing import Dict, Any

from deepface import DeepFace


class FaceRecognizer:
    def __init__(
        self,
        dataset_path: str,
        threshold: float = 0.35,
        model_name: str = "VGG-Face",
        detector_backend: str = "opencv",
        distance_metric: str = "cosine",
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.threshold = threshold
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.distance_metric = distance_metric

    def is_dataset_ready(self) -> bool:
        if not self.dataset_path.exists():
            return False
        image_count = 0
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            image_count += len(list(self.dataset_path.rglob(pattern)))
        return image_count > 0

    def recognize(self, face_image) -> Dict[str, Any]:
        result = {
            "matched_name": "Unknown",
            "distance": None,
            "confidence": 0.0,
            "verified": False,
            "reason": "No match",
        }

        if not self.is_dataset_ready():
            result["reason"] = "Dataset is empty or missing"
            return result

        try:
            matches = DeepFace.find(
                img_path=face_image,
                db_path=str(self.dataset_path),
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                distance_metric=self.distance_metric,
                enforce_detection=False,
                silent=True,
            )

            if not matches or matches[0].empty:
                result["reason"] = "No candidate found"
                return result

            best_row = matches[0].iloc[0]
            identity_path = Path(str(best_row.get("identity", "")))
            distance = float(best_row.get(self.distance_metric, best_row.get("distance", 1.0)))
            matched_name = identity_path.parent.name if identity_path.parent.name else "Unknown"
            if matched_name != "Unknown":
                matched_name = matched_name.strip().title()
            verified = distance <= self.threshold
            confidence = max(0.0, min(1.0, 1.0 - (distance / max(self.threshold, 1e-6))))
            display_name = matched_name if verified else "Unknown"

            result.update(
                {
                    "matched_name": display_name,
                    "distance": distance,
                    "confidence": confidence,
                    "verified": verified,
                    "reason": "Face matched" if verified else "Distance above threshold",
                }
            )
            return result

        except Exception as exc:
            result["reason"] = f"Recognition error: {exc}"
            return result
