import csv
from pathlib import Path
from typing import Dict, Any


class LocalCSVLogger:
    def __init__(self, file_path: str) -> None:
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.file_path.exists() or self.file_path.stat().st_size == 0:
            self._write_header()

    def _write_header(self) -> None:
        headers = [
            "timestamp",
            "detected_name",
            "confidence",
            "granted_or_denied",
            "reason",
        ]
        with self.file_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=headers)
            writer.writeheader()

    def log_attempt(self, row: Dict[str, Any]) -> None:
        with self.file_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "timestamp",
                    "detected_name",
                    "confidence",
                    "granted_or_denied",
                    "reason",
                ],
            )
            writer.writerow(row)
