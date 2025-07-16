import sys
import base64
import io
import os
import time
from typing import Optional, Tuple, List

import numpy as np
import requests
from PyQt6.QtCore import Qt, QTimer, QPointF, QObject, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPixmap, QPainter, QPen, QBrush, QPolygonF
from PyQt6.QtWidgets import (QApplication, QGridLayout, QLabel,
                               QMainWindow, QWidget, QPushButton)

from control_client import ControllerWrapper

# --- Shared Client/API Logic ---
# This should be refactored into a dedicated library later.

BASE_URL = "http://127.0.0.1:8000"
AI_SERVER_URL = "http://192.168.200.130:8001"
# Correct the path to be relative to the project root, not the visualizer dir
GOAL_IMAGE_PATH = "testdata/go2_dataset_20250714_191924/90.jpg"


def local_to_global(local_xy: np.ndarray, robot_pose: Tuple[float, float, float]) -> np.ndarray:
    """Convert a set of (x,y) points expressed in the robot frame to world frame."""
    x, y, theta = robot_pose
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return local_xy @ R.T + np.array([x, y])


def fetch_waypoints() -> Optional[List[np.ndarray]]:
    """Request waypoint predictions from the AI server."""
    try:
        headers = {"Cache-Control": "no-cache"}
        resp = requests.get(f"{AI_SERVER_URL}/get_latest_info/", timeout=1.0, headers=headers)
        resp.raise_for_status()
        waypoints_data = resp.json().get("latest_waypoints")

        if waypoints_data:
            try:
                # The endpoint returns a list of trajectories.
                return [np.array(traj, dtype=float) for traj in waypoints_data]
            except (ValueError, TypeError):
                print(f"[WARN] Waypoint data is not valid: {waypoints_data}")
                return None
        return None
    except requests.RequestException:
        # Quieter log for frequent timeouts
        return None


def fetch_pose():
    """Retrieve the robot's current pose (x, y, theta)."""
    headers = {"Cache-Control": "no-cache"}
    resp = requests.get(f"{BASE_URL}/pose", timeout=1.0, headers=headers)
    resp.raise_for_status()
    odo = resp.json()["odometry"]
    return odo["x"], odo["y"], odo["theta"]


def fetch_scan():
    """Retrieve the latest laser scan (angles, ranges) as NumPy arrays."""
    headers = {"Cache-Control": "no-cache"}
    resp = requests.get(f"{BASE_URL}/scan", timeout=1.0, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    angles = np.array(data["angles"], dtype=np.float32)
    ranges = np.array(data["ranges"], dtype=np.float32)
    return angles, ranges


def fetch_latest_frame(n: int = 1):
    """Fetch the most recent camera frame from the /frames endpoint."""
    try:
        headers = {"Cache-Control": "no-cache"}
        resp = requests.get(f"{BASE_URL}/frames?n={n}", timeout=1.0, headers=headers)
        resp.raise_for_status()
        frames = resp.json().get("frames", [])
        if not frames:
            return None
        return frames[-1]["image"]  # Base-64 encoded JPEG
    except requests.RequestException:
        return None

def get_goal_image_pixmap():
    """Load the goal image from file and return as a QPixmap."""
    try:
        with open(GOAL_IMAGE_PATH, "rb") as f:
            img_data = f.read()
            pixmap = QPixmap()
            pixmap.loadFromData(img_data)
            return pixmap
    except FileNotFoundError:
        print(f"[ERROR] Goal image not found at: {GOAL_IMAGE_PATH}")
        return None
    except Exception as e:
        print(f"[ERROR] Could not load goal image: {e}")
        return None


class Worker(QObject):
    """
    Handles all network requests and data processing in a background thread.
    """
    data_ready = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.controller = ControllerWrapper()
        # Fallback trajectory
        self.dummy_local_waypoints = np.zeros((10, 2))
        self.dummy_local_waypoints[:, 0] = np.linspace(0, 5, 10)
        # Store last known waypoints to decouple update rates
        self.last_known_waypoints = [self.dummy_local_waypoints]
        self._is_running = False

    def run(self):
        """Continuously fetches and processes data."""
        self._is_running = True
        loop_counter = 0
        while self._is_running:
            data_payload = {}
            try:
                # --- Fetch high-frequency data every loop ---
                pose = fetch_pose()
                angles, ranges = fetch_scan()
                frame_b64 = fetch_latest_frame()

                # --- Fetch low-frequency waypoint data periodically ---
                # if loop_counter % 5 == 0:  # Update waypoints every ~500ms
                #     fetched_waypoints = fetch_waypoints()
                #     if fetched_waypoints:
                #         self.last_known_waypoints = fetched_waypoints

                # --- Process data using the last known waypoints ---
                costmap_obj = self.controller.update_costmap(pose, angles, ranges)
                costmap_np = costmap_obj.get_costmap_numpy()
                global_waypoints = [local_to_global(traj, pose) for traj in self.last_known_waypoints]

                data_payload = {
                    "pose": pose,
                    "frame_b64": frame_b64,
                    "local_waypoints": self.last_known_waypoints,
                    "global_waypoints": global_waypoints,
                    "costmap_obj": costmap_obj,
                    "costmap_np": costmap_np,
                }
                self.data_ready.emit(data_payload)

            except requests.RequestException:
                # Don't spam the console for timeouts, just try again
                pass
            except Exception as e:
                print(f"[Worker Thread Error] {e}")

            loop_counter += 1
            time.sleep(0.1)  # Aim for 10 FPS for high-frequency data

    def stop(self):
        self._is_running = False


class VisualizerWindow(QMainWindow):
    """Main application window for the visualizer."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robot Visualizer")
        self.setGeometry(100, 100, 1200, 600)

        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        grid_layout = QGridLayout(central_widget)

        # --- Widgets to display data ---
        self.camera_frame_label = QLabel("Waiting for camera frame...")
        self.camera_frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_frame_label.setMinimumSize(640, 480)
        self.camera_frame_label.setStyleSheet("background-color: black; color: white;")

        self.goal_image_label = QLabel("Goal Image")
        self.goal_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.goal_image_label.setStyleSheet("background-color: black; color: white;")

        self.costmap_label = QLabel("Waiting for costmap...")
        self.costmap_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.costmap_label.setStyleSheet("background-color: black; color: white;")

        # --- Recording state and button ---
        self.is_recording = False
        self.recording_path = ""
        self.frame_count = 0
        self.record_button = QPushButton("Record")
        self.record_button.setCheckable(True)
        self.record_button.clicked.connect(self.toggle_recording)
        self.recording_frequency = 5  # Hz
        self.last_save_time = 0

        # --- Layout ---
        grid_layout.addWidget(self.camera_frame_label, 0, 0)
        grid_layout.addWidget(self.goal_image_label, 1, 0)
        grid_layout.addWidget(self.record_button, 2, 0)
        grid_layout.addWidget(self.costmap_label, 0, 1, 3, 1)
        grid_layout.setColumnStretch(0, 1)
        grid_layout.setColumnStretch(1, 2)

        self.load_goal_image()

        # --- Setup background worker thread ---
        self.thread = QThread()
        self.worker = Worker()
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.data_ready.connect(self.update_ui)
        
        self.thread.start()

    def update_ui(self, data: dict):
        """Slot to receive data from the worker and update all UI elements."""
        if not data:
            return

        self.update_camera_frame(data.get("frame_b64"), data.get("local_waypoints"))
        self.update_costmap(
            data.get("pose"),
            data.get("costmap_obj"),
            data.get("costmap_np"),
            data.get("global_waypoints"),
        )

    def load_goal_image(self):
        """Loads and displays the static goal image."""
        if not os.path.exists(GOAL_IMAGE_PATH):
            self.goal_image_label.setText(f"Goal Image not found:\n{GOAL_IMAGE_PATH}")
            return
            
        goal_pixmap = get_goal_image_pixmap()
        if goal_pixmap:
            self.goal_image_label.setPixmap(
                goal_pixmap.scaled(
                    320, 240, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                )
            )
        else:
            self.goal_image_label.setText("Could not load goal image.")

    def update_camera_frame(self, img_b64: Optional[str], local_waypoints: Optional[List[np.ndarray]]):
        """Updates the camera feed display and saves frames if recording."""
        if not img_b64:
            self.camera_frame_label.setText("No camera frame received")
            return
        
        try:
            img_data = base64.b64decode(img_b64)

            if self.is_recording:
                self.save_frame(img_data)

            pixmap = QPixmap()
            pixmap.loadFromData(img_data, "JPEG")

            if local_waypoints:
                self.draw_waypoints_on_frame(pixmap, local_waypoints)

            self.camera_frame_label.setPixmap(
                pixmap.scaled(
                    self.camera_frame_label.width(),
                    self.camera_frame_label.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        except Exception as e:
            print(f"Error displaying camera frame: {e}")
            self.camera_frame_label.setText("Error displaying frame")
    
    def update_costmap(self, pose, costmap_obj, costmap_np, global_waypoints):
        """Updates the costmap display."""
        if costmap_np is None or pose is None or costmap_obj is None:
            self.costmap_label.setText("Waiting for costmap data...")
            return

        try:
            costmap_np_flipped = np.flipud(costmap_np)
            q_image = self.numpy_to_qimage(costmap_np_flipped)
            pixmap = QPixmap.fromImage(q_image)

            self.draw_robot_on_pixmap(pixmap, pose, costmap_obj)

            if global_waypoints:
                self.draw_waypoints_on_costmap(pixmap, global_waypoints, costmap_obj, pose)

            self.costmap_label.setPixmap(
                pixmap.scaled(
                    self.costmap_label.width(),
                    self.costmap_label.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        except Exception as e:
            print(f"Error displaying costmap: {e}")
            self.costmap_label.setText("Error displaying costmap")

    def toggle_recording(self, checked: bool):
        """Starts or stops recording based on the button's state."""
        self.is_recording = checked
        if self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """Initializes a new recording session."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.recording_path = os.path.join("testdata", f"recording_{timestamp}")
        os.makedirs(self.recording_path, exist_ok=True)
        self.frame_count = 0
        self.last_save_time = time.time()
        self.record_button.setText("Stop Recording")
        self.record_button.setStyleSheet("background-color: red; color: white;")
        print(f"Started recording to {self.recording_path}")

    def stop_recording(self):
        """Finalizes the current recording session."""
        self.record_button.setText("Record")
        self.record_button.setStyleSheet("")  # Revert to default style
        if self.frame_count > 0:
            print(f"Stopped recording. Saved {self.frame_count} frames to {self.recording_path}")
        else:
            print("Stopped recording. No frames were saved.")
        # Reset state
        self.recording_path = ""
        self.is_recording = False
        if self.record_button.isChecked():
            self.record_button.setChecked(False)

    def save_frame(self, img_data: bytes):
        """Saves a single frame to the recording directory at the specified frequency."""
        if not self.recording_path:
            return

        # Enforce recording frequency
        current_time = time.time()
        if (current_time - self.last_save_time) < (1 / self.recording_frequency):
            return
        self.last_save_time = current_time

        try:
            filename = f"{self.frame_count:04d}.jpg"
            filepath = os.path.join(self.recording_path, filename)
            with open(filepath, "wb") as f:
                f.write(img_data)
            self.frame_count += 1
        except Exception as e:
            print(f"[ERROR] Could not save frame: {e}")

    def draw_waypoints_on_costmap(self, pixmap: QPixmap, trajectories: List[np.ndarray], costmap, robot_pose: Tuple[float, float, float]):
        """Draws multiple waypoint paths on the costmap pixmap."""
        painter = QPainter(pixmap)
        
        def world_to_pixel(wx, wy):
            resolution = costmap.get_resolution()
            origin_x = costmap.get_origin_x()
            origin_y = costmap.get_origin_y()
            px = (wx - origin_x) / resolution
            py = pixmap.height() - ((wy - origin_y) / resolution)
            return QPointF(px, py)

        robot_start_point = world_to_pixel(robot_pose[0], robot_pose[1])

        for trajectory in trajectories:
            if trajectory.size == 0:
                continue
            
            painter.setPen(QPen(Qt.GlobalColor.yellow, 2, Qt.PenStyle.DashLine))
            path = QPolygonF([robot_start_point])
            for point in trajectory:
                path.append(world_to_pixel(point[0], point[1]))
            painter.drawPolyline(path)

        painter.end()


    def draw_waypoints_on_frame(self, pixmap: QPixmap, trajectories: List[np.ndarray]):
        """Draws multiple local waypoint trajectories on the camera frame."""
        painter = QPainter(pixmap)
        for trajectory in trajectories[:1]:
            self.draw_single_trajectory_on_frame(painter, pixmap, trajectory)
        # No need for painter.end(), it's managed by the QPixmap lifetime

    def draw_single_trajectory_on_frame(self, painter: QPainter, pixmap: QPixmap, waypoints: np.ndarray):
        """Draws a single local waypoint path on the camera frame with faked perspective."""
        if waypoints.size == 0:
            return

        painter.setPen(QPen(Qt.GlobalColor.green, 3, Qt.PenStyle.SolidLine))

        img_width = pixmap.width()
        img_height = pixmap.height()

        vanishing_point = QPointF(img_width / 2, img_height * 0.55)
        horizontal_stretch = 120
        vertical_stretch = 100

        path = QPolygonF()
        path_origin = QPointF(img_width / 2, img_height)
        path.append(path_origin)

        for point in waypoints:
            robot_x, robot_y = point[0], point[1]
            if robot_x <= 0.1:
                continue

            perspective_scale = 1 / robot_x
            pixel_x = vanishing_point.x() - (robot_y * horizontal_stretch * perspective_scale)
            pixel_y = vanishing_point.y() + (vertical_stretch * perspective_scale)
            
            if pixel_y > img_height:
                continue
            path.append(QPointF(pixel_x, pixel_y))

        painter.drawPolyline(path)


    def draw_robot_on_pixmap(self, pixmap: QPixmap, robot_pose: tuple, costmap):
        """Draws the robot's pose as a quiver on the costmap pixmap."""
        painter = QPainter(pixmap)
        
        resolution = costmap.get_resolution()
        origin_x = costmap.get_origin_x()
        origin_y = costmap.get_origin_y()

        def world_to_pixel(wx, wy):
            px = (wx - origin_x) / resolution
            py = pixmap.height() - ((wy - origin_y) / resolution)
            return px, py

        robot_px, robot_py = world_to_pixel(robot_pose[0], robot_pose[1])
        
        painter.setPen(QPen(Qt.GlobalColor.red, 2))
        painter.setBrush(QBrush(Qt.GlobalColor.red))
        painter.drawEllipse(QPointF(robot_px, robot_py), 3, 3)

        robot_theta = robot_pose[2]
        arrow_length_px = 5
        dx = arrow_length_px * np.cos(robot_theta)
        dy = -arrow_length_px * np.sin(robot_theta)

        painter.setPen(QPen(Qt.GlobalColor.red, 3, Qt.PenStyle.SolidLine))
        painter.drawLine(int(robot_px), int(robot_py), int(robot_px + dx), int(robot_py + dy))
        # No painter.end(), managed by QPixmap lifetime

    def numpy_to_qimage(self, array: np.ndarray) -> QImage:
        """Converts a 2D numpy array (grayscale) to a QImage, optimized for speed."""
        if array.ndim != 2:
            raise ValueError("Input array must be 2D")

        # Normalize the array to 0-255
        min_val, max_val = array.min(), array.max()
        if min_val == max_val:
            norm_array = np.zeros(array.shape, dtype=np.uint8)
        else:
            norm_array = ((array - min_val) * (255 / (max_val - min_val))).astype(np.uint8)

        # Invert the colormap to match 'gray_r' (black for high cost)
        norm_array = 255 - norm_array

        # --- Fast NumPy to QImage conversion ---
        height, width = norm_array.shape
        # Create an RGB image by repeating the grayscale values across 3 channels
        rgb_array = np.stack([norm_array] * 3, axis=-1)
        
        # Ensure the numpy array's memory layout is contiguous
        rgb_array_contiguous = np.require(rgb_array, np.uint8, 'C')

        bytes_per_line = 3 * width
        q_image = QImage(rgb_array_contiguous.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Creating a QPixmap from this QImage makes a deep copy of the data,
        # ensuring it's safe even after the numpy array goes out of scope.
        return q_image

    def closeEvent(self, event):
        """Properly stop the worker thread on close."""
        print("Closing application...")
        self.worker.stop()
        self.thread.quit()
        self.thread.wait(500) # Wait up to 500ms for thread to finish
        event.accept()

def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    window = VisualizerWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 