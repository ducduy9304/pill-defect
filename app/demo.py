# Live pill inspection: camera view, choose the SKU, then check the pill in two steps (main.py):
# 1. surface (AnomalyDINO), 2. dimensions (length, width, holes). CHECK checks the selected step and shows
# the result picture, OK / NG and what is wrong; it stays on that step: only clicking a step selects
# another one. CHECK on step 1 starts a new pill. The pill is OK only when both steps are OK.
#
# Each step has its own camera exposure and gain, from configs/<sku>.json, applied when the step is selected.
# They can be changed in the panel and saved as the SKU's default.
#
# Save image / Save process write the shown result to inspections/<sku>/<date>/ (the photo and the result
# picture / the pictures of the process); Export data writes every check of this session to a CSV.
# Without a camera (no Hikrobot SDK, no camera found, or --no-camera) the app starts offline: click the
# camera view to open a saved photo, e.g. from data/raw/.
#
# Look: the dark theme of DemoTemplate_thadosoft (app/style.py).
#
# IMPORTANT: close MVS first (exclusive access).
#
# Run:  uv run python app/demo.py
#       uv run python app/demo.py --sku sku01 --camera 1
#       uv run python app/demo.py --no-camera
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
                               QGroupBox, QHBoxLayout, QLabel, QMainWindow, QPushButton, QSizePolicy,
                               QVBoxLayout, QWidget)

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR.parent))

import main as pipeline  # noqa: E402  (this file has its own main())
from style import APP_STYLESHEET, IDLE_COLOR, NG_COLOR, OK_COLOR  # noqa: E402

try:  # the Hikrobot SDK (hik_sdk/MvImport) is only on the inspection PC
    from camera import MvCamera, apply_camera_settings, close_camera, grab_bgr, open_camera
except ImportError as error:
    MvCamera = None
    CAMERA_IMPORT_ERROR = error

SIDE_PANEL_WIDTH = 400
LOGO_WIDTH = 64  # ~60 px tall: the window must fit a 1080 px screen
RESULT_FRAME_WIDTH = 4  # px of the result view's frame: green when OK, red when NG
INSPECTIONS = APP_DIR.parent / "inspections"
RECORD_FIELDS = ["time", "sku", "pill", "step", "verdict", "summary", "faults", "ms"]


def to_pixmap(color, size):
    # BGR (or grayscale) numpy image -> Qt pixmap that fits in `size`, aspect ratio kept.
    if color.ndim == 2:
        color = cv2.cvtColor(color, cv2.COLOR_GRAY2BGR)
    color = np.ascontiguousarray(color)  # QImage needs one block, a crop is a strided view
    height, width = color.shape[:2]
    image = QImage(color.data, width, height, color.strides[0], QImage.Format.Format_BGR888)
    # fromImage copies the pixels, so `color` may be freed afterwards.
    pixmap = QPixmap.fromImage(image)
    return pixmap.scaled(size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)


class ClickableLabel(QLabel):
    clicked = Signal()

    def mousePressEvent(self, event):
        self.clicked.emit()


class Grabber(QThread):
    # Grabs camera frames off the UI thread: waiting for a frame (1.7 s on a 100 Mbps link) froze the window.
    frame = Signal(object)

    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.running = True

    def run(self):
        while self.running:
            try:
                color, _ = grab_bgr(self.camera["cam"], self.camera["out_frame"])
            except Exception as error:  # e.g. a timeout: try again
                print(error)
                continue
            self.frame.emit(color)


class StepRow(QPushButton):
    # One step in the side panel: its name and its verdict.
    def __init__(self, step):
        super().__init__()
        self.setObjectName("stepRow")
        self.setCheckable(True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        name = QLabel(pipeline.STEPS[step]["name"])
        name.setObjectName("stepName")
        name.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout.addWidget(name, 1)
        self.status = QLabel("–")
        self.status.setObjectName("stepStatus")
        self.status.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout.addWidget(self.status)

    def show_status(self, text, color):
        self.status.setText(text)
        self.status.setStyleSheet("color: %s;" % color)


class LabelCheckWindow(QMainWindow):
    def __init__(self, camera, start_sku):
        super().__init__()
        self.camera = camera
        self.prepared = {}          # sku -> [what each step needs], made once
        self.latest_frame = None    # BGR photo that CHECK looks at
        self.results = [None] * len(pipeline.STEPS)  # (result, photo) of each step of the current pill
        self.step = 0               # the selected step: CHECK checks it
        self.label_id = None        # time of the current pill's first check, names its files
        self.shown = None           # (step, result, photo) of the result on screen, for Save image / process
        self.result_picture = None  # the picture on screen, kept to fit it again on resize
        self.records = []           # every check of this session, for Export data
        self.frame_count = 0
        self.live = camera is not None

        self.setWindowTitle("AITS - Pill inspection")
        self.setWindowIcon(QIcon(str(APP_DIR / "aits_logo.png")))
        self.resize(1600, 960)
        central = QWidget()
        central.setObjectName("centralwidget")
        root = QHBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 12)
        root.setSpacing(16)
        root.addLayout(self.build_views(), 1)
        root.addWidget(self.build_side_panel(start_sku), 0)
        self.setCentralWidget(central)
        self.setStyleSheet(APP_STYLESHEET)

        if self.live:
            self.live_caption.setText("CAMERA")
        else:
            self.live_caption.setText("NO CAMERA")
            self.live_view.setText("Click to open an image")
        if self.live:
            self.grabber = Grabber(camera)
            self.grabber.frame.connect(self.show_frame)
            self.grabber.start()
        self.on_sku_changed()

    def build_views(self):
        # Title, then the camera and the result next to each other.
        column = QVBoxLayout()
        column.setSpacing(10)
        title = QLabel("PILL INSPECTION")
        title.setObjectName("titleLabel")
        subtitle = QLabel("Silicone cord-lock pills")
        subtitle.setObjectName("subtitleLabel")
        column.addWidget(title)
        column.addWidget(subtitle)

        views = QHBoxLayout()
        views.setSpacing(12)
        self.live_caption = QLabel()
        self.live_caption.setObjectName("viewCaption")
        self.live_view = ClickableLabel("No signal")
        self.live_view.clicked.connect(self.open_image)
        self.result_caption = QLabel("RESULT")
        self.result_caption.setObjectName("viewCaption")
        self.result_view = QLabel("")
        for caption, view, name in [(self.live_caption, self.live_view, "raw"),
                                    (self.result_caption, self.result_view, "result")]:
            view.setObjectName(name)
            view.setAlignment(Qt.AlignmentFlag.AlignCenter)
            view.setWordWrap(True)
            view.setMinimumSize(320, 280)
            view.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
            view_column = QVBoxLayout()
            view_column.setSpacing(6)
            view_column.addWidget(caption)
            view_column.addWidget(view, 1)
            views.addLayout(view_column, 1)
        column.addLayout(views, 1)
        return column

    def build_side_panel(self, start_sku):
        # SKU, the two steps, CHECK and the saving buttons, camera settings, the pill's verdict, the logo.
        panel = QFrame()
        panel.setObjectName("sidePanel")
        panel.setFixedWidth(SIDE_PANEL_WIDTH)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        sku_group = QGroupBox("SKU")
        sku_layout = QVBoxLayout(sku_group)
        self.sku_box = QComboBox()
        self.sku_box.addItems(pipeline.list_skus())
        self.sku_box.setCurrentText(start_sku)
        self.sku_box.currentTextChanged.connect(self.on_sku_changed)
        sku_layout.addWidget(self.sku_box)
        layout.addWidget(sku_group)

        steps_group = QGroupBox("INSPECTION STEPS")
        steps_layout = QVBoxLayout(steps_group)
        steps_layout.setSpacing(6)
        self.step_rows = []
        for step in range(len(pipeline.STEPS)):
            row = StepRow(step)
            row.clicked.connect(lambda _, s=step: self.select_step(s))
            steps_layout.addWidget(row)
            self.step_rows.append(row)
        layout.addWidget(steps_group)

        check_button = QPushButton("CHECK")
        check_button.setObjectName("checkbtn")
        check_button.clicked.connect(self.check)
        layout.addWidget(check_button)
        save_row = QHBoxLayout()
        for text, slot in (("Save image", self.save_image), ("Save process", self.save_process),
                           ("Export data", self.export_data)):
            button = QPushButton(text)
            button.clicked.connect(slot)
            save_row.addWidget(button)
        layout.addLayout(save_row)
        self.save_label = QLabel("")
        self.save_label.setObjectName("saveLabel")
        self.save_label.setWordWrap(True)
        layout.addWidget(self.save_label)

        self.camera_group = QGroupBox("CAMERA SETTINGS")
        camera_layout = QVBoxLayout(self.camera_group)
        form = QFormLayout()
        self.exposure_box = QDoubleSpinBox()
        self.exposure_box.setRange(20, 1000000)
        self.exposure_box.setDecimals(0)
        self.exposure_box.setSingleStep(500)
        self.exposure_box.setSuffix(" µs")
        self.gain_box = QDoubleSpinBox()
        self.gain_box.setRange(0, 24)
        self.gain_box.setDecimals(1)
        self.gain_box.setSingleStep(0.5)
        self.gain_box.setSuffix(" dB")
        form.addRow("Exposure", self.exposure_box)
        form.addRow("Gain", self.gain_box)
        camera_layout.addLayout(form)
        camera_row = QHBoxLayout()
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.apply_settings)
        apply_button.setEnabled(self.camera is not None)
        default_button = QPushButton("Save as SKU default")
        default_button.clicked.connect(self.save_settings)
        camera_row.addWidget(apply_button)
        camera_row.addWidget(default_button)
        camera_layout.addLayout(camera_row)
        layout.addWidget(self.camera_group)

        verdict_group = QGroupBox("PILL VERDICT")
        verdict_layout = QVBoxLayout(verdict_group)
        self.verdict_label = QLabel("–")
        self.verdict_label.setObjectName("verdictLabel")
        self.verdict_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.faults_label = QLabel("")
        self.faults_label.setObjectName("faultsLabel")
        self.faults_label.setWordWrap(True)
        self.faults_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_label = QLabel("")
        self.time_label.setObjectName("procTime")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        verdict_layout.addWidget(self.verdict_label)
        verdict_layout.addWidget(self.faults_label)
        verdict_layout.addWidget(self.time_label)
        layout.addWidget(verdict_group)

        # The logo at the bottom of the panel.
        layout.addStretch(1)
        logo = QLabel()
        logo.setObjectName("logoLabel")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setPixmap(QPixmap(str(APP_DIR / "aits_logo.png")).scaledToWidth(
            LOGO_WIDTH, Qt.TransformationMode.SmoothTransformation))
        layout.addWidget(logo)
        return panel

    def current_sku(self):
        return self.sku_box.currentText()

    def frame_result(self, color=None):
        # The result view's frame in the verdict's color, or the plain frame.
        style = "QLabel#result { border: %dpx solid %s; }" % (RESULT_FRAME_WIDTH, color) if color else ""
        self.result_view.setStyleSheet(style)

    def show_text(self, text):
        self.result_picture = None
        self.frame_result()
        self.result_view.clear()
        self.result_view.setText(text)

    def show_waiting(self, step):
        name = pipeline.STEPS[step]["name"]
        self.shown = None
        self.result_caption.setText("RESULT · %s" % name)
        self.show_text("Waiting for CHECK: %s" % name)

    def show_result(self, step):
        # A step's result picture in a green (OK) or red (NG) frame.
        name = pipeline.STEPS[step]["name"]
        result, photo = self.results[step]
        self.shown = (step, result, photo)
        self.result_caption.setText("RESULT · %s · %s · %s" % (name, result["verdict"], result["summary"]))
        self.frame_result(OK_COLOR if result["verdict"] == "OK" else NG_COLOR)
        self.result_picture = result["image"]
        self.result_view.setPixmap(to_pixmap(self.result_picture, self.result_view.size()))

    def show_label_verdict(self):
        # NG as soon as one step is NG; OK only when both are OK.
        results = [r[0] if r else None for r in self.results]
        verdicts = [r["verdict"] for r in results if r is not None]
        faults = ["%s: %s" % (pipeline.STEPS[s]["name"], f) for s, r in enumerate(results) if r for f in r["faults"]]
        if "NG" in verdicts:
            text, color = "NG", NG_COLOR
        elif len(verdicts) == len(results):
            text, color = "OK", OK_COLOR
        else:
            text, color = "–", IDLE_COLOR
        self.verdict_label.setText(text)
        self.verdict_label.setStyleSheet("color: %s;" % color)
        self.faults_label.setText("\n".join(faults))
        for row, result in zip(self.step_rows, results):
            if result is None:
                row.show_status("–", IDLE_COLOR)
            else:
                row.show_status(result["verdict"], OK_COLOR if result["verdict"] == "OK" else NG_COLOR)

    def select_step(self, step, show=True):
        # Mark the step, load and apply its camera settings, clear the result view until its CHECK.
        self.step = step
        for number, row in enumerate(self.step_rows):
            row.setChecked(number == step)
        settings = pipeline.camera_settings(self.current_sku(), step)
        self.exposure_box.setValue(settings["ExposureTime"])
        self.gain_box.setValue(settings["Gain"])
        self.camera_group.setTitle("CAMERA SETTINGS · %s" % pipeline.STEPS[step]["name"].upper())
        if self.camera is not None:
            apply_camera_settings(self.camera, settings)
        if show:
            self.show_waiting(step)

    def apply_settings(self):
        # The exposure and gain in the boxes, on the camera now (not saved).
        if self.camera is not None:
            apply_camera_settings(self.camera, {"ExposureTime": self.exposure_box.value(), "Gain": self.gain_box.value()})
            self.save_label.setText("Camera: exposure %.0f µs, gain %.1f dB" % (self.exposure_box.value(), self.gain_box.value()))

    def save_settings(self):
        # The exposure and gain in the boxes as the default of this SKU's step.
        settings = pipeline.camera_settings(self.current_sku(), self.step)
        settings.update({"ExposureTime": self.exposure_box.value(), "Gain": self.gain_box.value()})
        pipeline.save_camera_settings(self.current_sku(), self.step, settings)
        self.apply_settings()
        self.save_label.setText("Saved as the default of %s, %s" % (self.current_sku(), pipeline.STEPS[self.step]["name"]))

    def on_sku_changed(self):
        # The SKU's model and size calibration, loaded once, then a new pill.
        sku = self.current_sku()
        if sku not in self.prepared:
            self.show_text("Loading %s ..." % sku)
            QApplication.processEvents()  # draw the message before the few seconds of work
            try:
                loaded = pipeline.load_sku(sku)
                self.prepared[sku] = [pipeline.prepare(loaded, step) for step in range(len(pipeline.STEPS))]
            except Exception as error:  # e.g. no checkpoint or good photos for this SKU
                self.show_text("Cannot load %s:\n%s" % (sku, error))
                return
        self.new_label()

    def new_label(self):
        self.results = [None] * len(pipeline.STEPS)
        self.label_id = None
        self.time_label.setText("")
        self.show_label_verdict()
        self.select_step(0)

    def show_frame(self, color):
        self.frame_count += 1
        self.latest_frame = color
        height, width = self.latest_frame.shape[:2]
        self.live_caption.setText("CAMERA · %d x %d · frame #%d" % (width, height, self.frame_count))
        self.live_view.setPixmap(to_pixmap(self.latest_frame, self.live_view.size()))

    def open_image(self):
        # Offline only: a saved photo instead of the camera.
        if self.camera is not None:
            return
        start_folder = pipeline.ROOT / "data" / "raw"
        path, _ = QFileDialog.getOpenFileName(self, "Open image", str(start_folder), "Images (*.bmp *.png *.jpg)")
        if path:
            self.load_photo(Path(path))

    def load_photo(self, path):
        self.latest_frame = cv2.imread(str(path))
        if self.latest_frame is None:
            self.show_text("Cannot read %s" % path.name)
            return
        height, width = self.latest_frame.shape[:2]
        self.live_caption.setText("IMAGE · %s · %d x %d" % (path.name, width, height))
        self.live_view.setPixmap(to_pixmap(self.latest_frame, self.live_view.size()))

    def check(self):
        sku = self.current_sku()
        if sku not in self.prepared:
            return
        if self.latest_frame is None:
            self.show_text("No image yet")
            return
        if self.step == 0:  # step 1 is always the first shot of a pill
            self.new_label()
        photo = self.latest_frame.copy()
        start = time.perf_counter()
        try:
            result = pipeline.check(photo, self.prepared[sku][self.step])
        except Exception as error:
            self.show_text("Check failed (no pill found?):\n%s" % error)
            return
        ms = (time.perf_counter() - start) * 1000
        self.time_label.setText("%.0f ms" % ms)
        if self.label_id is None:
            self.label_id = time.strftime("%Y%m%d_%H%M%S")
        self.results[self.step] = (result, photo)
        self.records.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "sku": sku, "pill": self.label_id,
                             "step": pipeline.STEPS[self.step]["name"], "verdict": result["verdict"],
                             "summary": result["summary"],
                             "faults": "; ".join(result["faults"]), "ms": "%.0f" % ms})
        self.show_label_verdict()
        self.show_result(self.step)

    def saved_stem(self):
        # inspections/<sku>/<date>/<label>_<step>_<verdict> of the result on screen.
        step, result, _ = self.shown
        folder = INSPECTIONS / self.current_sku() / self.label_id[:8]
        folder.mkdir(parents=True, exist_ok=True)
        return folder / ("%s_%d_%s" % (self.label_id, step + 1, result["verdict"]))

    def save_image(self):
        # The photo and the result picture on screen.
        if self.shown is None:
            self.save_label.setText("Nothing to save: press CHECK first")
            return
        stem = self.saved_stem()
        cv2.imwrite(str(stem) + "_photo.png", self.shown[2])
        cv2.imwrite(str(stem) + "_result.jpg", self.shown[1]["image"])
        self.save_label.setText("Saved %s_photo.png, _result.jpg" % stem.relative_to(INSPECTIONS.parent))

    def save_process(self):
        # The pictures of how the result on screen was found.
        if self.shown is None:
            self.save_label.setText("Nothing to save: press CHECK first")
            return
        stem = self.saved_stem()
        cv2.imwrite(str(stem) + "_process.jpg", self.shown[1]["process"])
        self.save_label.setText("Saved %s_process.jpg" % stem.relative_to(INSPECTIONS.parent))

    def export_data(self):
        # Every check of this session, one line each.
        if not self.records:
            self.save_label.setText("Nothing to export: press CHECK first")
            return
        INSPECTIONS.mkdir(exist_ok=True)
        default = INSPECTIONS / ("inspections_%s.csv" % time.strftime("%Y%m%d_%H%M%S"))
        path, _ = QFileDialog.getSaveFileName(self, "Export data", str(default), "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=RECORD_FIELDS)
            writer.writeheader()
            writer.writerows(self.records)
        self.save_label.setText("Exported %d checks to %s" % (len(self.records), Path(path).name))

    def closeEvent(self, event):
        # The grabber must stop before main() closes the camera.
        if self.live:
            self.grabber.running = False
            self.grabber.wait()
        super().closeEvent(event)

    def resizeEvent(self, event):
        # The live view is redrawn by the next frame; the other pictures must be fitted again here.
        super().resizeEvent(event)
        if self.result_picture is not None:
            self.result_view.setPixmap(to_pixmap(self.result_picture, self.result_view.size()))
        if not self.live and self.latest_frame is not None:
            self.live_view.setPixmap(to_pixmap(self.latest_frame, self.live_view.size()))


def main():
    parser = argparse.ArgumentParser(description="Live pill inspection: surface and dimensions.")
    parser.add_argument("--sku", help="SKU to start on, e.g. sku01 (default: the first one)")
    parser.add_argument("--camera", type=int, default=0, help="camera number (default 0)")
    parser.add_argument("--no-camera", action="store_true", help="start offline, check saved photos only")
    args = parser.parse_args()

    skus = pipeline.list_skus()
    if not skus:
        raise SystemExit("no SKU config in %s" % pipeline.CONFIGS)
    start_sku = args.sku if args.sku else skus[0]

    app = QApplication(sys.argv)
    camera = None
    use_camera = not args.no_camera and MvCamera is not None
    if not args.no_camera and MvCamera is None:
        print("no camera SDK (%s): starting offline, click the camera view to open an image" % CAMERA_IMPORT_ERROR)
    if use_camera:
        MvCamera.MV_CC_Initialize()
        try:
            camera = open_camera(args.camera)
        except Exception as error:
            # Working without the camera (e.g. at home) is a normal case: offline mode.
            print("no camera (%s): starting offline, click the camera view to open an image" % error)
    try:
        window = LabelCheckWindow(camera, start_sku)
        window.show()
        app.exec()
    finally:
        # Runs after the window is closed too, so the camera settings are always put back.
        if camera is not None:
            close_camera(camera)
        if use_camera:
            MvCamera.MV_CC_Finalize()


if __name__ == "__main__":
    main()
