# -*- coding: utf-8 -*-
"""
Planktoscope Microscope Image Segmentation Preview & Batch Tool
Author: brigchen@xmu.edu.cn

Interactive tool for previewing and tuning particle detection parameters,
then batch-cutting all images in a folder.
"""
import os
import sys
import json
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QApplication, QFrame, QGroupBox, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QFileDialog, QDoubleSpinBox,
    QSpinBox, QStatusBar, QSizePolicy, QMessageBox, QProgressBar,
    QComboBox, QProgressDialog, QCheckBox
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon, QPainter, QPen, QColor, QPolygon
from PyQt5.QtCore import Qt, QSize, QPoint, QRectF, QTimer

IMG_FORMATS = {"tif", "tiff", "jpg", "jpeg", "png", "bmp", "gif", "jfif"}
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "planktoscope.cfg")


BINARIZE_METHODS = ["Adaptive Threshold", "Otsu", "Global Threshold", "Canny Edge"]


def detect_contours(image, th_size, blur_kernel, morph_iter,
                    method="Adaptive Threshold",
                    adapt_block=25, adapt_C=3,
                    global_thresh=128,
                    canny_low=50, canny_high=150):
    """Run the detection pipeline and return (rects, binarized_image).

    rects: list of (x, y, w, h) tuples for each detected particle.
    binarized_image: the binary image after morphological close (uint8, 0/255).
    """
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (blur_kernel, blur_kernel), sigmaX=10)

    if method == "Adaptive Threshold":
        img_bin = cv2.adaptiveThreshold(
            img_blur, 175, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, adapt_block, adapt_C
        )
    elif method == "Otsu":
        _, img_bin = cv2.threshold(
            img_blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
    elif method == "Global Threshold":
        _, img_bin = cv2.threshold(
            img_blur, global_thresh, 255, cv2.THRESH_BINARY_INV
        )
    elif method == "Canny Edge":
        img_bin = cv2.Canny(img_blur, canny_low, canny_high, apertureSize=3, L2gradient=True)
    else:
        img_bin = cv2.adaptiveThreshold(
            img_blur, 175, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, adapt_block, adapt_C
        )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_ex = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel, iterations=morph_iter)
    cnts, _ = cv2.findContours(img_ex, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    rects = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < np.pi * (th_size / 2) ** 2:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 0.8 * image.shape[0] * image.shape[1]:
            continue
        rects.append((x, y, w, h))
    return rects, img_ex


def cut_with_rects(image, rects):
    """Crop and return sub-images for each (x, y, w, h) rect."""
    imgs = []
    for x, y, w, h in rects:
        imgs.append(image[y:y + h, x:x + w].copy())
    return imgs


def draw_rects_on_image(image, rects):
    """Draw green bounding boxes on a copy of the image and return it."""
    vis = image.copy()
    for x, y, w, h in rects:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return vis


def imread_unicode(path):
    """Read image file supporting non-ASCII (e.g. Chinese) paths."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def imwrite_unicode(path, img):
    """Write image file supporting non-ASCII (e.g. Chinese) paths."""
    try:
        ext = os.path.splitext(path)[1]
        result, buf = cv2.imencode(ext, img)
        if result:
            buf.tofile(path)
            return True
    except Exception:
        pass
    return False


def cv2_to_qpixmap(image):
    """Convert a BGR cv2 image to QPixmap."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


class NavArrowButton(QPushButton):
    """Semi-transparent triangular arrow button overlaid on the image."""
    def __init__(self, direction, parent=None):
        super().__init__(parent)
        self.direction = direction  # "left" or "right"
        self.setFixedSize(50, 100)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("background: transparent; border: none;")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Semi-transparent background
        bg_color = QColor(0, 0, 0, 80)
        painter.setBrush(bg_color)
        painter.setPen(Qt.NoPen)

        if self.direction == "left":
            bg_rect = QRectF(0, 0, w, h)
            painter.drawRect(bg_rect)
            # Triangle pointing right (indicates go forward/back visually)
            triangle = QPolygon([
                QPoint(10, h // 2),
                QPoint(w - 15, h // 2 - 20),
                QPoint(w - 15, h // 2 + 20),
            ])
        else:
            bg_rect = QRectF(0, 0, w, h)
            painter.drawRect(bg_rect)
            # Triangle pointing left
            triangle = QPolygon([
                QPoint(w - 10, h // 2),
                QPoint(15, h // 2 - 20),
                QPoint(15, h // 2 + 20),
            ])

        painter.setBrush(QColor(255, 255, 255, 200))
        painter.drawPolygon(triangle)
        painter.end()


class PlanktoscopeSegmentViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image_folder = ""
        self.output_folder = ""
        self.image_files = []
        self.current_index = 0
        self.current_image = None  # BGR cv2 image
        self.current_rects = []

        # Default parameters
        self.th_size = 10.0
        self.blur_kernel = 5
        self.adapt_block = 25
        self.adapt_C = 3
        self.morph_iter = 5
        self.global_thresh = 128
        self.canny_low = 50
        self.canny_high = 150

        self._defer_show_image = False

        self._setup_ui()
        self._load_settings()

    # ------------------------------------------------------------------ UI
    def _setup_ui(self):
        self.setWindowTitle("Planktoscope显微图像切割")
        desktop = QApplication.desktop()
        self.resize(desktop.width(), desktop.height())
        self.setMinimumSize(QSize(1000, 700))

        font = QFont("Times New Roman", 11)
        self.setFont(font)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(5, 5, 5, 5)
        root_layout.setSpacing(5)

        # ---- Left panel (controls) ----
        left_panel = QFrame()
        left_panel.setMinimumWidth(260)
        left_panel.setMaximumWidth(320)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(8)

        # Folder selection
        grp_folder = QGroupBox("Image Folder")
        lay_folder = QVBoxLayout(grp_folder)
        self.btn_open = QPushButton("Open Folder")
        self.btn_open.clicked.connect(self._open_folder)
        lay_folder.addWidget(self.btn_open)
        self.label_folder = QLabel("No folder selected")
        self.label_folder.setWordWrap(True)
        lay_folder.addWidget(self.label_folder)
        left_layout.addWidget(grp_folder)

        # Parameters
        grp_params = QGroupBox("Detection Parameters")
        lay_params = QVBoxLayout(grp_params)

        # Binarization method selector
        lay_params.addWidget(QLabel("Binarization method:"))
        self.combo_method = QComboBox()
        self.combo_method.addItems(BINARIZE_METHODS)
        self.combo_method.currentTextChanged.connect(self._on_method_changed)
        lay_params.addWidget(self.combo_method)

        # Common parameters
        lay_params.addWidget(QLabel("Min particle size (px):"))
        self.spin_th = QDoubleSpinBox()
        self.spin_th.setRange(1.0, 500.0)
        self.spin_th.setDecimals(0)
        self.spin_th.setValue(self.th_size)
        self.spin_th.setSingleStep(1)
        self.spin_th.valueChanged.connect(self._on_param_changed)
        lay_params.addWidget(self.spin_th)

        lay_params.addWidget(QLabel("Blur kernel (odd):"))
        self.spin_blur = QSpinBox()
        self.spin_blur.setRange(1, 31)
        self.spin_blur.setSingleStep(2)
        self.spin_blur.setValue(self.blur_kernel)
        self.spin_blur.valueChanged.connect(self._on_param_changed)
        lay_params.addWidget(self.spin_blur)

        # Adaptive-specific parameters
        self.label_block = QLabel("Adaptive block size (odd):")
        self.spin_block = QSpinBox()
        self.spin_block.setRange(3, 99)
        self.spin_block.setSingleStep(2)
        self.spin_block.setValue(self.adapt_block)
        self.spin_block.valueChanged.connect(self._on_param_changed)
        self.label_C = QLabel("Adaptive constant C:")
        self.spin_C = QSpinBox()
        self.spin_C.setRange(0, 50)
        self.spin_C.setValue(self.adapt_C)
        self.spin_C.valueChanged.connect(self._on_param_changed)
        lay_params.addWidget(self.label_block)
        lay_params.addWidget(self.spin_block)
        lay_params.addWidget(self.label_C)
        lay_params.addWidget(self.spin_C)

        # Global threshold parameter
        self.label_global_thresh = QLabel("Threshold value:")
        self.spin_global_thresh = QSpinBox()
        self.spin_global_thresh.setRange(0, 255)
        self.spin_global_thresh.setValue(self.global_thresh)
        self.spin_global_thresh.valueChanged.connect(self._on_param_changed)
        lay_params.addWidget(self.label_global_thresh)
        lay_params.addWidget(self.spin_global_thresh)

        # Canny parameters
        self.label_canny_low = QLabel("Canny low threshold:")
        self.spin_canny_low = QSpinBox()
        self.spin_canny_low.setRange(0, 500)
        self.spin_canny_low.setValue(self.canny_low)
        self.spin_canny_low.valueChanged.connect(self._on_param_changed)
        self.label_canny_high = QLabel("Canny high threshold:")
        self.spin_canny_high = QSpinBox()
        self.spin_canny_high.setRange(0, 500)
        self.spin_canny_high.setValue(self.canny_high)
        self.spin_canny_high.valueChanged.connect(self._on_param_changed)
        lay_params.addWidget(self.label_canny_low)
        lay_params.addWidget(self.spin_canny_low)
        lay_params.addWidget(self.label_canny_high)
        lay_params.addWidget(self.spin_canny_high)

        # Common: morphology
        lay_params.addWidget(QLabel("Morphology close iterations:"))
        self.spin_morph = QSpinBox()
        self.spin_morph.setRange(1, 30)
        self.spin_morph.setValue(self.morph_iter)
        self.spin_morph.valueChanged.connect(self._on_param_changed)
        lay_params.addWidget(self.spin_morph)

        left_layout.addWidget(grp_params)

        # Set initial visibility
        self._on_method_changed(self.combo_method.currentText())

        # Info
        grp_info = QGroupBox("Detection Result")
        lay_info = QVBoxLayout(grp_info)
        self.label_count = QLabel("Particles: 0")
        self.label_count.setFont(QFont("Arial", 12, QFont.Bold))
        lay_info.addWidget(self.label_count)
        self.label_image_info = QLabel("")
        lay_info.addWidget(self.label_image_info)
        left_layout.addWidget(grp_info)

        # Show binarized image toggle
        self.chk_show_bin = QCheckBox("Show binarized image")
        self.chk_show_bin.stateChanged.connect(self._on_param_changed)
        left_layout.addWidget(self.chk_show_bin)

        # Batch button
        self.btn_batch = QPushButton("Batch Cut All Images")
        self.btn_batch.setMinimumHeight(45)
        self.btn_batch.setFont(QFont("Arial", 12))
        self.btn_batch.clicked.connect(self._batch_cut)
        left_layout.addWidget(self.btn_batch)

        left_layout.addStretch()

        # ---- Center panel (image display with overlay arrows) ----
        self.center_panel = QFrame()
        self.center_panel.setStyleSheet("background-color: #2b2b2b;")
        center_layout = QVBoxLayout(self.center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        self.label_image = QLabel("Open a folder to start")
        self.label_image.setAlignment(Qt.AlignCenter)
        self.label_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_image.setStyleSheet("background-color: #2b2b2b; color: #aaa;")
        center_layout.addWidget(self.label_image)

        # Page info bar at bottom
        self.label_page = QLabel("")
        self.label_page.setAlignment(Qt.AlignCenter)
        self.label_page.setStyleSheet("color: white; background: rgba(0,0,0,120); padding: 2px 10px;")
        self.label_page.setFixedHeight(24)
        center_layout.addWidget(self.label_page)

        # Arrow buttons as children of center_panel, positioned in resizeEvent
        self.btn_prev = NavArrowButton("left", self.center_panel)
        self.btn_prev.clicked.connect(self._prev_image)

        self.btn_next = NavArrowButton("right", self.center_panel)
        self.btn_next.clicked.connect(self._next_image)

        root_layout.addWidget(left_panel)
        root_layout.addWidget(self.center_panel, 1)

        # Status bar
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

    # ------------------------------------------------------------------ Folder
    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Microscope Image Folder", "./")
        if not folder:
            return
        self.image_folder = folder
        self.label_folder.setText(folder)
        self._load_file_list()
        if self.image_files:
            self.current_index = 0
            self._show_current_image()
        else:
            self.label_image.setText("No images found in folder")
            self.label_image.setStyleSheet("background-color: #2b2b2b; color: #aaa;")

    def _load_file_list(self):
        self.image_files = sorted([
            f for f in os.listdir(self.image_folder)
            if f.split(".")[-1].lower() in IMG_FORMATS
        ])

    # ------------------------------------------------------------------ Navigation
    def _show_current_image(self):
        if not self.image_files:
            return
        fname = self.image_files[self.current_index]
        path = os.path.join(self.image_folder, fname)
        img = imread_unicode(path)
        if img is None:
            import imageio
            try:
                tmp = imageio.mimread(path)
                if tmp:
                    arr = np.array(tmp[0])
                    img = arr[:, :, 0:3]
            except Exception:
                img = None
        if img is None:
            self.label_image.setText("Failed to load: %s" % fname)
            return

        self.current_image = img
        self._run_detection_and_display()

        h, w = img.shape[:2]
        self.label_image_info.setText("%s\n%d x %d" % (fname, w, h))
        self.label_page.setText("%d / %d" % (self.current_index + 1, len(self.image_files)))
        self.statusbar.showMessage(path)

    def _run_detection_and_display(self):
        if self.current_image is None:
            return
        # Read parameters
        th = self.spin_th.value()
        bk = self.spin_blur.value()
        if bk % 2 == 0:
            bk += 1
        mi = self.spin_morph.value()
        method = self.combo_method.currentText()

        rects, img_bin = detect_contours(
            self.current_image, th, bk, mi, method,
            adapt_block=self.spin_block.value() if self.spin_block.value() % 2 == 1 else self.spin_block.value() + 1,
            adapt_C=self.spin_C.value(),
            global_thresh=self.spin_global_thresh.value(),
            canny_low=self.spin_canny_low.value(),
            canny_high=self.spin_canny_high.value()
        )
        self.current_rects = rects
        self.label_count.setText("Particles: %d" % len(rects))

        if self.chk_show_bin.isChecked():
            # Convert single-channel bin to BGR for drawing rects
            base_img = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)
        else:
            base_img = self.current_image

        vis = draw_rects_on_image(base_img, rects)
        pixmap = cv2_to_qpixmap(vis)
        scaled = pixmap.scaled(
            self.label_image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.label_image.setPixmap(scaled)
        self.label_image.setStyleSheet("")

    def _prev_image(self):
        if not self.image_files:
            return
        self.current_index = max(0, self.current_index - 1)
        self._show_current_image()

    def _next_image(self):
        if not self.image_files:
            return
        self.current_index = min(len(self.image_files) - 1, self.current_index + 1)
        self._show_current_image()

    # ------------------------------------------------------------------ Params
    def _on_method_changed(self, method):
        is_adaptive = (method == "Adaptive Threshold")
        is_global = (method == "Global Threshold")
        is_canny = (method == "Canny Edge")
        is_otsu = (method == "Otsu")

        self.label_block.setVisible(is_adaptive)
        self.spin_block.setVisible(is_adaptive)
        self.label_C.setVisible(is_adaptive)
        self.spin_C.setVisible(is_adaptive)

        self.label_global_thresh.setVisible(is_global)
        self.spin_global_thresh.setVisible(is_global)

        self.label_canny_low.setVisible(is_canny)
        self.spin_canny_low.setVisible(is_canny)
        self.label_canny_high.setVisible(is_canny)
        self.spin_canny_high.setVisible(is_canny)

        self._run_detection_and_display()

    def _on_param_changed(self):
        self._run_detection_and_display()

    # ------------------------------------------------------------------ Batch
    def _batch_cut(self):
        if not self.image_folder or not self.image_files:
            QMessageBox.warning(self, "Error", "No images loaded. Open a folder first.")
            return

        output = QFileDialog.getExistingDirectory(self, "Select Output Folder for Cropped Images", "./")
        if not output:
            return
        self.output_folder = output

        th = self.spin_th.value()
        bk = self.spin_blur.value()
        if bk % 2 == 0:
            bk += 1
        mi = self.spin_morph.value()
        method = self.combo_method.currentText()
        ab = self.spin_block.value()
        if ab % 2 == 0:
            ab += 1
        ac = self.spin_C.value()
        gt = self.spin_global_thresh.value()
        cl = self.spin_canny_low.value()
        ch = self.spin_canny_high.value()

        total = len(self.image_files)
        errors = 0
        total_particles = 0

        progress = QProgressDialog("Batch cutting...", "Cancel", 0, total, self)
        progress.setWindowTitle("Batch Cut Progress")
        progress.setMinimumWidth(400)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        for idx, fname in enumerate(self.image_files):
            progress.setValue(idx)
            progress.setLabelText("Processing: %s\nImages: %d / %d  |  Particles: %d" % (
                fname, idx + 1, total, total_particles))
            QApplication.processEvents()

            if progress.wasCanceled():
                break

            path = os.path.join(self.image_folder, fname)
            img = imread_unicode(path)
            if img is None:
                import imageio
                try:
                    tmp = imageio.mimread(path)
                    if tmp:
                        arr = np.array(tmp[0])
                        img = arr[:, :, 0:3]
                except Exception:
                    errors += 1
                    continue
            if img is None:
                errors += 1
                continue

            rects, _ = detect_contours(img, th, bk, mi, method,
                                       adapt_block=ab, adapt_C=ac,
                                       global_thresh=gt,
                                       canny_low=cl, canny_high=ch)
            imgs = cut_with_rects(img, rects)
            total_particles += len(imgs)

            base = os.path.splitext(fname)[0]
            for i, crop in enumerate(imgs):
                out_name = "%s_%03d.png" % (base, i + 1)
                imwrite_unicode(os.path.join(output, out_name), crop)

        progress.setValue(total)

        processed = idx + 1 if progress.wasCanceled() else total
        msg = "%d images processed, %d particles cropped.\nSaved to: %s" % (
            processed - errors, total_particles, output
        )
        if errors:
            msg += "\n%d images failed to load." % errors
        if progress.wasCanceled():
            msg = "Batch cut cancelled.\n" + msg
        QMessageBox.information(self, "Batch Cut Complete", msg)

    # ------------------------------------------------------------------ Layout
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'center_panel'):
            ch = self.center_panel.height() - 24  # subtract page label height
            btn_h = min(100, max(60, ch // 3))
            btn_y = (ch - btn_h) // 2
            self.btn_prev.setGeometry(0, btn_y, 50, btn_h)
            self.btn_next.setGeometry(self.center_panel.width() - 50, btn_y, 50, btn_h)

    # ------------------------------------------------------------------ Show/Close
    def showEvent(self, event):
        super().showEvent(event)
        if self._defer_show_image and self.image_files:
            self._defer_show_image = False
            QTimer.singleShot(50, self._show_current_image)

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)

    def _save_settings(self):
        cfg = {
            "method": self.combo_method.currentText(),
            "th_size": self.spin_th.value(),
            "blur_kernel": self.spin_blur.value(),
            "adapt_block": self.spin_block.value(),
            "adapt_C": self.spin_C.value(),
            "morph_iter": self.spin_morph.value(),
            "global_thresh": self.spin_global_thresh.value(),
            "canny_low": self.spin_canny_low.value(),
            "canny_high": self.spin_canny_high.value(),
            "show_bin": self.chk_show_bin.isChecked(),
            "image_folder": self.image_folder,
        }
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load_settings(self):
        if not os.path.exists(CONFIG_FILE):
            return
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            return

        method = cfg.get("method", "Adaptive Threshold")
        idx = self.combo_method.findText(method)
        if idx >= 0:
            self.combo_method.setCurrentIndex(idx)
        self.spin_th.setValue(cfg.get("th_size", 10.0))
        self.spin_blur.setValue(cfg.get("blur_kernel", 5))
        self.spin_block.setValue(cfg.get("adapt_block", 25))
        self.spin_C.setValue(cfg.get("adapt_C", 3))
        self.spin_morph.setValue(cfg.get("morph_iter", 5))
        self.spin_global_thresh.setValue(cfg.get("global_thresh", 128))
        self.spin_canny_low.setValue(cfg.get("canny_low", 50))
        self.spin_canny_high.setValue(cfg.get("canny_high", 150))
        self.chk_show_bin.setChecked(cfg.get("show_bin", False))

        folder = cfg.get("image_folder", "")
        if folder and os.path.isdir(folder):
            self.image_folder = folder
            self.label_folder.setText(folder)
            self._load_file_list()
            # Defer image display to showEvent so layout is computed first
            self._defer_show_image = True


# ------------------------------------------------------------------ Entry
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = PlanktoscopeSegmentViewer()
    gui.showMaximized()
    sys.exit(app.exec_())
