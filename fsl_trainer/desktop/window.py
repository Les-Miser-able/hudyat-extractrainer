"""Dockable native workspaces connected directly to the Python job services."""
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

from PySide6.QtCore import QSettings, Qt, QThreadPool, QTimer, QUrl
from PySide6.QtGui import QAction, QColor, QDesktopServices, QFont, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDockWidget, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QFileDialog, QPlainTextEdit, QProgressBar,
    QPushButton, QScrollArea, QSizePolicy, QSlider, QSpinBox, QStackedWidget,
    QSplitter, QTabWidget, QTableWidget, QTableWidgetItem, QToolBar, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget, QCheckBox,
)

from ..dashboard import server as backend
from ..dashboard.live import LiveInference
from ..dashboard.model_exports import native_archive
from ..dashboard.model_records import DECISIONS, read_records, save_record
from ..dashboard.prediction_tests import export_model, save_test
from ..data.preprocessing import HAND_RELATIVE
from ..data.source_selection import SUPPORTED
from ..reporting.benchmark import benchmark_csv
from .artifacts import import_category, read_history, stage_progress
from .widgets import CurvePlot, MediaCanvas, STYLE
from .workers import CameraThread, MediaThread, Task


def label(text="", *, heading=False, muted=False):
    widget = QLabel(text)
    widget.setWordWrap(True)
    widget.setProperty("heading", heading)
    widget.setProperty("muted", muted)
    return widget


def button(text, callback, primary=False):
    widget = QPushButton(text)
    widget.setProperty("primary", primary)
    widget.clicked.connect(callback)
    return widget


def spin(value, minimum=1, maximum=10000):
    widget = QSpinBox()
    widget.setRange(minimum, maximum)
    widget.setValue(value)
    return widget


def decimal(value, minimum=0, maximum=1, decimals=3):
    widget = QDoubleSpinBox()
    widget.setDecimals(decimals)
    widget.setRange(minimum, maximum)
    widget.setSingleStep(10 ** -min(decimals, 2))
    widget.setValue(value)
    return widget


def choice(items):
    widget = QComboBox()
    for text, value in items:
        widget.addItem(text, value)
    return widget


def page():
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(18, 16, 18, 16)
    layout.setSpacing(12)
    return widget, layout


def form_group(title, rows):
    box = QGroupBox(title)
    layout = QFormLayout(box)
    layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
    layout.setVerticalSpacing(10)
    for title, widget in rows:
        if title:
            layout.addRow(title, widget)
        else:
            layout.addRow(widget)
    return box


def refill(combo, items):
    selected = combo.currentData()
    combo.blockSignals(True)
    combo.clear()
    for text, value in items:
        combo.addItem(text, value)
    index = combo.findData(selected)
    combo.setCurrentIndex(index if index >= 0 else 0)
    combo.blockSignals(False)


def percent(value):
    return "—" if value is None else f"{value:.1%}"


class StudioWindow(QMainWindow):
    def __init__(self, jobs, *, restore=True):
        super().__init__()
        self.jobs, self.root = jobs, backend.BASE
        self.live = LiveInference(jobs, self.root)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(3)
        self.tasks = set()
        self.closed = self.scanning = False
        self.refresh_again = False
        self.job_starting = False
        self.categories, self.runs, self.records = [], [], {}
        self.media = self.camera = None
        self.camera_preview = None
        self.media_workers = set()
        self.media_path = self.training_path = None
        self.training_job_id = None
        self.training_epochs = 100
        self.last_jobs, self.processed_jobs = {}, set()
        self.last_log = None
        self.last_prediction = None
        self.review_drafts = {}
        self.loading_review = False
        self.review_run_id = None
        settings_path = self.root / "outputs" / "desktop.ini"
        self.settings = QSettings(str(settings_path), QSettings.Format.IniFormat)
        self.setWindowTitle("FSL Studio — Native workspace")
        self.resize(1500, 980)
        self.setMinimumSize(1080, 740)
        palette = QPalette()
        for role, color in ((QPalette.ColorRole.Window, "#202c35"), (QPalette.ColorRole.WindowText, "#e3ebef"),
                            (QPalette.ColorRole.Base, "#1b2831"), (QPalette.ColorRole.Text, "#e3ebef"),
                            (QPalette.ColorRole.Button, "#334650"), (QPalette.ColorRole.ButtonText, "#e3ebef"),
                            (QPalette.ColorRole.Highlight, "#437b70"), (QPalette.ColorRole.HighlightedText, "#ffffff")):
            palette.setColor(role, QColor(color))
        self.setPalette(palette)
        self.setStyleSheet(STYLE)
        self.setDockNestingEnabled(True)
        self.build_toolbar()
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.build_extractor()
        self.build_trainer()
        self.build_models()
        self.build_predictor()
        self.build_explorer()
        self.build_inspector()
        self.build_output()
        self.build_menu()
        self.tabs.currentChanged.connect(self.workspace_changed)
        self.status_text = label("●  Workspace: local", muted=True)
        self.statusBar().addWidget(self.status_text, 1)
        environment = label(f"Python {sys.version.split()[0]}  •  32 frames × 128 features", muted=True)
        environment.setWordWrap(False)
        self.statusBar().addPermanentWidget(environment)
        self.default_layout = self.saveState()
        self.resizeDocks([self.explorer_dock, self.inspector_dock], [245, 300], Qt.Orientation.Horizontal)
        self.resizeDocks([self.output_dock], [210], Qt.Orientation.Vertical)
        if restore:
            if self.settings.contains("geometry"):
                self.restoreGeometry(self.settings.value("geometry"))
            if self.settings.contains("layout"):
                self.restoreState(self.settings.value("layout"))
            self.tabs.setCurrentIndex(int(self.settings.value("workspace", 0)))
            self.source_root.setText(self.settings.value("source_root", "data"))
            self.dataset_root.setText(self.settings.value("dataset_root", "datasets_augmented"))
            self.camera_mirror.setChecked(self.settings.value("mirror_preview", False, type=bool))
        self.workspace_changed(self.tabs.currentIndex())
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll_jobs)
        self.timer.start(700)
        self.refresh()

    def build_toolbar(self):
        toolbar = QToolBar("Workspace tools", self)
        toolbar.setObjectName("workspace-toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        brand = label("f.  FSL STUDIO")
        brand.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        toolbar.addWidget(brand)
        toolbar.addSeparator()
        self.open_action = toolbar.addAction("Open sample…", self.open_sample)
        self.open_action.setShortcut("Ctrl+O")
        toolbar.addAction("Import category…", self.import_source)
        self.refresh_action = toolbar.addAction("Refresh", self.refresh)
        self.refresh_action.setShortcut("F5")
        toolbar.addAction("Build paper report", lambda: self.request_job({"action": "paper-report", **self.roots()}))
        toolbar.addAction("Open paper report", lambda: self.open_paper_report(self.root / "reports" / "paper"))
        toolbar.addSeparator()
        self.start_action = toolbar.addAction("Start extraction", self.start_workspace_job)
        self.start_action.setShortcut("Ctrl+Return")
        toolbar.widgetForAction(self.start_action).setProperty("primary", True)
        self.stop_action = toolbar.addAction("Stop job", self.stop_job)
        self.stop_action.setEnabled(False)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        self.engine_status = label("●  Local engine ready", muted=True)
        self.engine_status.setProperty("accent", True)
        toolbar.addWidget(self.engine_status)

    def build_extractor(self):
        widget, layout = page()
        self.source_title = label("Source monitor", heading=True)
        layout.addWidget(self.source_title)
        self.source_caption = label("Select a category in the explorer, or open a local sample.", muted=True)
        layout.addWidget(self.source_caption)
        self.canvas = MediaCanvas()
        layout.addWidget(self.canvas, 1)
        controls = QHBoxLayout()
        self.play_button = button("Play", self.toggle_play)
        self.play_button.setEnabled(False)
        controls.addWidget(self.play_button)
        self.frame_label = label("Frame — / —", muted=True)
        self.frame_label.setWordWrap(False)
        controls.addWidget(self.frame_label)
        controls.addStretch()
        self.overlay = QCheckBox("Preview hand landmarks")
        self.overlay.setToolTip("Runs the default MediaPipe hand detector on preview frames. Does not write a dataset.")
        self.overlay.toggled.connect(self.reload_preview)
        controls.addWidget(self.overlay)
        layout.addLayout(controls)
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setEnabled(False)
        self.timeline.sliderMoved.connect(self.seek_media)
        self.timeline.sliderPressed.connect(self.pause_media)
        layout.addWidget(self.timeline)
        self.source_summary = label("FRAME NAVIGATION  •  Scrub a video or inspect an extracted sequence", muted=True)
        layout.addWidget(self.source_summary)
        self.extraction_progress = QProgressBar()
        self.extraction_progress.setValue(0)
        self.extraction_progress.setTextVisible(False)
        layout.addWidget(self.extraction_progress)
        self.extraction_status = label("No extraction job running. Prepare a category to create a named dataset.", muted=True)
        layout.addWidget(self.extraction_status)
        self.tabs.addTab(widget, "Extractor")

    def build_trainer(self):
        widget, layout = page()
        self.training_title = label("Training monitor", heading=True)
        layout.addWidget(self.training_title)
        layout.addWidget(label("Conv1D–BiLSTM  /  category gesture classifier", muted=True))
        self.epoch_label = label("Choose a prepared dataset in Run configuration.", muted=True)
        layout.addWidget(self.epoch_label)
        self.training_progress = QProgressBar()
        self.training_progress.setValue(0)
        self.training_progress.setTextVisible(False)
        layout.addWidget(self.training_progress)
        self.accuracy_plot = CurvePlot("Accuracy", "accuracy")
        self.loss_plot = CurvePlot("Loss", "loss")
        layout.addWidget(self.accuracy_plot, 1)
        layout.addWidget(self.loss_plot, 1)
        self.training_metrics = label("Training and validation metrics appear after each completed epoch.", muted=True)
        layout.addWidget(self.training_metrics)
        self.tabs.addTab(widget, "Trainer")

    def build_models(self):
        widget, layout = page()
        layout.addWidget(label("Model runs", heading=True))
        layout.addWidget(label("Saved checkpoints and evaluation results from this project.", muted=True))
        self.run_table = QTableWidget(0, 6)
        self.run_table.setHorizontalHeaderLabels(["Category / run", "Accuracy", "Macro F1", "Classes", "Checkpoint", "Decision"])
        self.run_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.run_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.run_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.run_table.setAlternatingRowColors(True)
        self.run_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.run_table.verticalHeader().setVisible(False)
        self.run_table.itemSelectionChanged.connect(self.select_run)
        layout.addWidget(self.run_table, 1)
        self.run_detail = QPlainTextEdit()
        self.run_detail.setReadOnly(True)
        self.run_detail.setMaximumHeight(230)
        self.run_detail.setPlaceholderText("Select a run to inspect its metrics and provenance.")
        layout.addWidget(self.run_detail)
        actions = QHBoxLayout()
        actions.addWidget(button("View learning curves", self.view_run_curves))
        actions.addWidget(button("Try this model", self.use_run, True))
        actions.addWidget(button("Export benchmarks CSV…", self.export_benchmarks))
        actions.addWidget(button("Build tables and charts", lambda: self.model_job("paper-report")))
        actions.addWidget(button("Open tables and charts", self.open_model_report))
        layout.addLayout(actions)
        self.tabs.addTab(widget, "Model runs")

    def build_predictor(self):
        widget, layout = page()
        heading = QHBoxLayout()
        heading.addWidget(label("Try a prediction", heading=True), 1)
        self.camera_mirror = QCheckBox("Mirror preview")
        self.camera_mirror.setToolTip("Flip the view like a mirror. Recognition keeps the original camera frames.")
        self.camera_mirror.toggled.connect(self.mirror_camera_preview)
        heading.addWidget(self.camera_mirror)
        heading.addWidget(button("Fullscreen camera", self.open_camera_preview))
        layout.addLayout(heading)
        self.camera_status = label("Choose a saved model, then recognize a sample or start the camera.", muted=True)
        layout.addWidget(self.camera_status)
        self.prediction_splitter = QSplitter(Qt.Orientation.Vertical)
        self.camera_canvas = MediaCanvas()
        self.camera_canvas.setMinimumHeight(300)
        self.camera_canvas.message = "LIVE CAMERA\nFrames are processed locally"
        self.prediction_splitter.addWidget(self.camera_canvas)
        results = QWidget()
        result_layout = QVBoxLayout(results)
        result_layout.setContentsMargins(0, 4, 0, 0)
        result_heading = QHBoxLayout()
        self.prediction_title = label("Recognition", heading=True)
        result_heading.addWidget(self.prediction_title, 1)
        self.save_prediction_button = button("Save prediction test", self.save_prediction)
        self.save_prediction_button.setEnabled(False)
        result_heading.addWidget(self.save_prediction_button)
        result_layout.addLayout(result_heading)
        self.prediction_table = QTableWidget(0, 2)
        self.prediction_table.setHorizontalHeaderLabels(["Gesture", "Confidence"])
        self.prediction_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.prediction_table.verticalHeader().setVisible(False)
        self.prediction_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.prediction_table.setMinimumHeight(60)
        result_layout.addWidget(self.prediction_table)
        results.setMaximumHeight(160)
        self.prediction_splitter.addWidget(results)
        self.prediction_splitter.setCollapsible(0, False)
        self.prediction_splitter.setCollapsible(1, True)
        self.prediction_splitter.setStretchFactor(0, 1)
        self.prediction_splitter.setStretchFactor(1, 0)
        self.prediction_splitter.setSizes([600, 130])
        layout.addWidget(self.prediction_splitter, 1)
        self.tabs.addTab(widget, "Prediction")

    def dock(self, title, name, widget, area):
        dock = QDockWidget(title, self)
        dock.setObjectName(name)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def build_explorer(self):
        widget, layout = page()
        layout.setContentsMargins(8, 8, 8, 8)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setMinimumWidth(175)
        self.tree.itemActivated.connect(self.activate_item)
        self.tree.itemClicked.connect(self.activate_item)
        self.tree.itemExpanded.connect(self.expand_folder)
        layout.addWidget(self.tree, 1)
        project = QGroupBox("LOCAL PROJECT")
        box = QVBoxLayout(project)
        box.addWidget(label(self.root.name))
        box.addWidget(label("Datasets, checkpoints, and reports\nstay in your project folders.", muted=True))
        box.addWidget(button("Open project folder", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.root)))))
        layout.addWidget(project)
        self.explorer_dock = self.dock("PROJECT EXPLORER", "project-explorer", widget, Qt.DockWidgetArea.LeftDockWidgetArea)

    def build_inspector(self):
        self.inspectors = QStackedWidget()
        self.prepare_fields = {
            "category": QComboBox(), "dataset_name": QLineEdit("prepared_v1"),
            "normalization": choice([("Automatic", "auto"), ("Image coordinates", "image"), ("Hand-relative sequence", HAND_RELATIVE)]),
            "validation": decimal(.15, .001, .99), "test": decimal(.15, .001, .99),
            "variants": spin(8, 1, 1000), "jitter": QLineEdit(),
            "confidence": decimal(.5), "videos_only": QCheckBox("Use videos only"),
            "swap_hands": QCheckBox("Swap left / right slots"),
            "source_folders": QLineEdit(), "groups": QLineEdit(),
            'split_strategy': choice([('Source groups', 'grouped'), ('Leave one signer out', 'loso')]),
            'test_signer': QComboBox(), 'validation_signer': QComboBox(),
            "video_extra_samples": spin(0, 0, 1000), "video_time_warp": decimal(.05, 0, .5),
            'video_mirror_hands': QCheckBox('Add opposite-hand video copies'),
            "seed": spin(42, 0, 2147483647), "hand_model": QLineEdit(),
        }
        self.prepare_fields["jitter"].setPlaceholderText("Automatic for feature convention")
        self.prepare_fields["source_folders"].setPlaceholderText("All folders, or S1, S2")
        self.prepare_fields["hand_model"].setPlaceholderText("Default hand detector")
        self.prepare_fields['video_mirror_hands'].setToolTip(
            'Adds training-only mirrored copies of videos and their timing variants. '
            'Use only for signs whose mirrored form has the same meaning. Prepare a new dataset and retrain.')
        self.source_root, self.dataset_root = QLineEdit("data"), QLineEdit("datasets_augmented")
        extract, layout = page()
        layout.setContentsMargins(12, 8, 12, 12)
        layout.addWidget(form_group("Source and output", [("Category", self.prepare_fields["category"]),
            ("New dataset name", self.prepare_fields["dataset_name"])]))
        self.category_summary = label("Scanning source library…", muted=True)
        layout.addWidget(self.category_summary)
        layout.addWidget(form_group("MediaPipe Hands · 32 × 128", [("Feature convention", self.prepare_fields["normalization"]),
            ("Detection confidence", self.prepare_fields["confidence"]), ("Custom .task model (project path)", self.prepare_fields["hand_model"]),
            ("", self.prepare_fields["videos_only"]), ("", self.prepare_fields["swap_hands"])]))
        layout.addWidget(form_group("Splits and augmentation", [("Validation fraction", self.prepare_fields["validation"]),
            ("Test fraction", self.prepare_fields["test"]), ("Static variants", self.prepare_fields["variants"]),
            ("Jitter", self.prepare_fields["jitter"]), ("Extra video samples", self.prepare_fields["video_extra_samples"]),
            ("Video time warp", self.prepare_fields["video_time_warp"]), ("Random seed", self.prepare_fields["seed"]),
            ('', self.prepare_fields['video_mirror_hands']),
            ("Source subfolders (comma separated)", self.prepare_fields["source_folders"]), ("Group CSV (project path)", self.prepare_fields["groups"])]))
        layout.addWidget(form_group('Signer evaluation', [('Split strategy', self.prepare_fields['split_strategy']),
            ('Held-out test signer', self.prepare_fields['test_signer']),
            ('Validation signer', self.prepare_fields['validation_signer'])]))
        layout.addWidget(label('LOSO reads gesture/signer/recording subfolders. Use at least three signers. '
            'Fractions are ignored; validation and test contain clean sources only. '
            'Prepare one named dataset per held-out signer.', muted=True))
        self.prepare_fields['split_strategy'].currentIndexChanged.connect(self.split_strategy_changed)
        self.split_strategy_changed()
        self.prepare_button = button("Extract & prepare dataset", self.prepare, True)
        layout.addWidget(self.prepare_button)
        layout.addWidget(form_group("Project folders", [("Source root (project path)", self.source_root),
            ("Dataset root (project path)", self.dataset_root)]))
        layout.addWidget(button("Scan folders", self.refresh))
        layout.addStretch()
        self.add_inspector(extract)
        self.prepare_fields["category"].currentIndexChanged.connect(self.category_changed)

        self.train_fields = {"category": QComboBox(), "dataset": QComboBox(), "run": QLineEdit("experiment_001"),
            "epochs": spin(100), "batch_size": spin(32, 1, 65536),
            "learning_rate": decimal(.001, .0000000001, 1, 10), "patience": spin(12),
            "seed": spin(42, 0, 2147483647), "class_weights": QCheckBox("Balance class weights")}
        train, layout = page()
        layout.setContentsMargins(12, 8, 12, 12)
        layout.addWidget(form_group("Run configuration", [("Category", self.train_fields["category"]),
            ("Prepared dataset", self.train_fields["dataset"]), ("New run name", self.train_fields["run"])]))
        layout.addWidget(form_group("Conv1D–BiLSTM", [("Epochs", self.train_fields["epochs"]),
            ("Batch size", self.train_fields["batch_size"]), ("Learning rate", self.train_fields["learning_rate"]),
            ("Early stopping patience", self.train_fields["patience"]), ("Random seed", self.train_fields["seed"]),
            ("", self.train_fields["class_weights"])]))
        layout.addWidget(label("Input: 32 frames × 128 features\nBest checkpoint and reports are saved automatically.", muted=True))
        self.dataset_protocol = label('', muted=True)
        layout.addWidget(self.dataset_protocol)
        self.train_button = button("Start training", self.train, True)
        layout.addWidget(self.train_button)
        self.inspect_button = button("Validate selected dataset", self.inspect_dataset)
        layout.addWidget(self.inspect_button)
        layout.addWidget(button("Build dataset tables and charts", self.build_dataset_report))
        layout.addWidget(button("Open dataset tables and charts", self.open_dataset_report))
        self.delete_dataset_button = button('Delete selected dataset…', self.delete_dataset)
        layout.addWidget(self.delete_dataset_button)
        layout.addWidget(label("Stop keeps partial outputs. Continue a checkpoint into a new run with the trainer CLI.", muted=True))
        layout.addStretch()
        self.add_inspector(train)
        self.train_fields["category"].currentIndexChanged.connect(self.datasets_changed)
        self.train_fields['dataset'].currentIndexChanged.connect(self.dataset_protocol_changed)

        model, layout = page()
        layout.setContentsMargins(12, 8, 12, 12)
        self.model_caption = label("Select a saved run.", muted=True)
        layout.addWidget(self.model_caption)
        self.decision = choice([(v, v) for v in DECISIONS])
        self.notes = QPlainTextEdit()
        self.notes.setPlaceholderText("Notes about this model")
        self.notes.setMaximumHeight(130)
        self.notes.textChanged.connect(self.review_edited)
        self.decision.currentIndexChanged.connect(self.review_edited)
        layout.addWidget(form_group("Review", [("Decision", self.decision), ("Notes", self.notes)]))
        layout.addWidget(button("Save model review", self.save_review))
        layout.addWidget(button("Save Keras model ZIP…", self.export_keras, True))
        layout.addWidget(button("Build HUDYAT bundle", lambda: self.model_job("export")))
        layout.addWidget(button("Run benchmark", lambda: self.model_job("benchmark")))
        layout.addWidget(button("Export model record JSON…", self.export_record))
        layout.addWidget(button("Open run folder", self.open_run_folder))
        self.delete_model_button = button('Delete selected model…', self.delete_model)
        layout.addWidget(self.delete_model_button)
        layout.addWidget(label("Export and benchmark jobs appear in Output / Jobs. Saved prediction tests are included in the model record.", muted=True))
        layout.addStretch()
        self.add_inspector(model)

        prediction, layout = page()
        layout.setContentsMargins(12, 8, 12, 12)
        self.predict_run = QComboBox()
        self.sample_path = QLineEdit()
        self.sample_path.setPlaceholderText("data/category/gesture/sample.mp4")
        self.predict_hand_model = QLineEdit()
        layout.addWidget(form_group("Model and sample", [("Saved model", self.predict_run),
            ("Sample (project path)", self.sample_path), ("Custom .task model (optional)", self.predict_hand_model)]))
        layout.addWidget(button("Choose sample…", self.choose_prediction_sample))
        self.predict_button = button("Recognize sample", self.predict_sample, True)
        layout.addWidget(self.predict_button)
        self.camera_mode = choice([("Perform once", "gesture"), ("Moving gesture", "motion"), ("Held pose", "pose")])
        self.camera_device = spin(0, 0, 10)
        self.settle_ms, self.missing_ms, self.interval_ms = spin(600, 100, 2000), spin(400, 100, 2000), spin(50, 20, 500)
        layout.addWidget(form_group("Live camera", [("Capture mode", self.camera_mode), ("Camera index", self.camera_device),
            ("Stillness delay (ms)", self.settle_ms), ("Hands missing delay (ms)", self.missing_ms),
            ("Frame interval (ms)", self.interval_ms)]))
        self.camera_button = button("Start camera", self.start_camera, True)
        self.camera_stop = button("Stop camera", self.stop_camera)
        self.camera_stop.setEnabled(False)
        self.camera_next = button("Next gesture", self.next_gesture)
        self.camera_next.setEnabled(False)
        layout.addWidget(self.camera_button)
        layout.addWidget(self.camera_stop)
        layout.addWidget(self.camera_next)
        layout.addWidget(label("Mirror preview flips the display. Recognition uses the original frames. Drag the divider below the camera to resize it, or use Fullscreen camera. Timing settings apply at camera start.", muted=True))
        layout.addStretch()
        self.add_inspector(prediction)
        self.predict_run.currentIndexChanged.connect(self.prediction_model_changed)
        self.inspector_dock = self.dock("EXTRACTION INSPECTOR", "inspector", self.inspectors, Qt.DockWidgetArea.RightDockWidgetArea)
        self.inspector_dock.setMinimumWidth(275)

    def add_inspector(self, widget):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        scroll.setMinimumWidth(260)
        self.inspectors.addWidget(scroll)

    def build_output(self):
        widget, layout = page()
        layout.setContentsMargins(10, 6, 10, 8)
        controls = QHBoxLayout()
        self.job_select = QComboBox()
        self.job_select.setMinimumWidth(250)
        self.job_select.currentIndexChanged.connect(self.poll_jobs)
        controls.addWidget(self.job_select, 1)
        self.job_status = label("Idle", muted=True)
        controls.addWidget(self.job_status)
        self.open_export_button = button("Show exported bundle", self.open_export_folder)
        self.open_export_button.setVisible(False)
        controls.addWidget(self.open_export_button)
        controls.addWidget(button("Stop job", self.stop_job))
        layout.addLayout(controls)
        self.logs = QPlainTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setFont(QFont("Consolas", 9))
        self.logs.setPlaceholderText("No jobs yet. Your workspace is ready.")
        self.logs.setMinimumHeight(80)
        layout.addWidget(self.logs, 1)
        self.output_dock = self.dock("OUTPUT / JOBS", "output-jobs", widget, Qt.DockWidgetArea.BottomDockWidgetArea)

    def build_menu(self):
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self.open_action)
        file_menu.addAction("Import category…", self.import_source)
        file_menu.addAction(self.refresh_action)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close, "Alt+F4")
        view = self.menuBar().addMenu("View")
        for dock in (self.explorer_dock, self.inspector_dock, self.output_dock):
            view.addAction(dock.toggleViewAction())
        view.addSeparator()
        view.addAction("Reset dock layout", self.reset_layout)
        workspace = self.menuBar().addMenu("Workspace")
        for index in range(self.tabs.count()):
            action = QAction(self.tabs.tabText(index), self)
            action.setShortcut(f"Ctrl+{index+1}")
            action.triggered.connect(lambda checked=False, index=index: self.tabs.setCurrentIndex(index))
            workspace.addAction(action)
        run = self.menuBar().addMenu("Run")
        run.addAction(self.start_action)
        run.addAction(self.stop_action)
        help_menu = self.menuBar().addMenu("Help")
        help_menu.addAction("About FSL Studio", lambda: QMessageBox.information(self, "FSL Studio",
            "Native Python / Qt workspace for Filipino Sign Language.\n\n"
            "Prepare datasets, train Conv1D–BiLSTM models, and test saved checkpoints locally.\n\n"
            "Ctrl+1–4: switch workspaces · Ctrl+O: open sample · F5: refresh\n"
            "Drag panel titles to dock or float them. View restores hidden panels.\n\n"
            "See USER_GUIDE.md and docs/NATIVE_DESKTOP.md in the project folder."))

    def workspace_changed(self, index):
        self.inspectors.setCurrentIndex(index)
        titles = ["EXTRACTION INSPECTOR", "RUN CONFIGURATION", "MODEL REVIEW", "PREDICTION SETTINGS"]
        self.inspector_dock.setWindowTitle(titles[index])
        self.start_action.setText(["Start extraction", "Start training", "Try selected model", "Recognize sample"][index])
        if self.media and index != 0:
            self.pause_media()

    def reset_layout(self):
        self.restoreState(self.default_layout)
        self.resizeDocks([self.explorer_dock, self.inspector_dock], [245, 300], Qt.Orientation.Horizontal)
        self.resizeDocks([self.output_dock], [210], Qt.Orientation.Vertical)

    def error(self, message):
        if not self.closed:
            QMessageBox.warning(self, "FSL Studio", message)

    def background(self, function, callback, failed=None):
        task = Task(function)
        self.tasks.add(task)

        def finish(value, success):
            self.tasks.discard(task)
            if not self.closed:
                (callback if success else failed or self.error)(value)

        task.signals.done.connect(lambda value: finish(value, True))
        task.signals.failed.connect(lambda value: finish(value, False))
        self.pool.start(task)

    def refresh(self):
        if self.closed:
            return
        if self.scanning:
            self.refresh_again = True
            return
        self.scanning = True
        data, datasets = self.source_root.text().strip(), self.dataset_root.text().strip()
        self.engine_status.setText("●  Scanning project…")

        def loaded(result):
            self.scanning = False
            self.categories, self.runs, self.records = result
            self.populate_explorer(data)
            for fields in (self.prepare_fields, self.train_fields):
                refill(fields["category"], [(r["name"], r["name"]) for r in self.categories])
            self.category_changed()
            self.datasets_changed()
            previous_model = self.predict_run.currentData()
            refill(self.predict_run, [(r["category"] + " / " + r["name"], r["id"]) for r in self.runs if r["checkpoint"]])
            if previous_model != self.predict_run.currentData():
                self.prediction_model_changed()
            self.populate_runs()
            self.engine_status.setText("●  Local engine ready")
            self.status_text.setText(f"●  Workspace: local  •  {len(self.categories)} categories  •  {len(self.runs)} saved runs")
            if self.refresh_again:
                self.refresh_again = False
                self.refresh()

        def failed(message):
            self.scanning = False
            self.engine_status.setText("●  Scan failed")
            self.error(message)

        self.background(lambda: (backend.inventory(data, datasets), backend.runs(), read_records(self.root)), loaded, failed)

    def tree_item(self, parent, text, payload):
        item = QTreeWidgetItem(parent, [text])
        item.setData(0, Qt.ItemDataRole.UserRole, payload)
        item.setToolTip(0, text)
        return item

    def populate_explorer(self, source_root):
        self.tree.clear()
        source = self.tree_item(self.tree, "SOURCE LIBRARY", {})
        prepared = self.tree_item(self.tree, "PREPARED DATASETS", {})
        runs = self.tree_item(self.tree, "MODEL RUNS", {})
        for row in self.categories:
            category = self.tree_item(source, f"{row['name']}  ·  {row['samples']:,}", {"kind": "category", "category": row["name"]})
            for gesture in row["classes"]:
                item = self.tree_item(category, f"{gesture['name']}  ({gesture['samples']:,})", {
                    "kind": "folder", "path": str(self.root / source_root / row["name"] / gesture["name"])})
                item.setChildIndicatorPolicy(QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator)
            for dataset in row["datasets"]:
                self.tree_item(prepared, row["name"] + " / " + dataset["name"],
                               {"kind": "dataset", "category": row["name"], "path": dataset["path"]})
        for row in self.runs:
            self.tree_item(runs, row["category"] + " / " + row["name"], {"kind": "run", "id": row["id"]})
        for item in (source, prepared, runs):
            item.setExpanded(True)

    def expand_folder(self, item):
        data = item.data(0, Qt.ItemDataRole.UserRole) or {}
        if data.get("kind") != "folder" or item.childCount():
            return
        try:
            paths = sorted(Path(data["path"]).iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            shown = 0
            for path in paths:
                if self.root.resolve() not in path.resolve().parents:
                    continue
                if path.is_dir() or path.suffix.lower() in SUPPORTED:
                    child = self.tree_item(item, path.name, {"kind": "folder" if path.is_dir() else "media", "path": str(path)})
                    if path.is_dir():
                        child.setChildIndicatorPolicy(QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator)
                    shown += 1
                if shown >= 500:
                    self.tree_item(item, "More files available in the project folder…", {})
                    break
        except OSError as error:
            self.error(str(error))

    def activate_item(self, item, column=0):
        data = item.data(0, Qt.ItemDataRole.UserRole) or {}
        kind = data.get("kind")
        if kind == "category":
            self.prepare_fields["category"].setCurrentIndex(self.prepare_fields["category"].findData(data["category"]))
            self.tabs.setCurrentIndex(0)
        elif kind == "dataset":
            self.train_fields["category"].setCurrentIndex(self.train_fields["category"].findData(data["category"]))
            self.train_fields["dataset"].setCurrentIndex(self.train_fields["dataset"].findData(data["path"]))
            self.tabs.setCurrentIndex(1)
        elif kind == "run":
            for index, row in enumerate(self.runs):
                if row["id"] == data["id"]:
                    self.run_table.selectRow(index)
                    break
            self.tabs.setCurrentIndex(2)
        elif kind == "media":
            self.open_media(Path(data["path"]))

    def category_changed(self):
        row = next((r for r in self.categories if r["name"] == self.prepare_fields["category"].currentData()), None)
        self.category_summary.setText(f"{len(row['classes'])} classes · {row['samples']:,} sources\n" +
            (row["preparation_issue"] or "Ready to extract and split into a new dataset.") if row else "No source categories. Import a category folder to begin.")
        signers = sorted({s for c in row['classes'] for s in c.get('folders', {})}) if row else []
        refill(self.prepare_fields['test_signer'], [(s, s) for s in signers])
        refill(self.prepare_fields['validation_signer'], [('Automatic (next signer)', None), *[(s, s) for s in signers]])

    def split_strategy_changed(self):
        loso = self.prepare_fields['split_strategy'].currentData() == 'loso'
        for key in ('test_signer', 'validation_signer'):
            self.prepare_fields[key].setEnabled(loso)
        for key in ('validation', 'test'):
            self.prepare_fields[key].setEnabled(not loso)

    def datasets_changed(self):
        row = next((r for r in self.categories if r["name"] == self.train_fields["category"].currentData()), None)
        refill(self.train_fields["dataset"], [(d["name"], d["path"]) for d in row["datasets"]] if row else [])
        self.dataset_protocol_changed()

    def dataset_protocol_changed(self):
        dataset = next((d for c in self.categories for d in c['datasets']
                        if d['path'] == self.train_fields['dataset'].currentData()), None)
        protocol = dataset.get('evaluation_protocol') if dataset else None
        self.dataset_protocol.setText(
            f"LOSO test: {protocol['test_signer']} · validation: {protocol['validation_signer']}\n"
            f"Training signers: {', '.join(protocol['train_signers'])}" if protocol else
            'Uses the selected dataset’s saved source-group split.')

    def open_sample(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open source sample", str(self.root / "data"),
            "Source samples (*.mp4 *.avi *.mov *.mkv *.webm *.m4v *.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff *.npy);;All files (*)")
        if path:
            self.open_media(Path(path))

    def open_media(self, path):
        if self.media:
            self.media.stop()
        self.media_path = path
        self.tabs.setCurrentIndex(0)
        self.source_title.setText(path.name)
        self.source_caption.setText(str(path))
        self.canvas.frame = None
        self.canvas.message = "Loading source…"
        self.canvas.update()
        self.media = MediaThread(path, self.overlay.isChecked(), self)
        self.media_workers.add(self.media)
        self.media.frame.connect(self.media_frame)
        self.media.failed.connect(self.media_error)
        worker = self.media
        self.media.finished.connect(lambda: self.release_media(worker))
        self.media.start()
        self.play_button.setText("Play")

    def reload_preview(self):
        if self.media_path:
            self.open_media(self.media_path)

    def media_frame(self, frame, info):
        if self.sender() is not self.media or self.closed:
            return
        self.canvas.set_frame(frame, info)
        self.timeline.setRange(0, info["total"] - 1)
        if not self.timeline.isSliderDown():
            self.timeline.setValue(info["index"])
        self.timeline.setEnabled(info["total"] > 1)
        self.play_button.setEnabled(info["total"] > 1)
        self.frame_label.setText(f"Frame {info['index'] + 1:,} / {info['total']:,}   ·   {info['index'] / info['fps']:.2f} s")
        self.source_summary.setText(f"{info['kind'].upper()}  •  {frame.width()} × {frame.height()}  •  {info['fps']:.1f} FPS  •  " +
            ("128 features · 2 hand slots" if info.get("landmarks") else "Preview only · prepare the category to save landmarks"))
        if info["index"] == info["total"] - 1:
            self.play_button.setText("Play")

    def media_error(self, message):
        if self.sender() is self.media:
            self.error(message)

    def release_media(self, worker):
        self.media_workers.discard(worker)
        if worker is self.media:
            self.media = None
            self.play_button.setEnabled(False)
        worker.deleteLater()

    def pause_media(self):
        if self.media:
            self.media.play(False)
        self.play_button.setText("Play")

    def toggle_play(self):
        if self.media:
            playing = not self.media.playing
            if playing and self.timeline.value() == self.timeline.maximum():
                self.media.seek(0)
            self.media.play(playing)
            self.play_button.setText("Pause" if playing else "Play")

    def seek_media(self, index):
        if self.media:
            self.media.seek(index)

    def import_source(self):
        source = QFileDialog.getExistingDirectory(self, "Choose a category containing gesture class folders")
        if not source:
            return
        name, ok = QInputDialog.getText(self, "Import category", "Copy into a new project category named:", text=Path(source).name)
        if not ok:
            return
        try:
            name = backend.folder_name(name)
            destination = backend.local_path(name, backend.local_path(self.source_root.text().strip()))
        except ValueError as error:
            return self.error(str(error))
        self.engine_status.setText("●  Importing source files…")
        self.background(lambda: import_category(source, destination), lambda result: self.refresh())

    def values(self, fields):
        result = {}
        for key, widget in fields.items():
            if isinstance(widget, QComboBox):
                result[key] = widget.currentData()
            elif isinstance(widget, QCheckBox):
                result[key] = widget.isChecked()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                result[key] = widget.value()
            else:
                result[key] = widget.text().strip()
        return result

    def roots(self):
        return {"data": self.source_root.text().strip(), "datasets": self.dataset_root.text().strip()}

    def prepare(self):
        data = self.values(self.prepare_fields)
        if data['split_strategy'] != 'loso':
            data.pop('test_signer', None)
            data.pop('validation_signer', None)
        data["source_folders"] = [s.strip() for s in data["source_folders"].split(",") if s.strip()] or None
        self.request_job({**data, **self.roots(), "action": "prepare"})

    def train(self):
        self.request_job({**self.values(self.train_fields), **self.roots(), "action": "train"})

    def inspect_dataset(self):
        self.request_job({**self.values(self.train_fields), **self.roots(), "action": "inspect"})

    def delete_dataset(self):
        path = self.train_fields['dataset'].currentData()
        if not path:
            return self.error('Choose a prepared dataset first.')
        self.delete_saved('dataset', path)

    def delete_model(self):
        row = self.require_run()
        if row:
            self.delete_saved('model', row['id'])

    def delete_saved(self, kind, path):
        if self.job_starting or self.camera and self.camera.isRunning() or any(
                j['status'] == 'running' for j in self.jobs.snapshot()):
            return self.error('Stop the running job or camera prediction before deleting saved artifacts.')
        if QMessageBox.question(self, f'Delete {kind}?',
                f'Move {path} to local trash?\n\nSource recordings and other saved datasets/models stay available. '
                'Files can be recovered from .trash in the project folder.',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        self.job_starting = True
        self.poll_jobs()
        def finished(result):
            self.job_starting = False
            self.review_drafts.pop(path, None)
            self.statusBar().showMessage('Moved to ' + result['trash'], 10000)
            self.refresh()
            self.poll_jobs()
        def failed(message):
            self.job_starting = False
            self.poll_jobs()
            self.error(message)
        self.background(lambda: self.jobs.delete({'kind': kind, 'path': path, **self.roots()}), finished, failed)

    def build_dataset_report(self):
        dataset = self.train_fields["dataset"].currentData()
        if not dataset:
            return self.error("Choose a prepared dataset in Run configuration.")
        self.request_job({"action": "paper-report", "dataset": dataset, **self.roots()})

    def open_paper_report(self, folder):
        path = folder / "index.html"
        if not path.is_file():
            return self.error("Build the tables and charts first. Follow progress in Output / Jobs.")
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def open_dataset_report(self):
        dataset = self.train_fields["dataset"].currentData()
        if not dataset:
            return self.error("Choose a prepared dataset in Run configuration.")
        self.open_paper_report(self.root / dataset / "paper_report")

    def open_model_report(self):
        row = self.require_run()
        if row:
            self.open_paper_report(self.root / row["id"] / "paper_report")

    def start_workspace_job(self):
        (self.prepare, self.train, self.use_run, self.predict_sample)[self.tabs.currentIndex()]()

    def request_job(self, data):
        if self.job_starting:
            return self.error("A job is starting. Wait for its validation to finish.")
        if self.camera and self.camera.isRunning():
            return self.error("Stop camera prediction before starting a job.")

        def started(job_id):
            self.job_starting = False
            self.output_dock.show()
            if data["action"] == "train":
                self.training_job_id = job_id
                self.training_path = self.root / "runs" / data["category"] / data["run"]
                self.training_epochs = data["epochs"]
                self.training_title.setText("Conv1D–BiLSTM / " + data["run"])
                self.accuracy_plot.set_history({})
                self.loss_plot.set_history({})
                self.tabs.setCurrentIndex(1)
            self.poll_jobs()
            self.job_select.setCurrentIndex(self.job_select.findData(job_id))

        def failed(message):
            self.job_starting = False
            self.poll_jobs()
            self.error(message)

        self.job_starting = True
        self.poll_jobs()
        self.background(lambda: self.jobs.start(data), started, failed)

    def stop_job(self):
        running = next((j for j in self.jobs.snapshot() if j["status"] == "running"), None)
        if running:
            try:
                self.jobs.cancel(running["id"])
            except (ValueError, OSError) as error:
                self.error(str(error))
        self.poll_jobs()

    def poll_jobs(self):
        if self.closed:
            return
        items = self.jobs.snapshot(blocking=False)
        if items is None:
            return  # Validation can scan many files while holding the job lock.
        self.last_jobs = {j["id"]: j for j in items}
        selected = self.job_select.currentData()
        refill(self.job_select, [(f"{j['action']} / {j['category']}  ·  {j['status']}", j["id"]) for j in items])
        if selected is None and items:
            self.job_select.setCurrentIndex(0)
        running = next((j for j in items if j["status"] == "running"), None)
        busy = self.job_starting or bool(running) or bool(self.camera and self.camera.isRunning())
        self.stop_action.setEnabled(bool(running))
        for control in (self.prepare_button, self.train_button, self.inspect_button, self.predict_button, self.start_action,
                        self.delete_dataset_button, self.delete_model_button):
            control.setEnabled(not busy)
        job = self.last_jobs.get(self.job_select.currentData())
        if job:
            text = "\n".join(job["logs"])
            if (job["id"], text) != self.last_log:
                scrollbar = self.logs.verticalScrollBar()
                at_bottom, position = scrollbar.value() >= scrollbar.maximum() - 4, scrollbar.value()
                self.logs.setPlainText(text)
                scrollbar.setValue(scrollbar.maximum() if at_bottom else position)
                self.last_log = job["id"], text
            self.job_status.setText(job["status"].capitalize() + "  •  " + job["started"])
            self.open_export_button.setVisible(job["action"] == "export" and job["status"] == "completed")
        for item in items:
            if item["status"] != "running" and item["id"] not in self.processed_jobs:
                self.processed_jobs.add(item["id"])
                if item["action"] == "predict" and item["status"] == "completed":
                    self.sample_result(item)
                self.refresh()
        prep = next((j for j in items if j["action"] == "prepare"), None)
        if prep:
            stage = stage_progress(prep["logs"])
            if prep["status"] == "running" and stage:
                name, count, total = stage
                self.extraction_progress.setRange(0, max(1, total))
                self.extraction_progress.setValue(count)
                self.extraction_status.setText(f"{name}: {count:,} / {total:,}  •  {prep['category']}")
            else:
                self.extraction_progress.setRange(0, 0 if prep["status"] == "running" else 100)
                self.extraction_progress.setValue(100 if prep["status"] == "completed" else 0)
                self.extraction_status.setText(f"{prep['category']}  •  {prep['status']}  •  {prep['dataset']}")
        if self.training_path:
            history = read_history(self.training_path)
            self.accuracy_plot.set_history(history)
            self.loss_plot.set_history(history)
            count = len(history["loss"])
            self.training_progress.setRange(0, self.training_epochs)
            self.training_progress.setValue(count)
            training = self.last_jobs.get(self.training_job_id)
            status = training["status"] if training else "saved run"
            self.epoch_label.setText(f"Epoch {count} / {self.training_epochs}  •  {status}")
            if count:
                self.training_metrics.setText(f"Train accuracy {percent(history['accuracy'][-1] if history['accuracy'] else None)}   •   "
                    f"Validation {percent(history['val_accuracy'][-1] if history['val_accuracy'] else None)}   •   Loss {history['loss'][-1]:.4f}")

    def populate_runs(self):
        selected = self.selected_run()
        selected_id = selected["id"] if selected else None
        self.run_table.blockSignals(True)
        self.run_table.setRowCount(len(self.runs))
        for index, row in enumerate(self.runs):
            cells = [row["category"] + " / " + row["name"], percent(row["accuracy"]), percent(row["f1"]),
                str(row["classes"]), "Saved" if row["checkpoint"] else "Missing", self.records.get(row["id"], {}).get("decision", "Not tested")]
            for col, value in enumerate(cells):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, row["id"])
                self.run_table.setItem(index, col, item)
        self.run_table.blockSignals(False)
        index = next((i for i, r in enumerate(self.runs) if r["id"] == selected_id), 0)
        if self.runs:
            self.run_table.selectRow(index)
            self.select_run()
        else:
            self.run_detail.clear()
            self.model_caption.setText('Select a saved run.')
            self.review_run_id = None

    def selected_run(self):
        index = self.run_table.currentRow()
        item = self.run_table.item(index, 0) if index >= 0 else None
        return next((r for r in self.runs if r["id"] == item.data(Qt.ItemDataRole.UserRole)), None) if item else None

    def select_run(self):
        row = self.selected_run()
        if not row:
            return
        self.model_caption.setText(row["category"] + " / " + row["name"] + f"\n{row['classes']} gesture classes")
        self.run_detail.setPlainText(json.dumps({"run": row["id"], "accuracy": row["accuracy"], "macro_f1": row["f1"],
            "classes": row["class_labels"], "metadata": row["metadata"], "benchmark": row["benchmark"]}, indent=2, ensure_ascii=False))
        record = self.review_drafts.get(row["id"], self.records.get(row["id"], {}))
        self.loading_review = True
        self.decision.setCurrentIndex(max(0, self.decision.findData(record.get("decision", "Not tested"))))
        self.notes.setPlainText(record.get("notes", ""))
        self.review_run_id = row["id"]
        self.loading_review = False

    def review_edited(self):
        if not self.loading_review and self.review_run_id:
            self.review_drafts[self.review_run_id] = {
                "decision": self.decision.currentData(), "notes": self.notes.toPlainText()}

    def require_run(self):
        row = self.selected_run()
        if not row:
            self.error("Select a saved run first.")
        return row

    def view_run_curves(self):
        row = self.require_run()
        if row:
            self.training_job_id = None
            self.training_path = self.root / row["id"]
            config = backend.read_json(self.training_path / "training_config.json", {})
            self.training_epochs = config.get("epochs", len(read_history(self.training_path)["loss"]) or 1)
            self.training_title.setText("Conv1D–BiLSTM / " + row["name"])
            self.tabs.setCurrentIndex(1)
            self.poll_jobs()

    def use_run(self):
        row = self.require_run()
        if row:
            self.predict_run.setCurrentIndex(self.predict_run.findData(row["id"]))
            self.tabs.setCurrentIndex(3)

    def save_review(self):
        row = self.require_run()
        if not row:
            return
        data = {**self.records.get(row["id"], {}), "run": row["id"],
                "decision": self.decision.currentData(), "notes": self.notes.toPlainText()}
        def saved(records):
            submitted = {key: data[key] for key in ("decision", "notes")}
            if self.review_drafts.get(row["id"]) == submitted:
                self.review_drafts.pop(row["id"], None)
            self.review_saved(records)
        self.background(lambda: save_record(self.root, data, self.runs), saved)

    def review_saved(self, records):
        self.records = records
        self.populate_runs()
        self.statusBar().showMessage("Model review saved.", 5000)

    def export_keras(self):
        row = self.require_run()
        if not row:
            return
        target, _ = QFileDialog.getSaveFileName(self, "Save Keras model bundle", str(self.root / "exports" / (row["name"] + ".zip")), "ZIP archive (*.zip)")
        if not target:
            return
        def save():
            with native_archive(self.root, row["id"], self.runs) as (handle, filename):
                Path(target).parent.mkdir(parents=True, exist_ok=True)
                with Path(target).open("wb") as destination:
                    shutil.copyfileobj(handle, destination)
            return target
        self.background(save, lambda path: self.statusBar().showMessage("Model saved: " + path, 8000))

    def model_job(self, action):
        row = self.require_run()
        if row:
            self.request_job({"action": action, "run": row["id"]})

    def save_text(self, title, filename, text, file_filter):
        target, _ = QFileDialog.getSaveFileName(self, title, str(self.root / "exports" / filename), file_filter)
        if target:
            try:
                Path(target).parent.mkdir(parents=True, exist_ok=True)
                Path(target).write_text(text, encoding="utf-8-sig" if filename.endswith(".csv") else "utf-8")
                self.statusBar().showMessage("Saved: " + target, 8000)
            except OSError as error:
                self.error(str(error))

    def export_record(self):
        row = self.require_run()
        if row:
            try:
                result = export_model(self.root, row["id"], self.runs)
                self.save_text("Export model record", row["name"] + "-record.json", json.dumps(result, indent=2, ensure_ascii=False), "JSON (*.json)")
            except (ValueError, OSError) as error:
                self.error(str(error))

    def export_benchmarks(self):
        self.save_text("Export model benchmarks", "model-benchmarks.csv", benchmark_csv(self.runs), "CSV (*.csv)")

    def open_run_folder(self):
        row = self.require_run()
        if row:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.root / row["id"])))

    def open_export_folder(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.root / "exports" / "hudyat")))

    def choose_prediction_sample(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose a project sample", str(self.root / "data"), "Source samples (*)")
        if path:
            try:
                self.sample_path.setText(Path(path).resolve().relative_to(self.root.resolve()).as_posix())
            except ValueError:
                self.error("Choose a sample inside this project, or import its category first.")

    def predict_sample(self):
        self.request_job({"action": "predict", "run": self.predict_run.currentData(),
            "sample": self.sample_path.text().strip(), "hand_model": self.predict_hand_model.text().strip()})

    def sample_result(self, job):
        text = "\n".join(job["logs"])
        for index, character in enumerate(text):
            if character != "{":
                continue
            try:
                result, end = json.JSONDecoder().raw_decode(text[index:])
                if result and isinstance(result, dict) and all(isinstance(v, (int, float)) for v in result.values()):
                    self.show_prediction(result, job["run"], "sample", job["sample"])
                    return
            except ValueError:
                continue

    def show_prediction(self, probabilities, run, source, sample=""):
        self.last_prediction = {"run": run, "prediction": {"source": source, "sample": sample,
            "captured_at": datetime.now().isoformat(timespec="seconds"), "probabilities": probabilities}}
        ordered = sorted(probabilities.items(), key=lambda pair: -pair[1])
        self.prediction_table.setRowCount(min(5, len(ordered)))
        for i, (gesture, score) in enumerate(ordered[:5]):
            self.prediction_table.setItem(i, 0, QTableWidgetItem(gesture))
            self.prediction_table.setItem(i, 1, QTableWidgetItem(percent(score)))
        if ordered:
            self.prediction_title.setText(f"{ordered[0][0]}  ·  {percent(ordered[0][1])}")
        if self.camera_preview:
            self.preview_recognition.setText(self.prediction_title.text())
        self.save_prediction_button.setEnabled(bool(ordered))

    def prediction_model_changed(self):
        self.last_prediction = None
        self.save_prediction_button.setEnabled(False)
        self.prediction_title.setText("Recognition")
        self.prediction_table.setRowCount(0)
        if self.camera_preview:
            self.preview_recognition.setText("Recognition")

    def save_prediction(self):
        if self.last_prediction:
            data = dict(self.last_prediction)
            self.background(lambda: save_test(self.root, data, self.runs),
                lambda result: self.statusBar().showMessage(f"Saved prediction test: {result['test']['name']}", 6000))

    def start_camera(self):
        if self.job_starting:
            return self.error("Wait for job startup to finish before using the camera.")
        if not self.predict_run.currentData():
            return self.error("Choose a model with a saved checkpoint first.")
        if self.camera and self.camera.isRunning():
            return
        if any(j["status"] == "running" for j in self.jobs.snapshot()):
            return self.error("Wait for the running job, or stop it before starting the camera.")
        try:
            run = backend.local_path(self.predict_run.currentData())
            model = backend.local_path(self.predict_hand_model.text().strip()) if self.predict_hand_model.text().strip() else None
        except ValueError as error:
            return self.error(str(error))
        self.prediction_model_changed()
        self.camera_run_id = self.predict_run.currentData()
        timing = {"settle_ms": self.settle_ms.value(), "missing_ms": self.missing_ms.value(), "frame_interval_ms": self.interval_ms.value()}
        self.camera = CameraThread(self.live, run, self.camera_mode.currentData(), self.camera_device.value(), timing, model, self)
        self.camera.frame.connect(self.camera_frame)
        self.camera.status.connect(self.set_camera_status)
        self.camera.prediction.connect(self.camera_result)
        self.camera.failed.connect(self.error)
        self.camera.finished.connect(self.camera_finished)
        self.camera_button.setEnabled(False)
        self.camera_stop.setEnabled(True)
        self.camera_next.setEnabled(self.camera_mode.currentData() != "pose")
        self.predict_run.setEnabled(False)
        self.sync_camera_preview_controls()
        self.camera.start()
        self.poll_jobs()

    def camera_result(self, result):
        self.set_camera_status(result.get("message", result.get("status", "Processing…")))
        if result.get("probabilities"):
            self.show_prediction(result["probabilities"], self.camera_run_id, "live")

    def stop_camera(self):
        if self.camera:
            self.camera.stop()
            self.camera_stop.setEnabled(False)
            self.set_camera_status("Stopping camera…")
            self.sync_camera_preview_controls()

    def next_gesture(self):
        if self.camera:
            self.camera.reset_requested.set()

    def camera_finished(self):
        self.camera_button.setEnabled(True)
        self.camera_stop.setEnabled(False)
        self.camera_next.setEnabled(False)
        self.predict_run.setEnabled(True)
        self.set_camera_status("Camera stopped. Your last recognition result is kept.")
        self.sync_camera_preview_controls()
        self.poll_jobs()

    def camera_frame(self, frame, info=None):
        self.camera_canvas.set_frame(frame, info)
        if self.camera_preview and self.camera_preview.isVisible():
            self.preview_canvas.set_frame(frame, info)

    def set_camera_status(self, text):
        self.camera_status.setText(text)
        if self.camera_preview:
            self.preview_status.setText(text)

    def mirror_camera_preview(self, enabled):
        self.camera_canvas.set_mirrored(enabled)
        self.settings.setValue("mirror_preview", enabled)
        if self.camera_preview:
            self.preview_canvas.set_mirrored(enabled)
            self.preview_mirror.setChecked(enabled)

    def open_camera_preview(self):
        if self.camera_preview is None:
            self.camera_preview = QDialog(self)
            self.camera_preview.setWindowTitle("FSL Studio — Camera preview")
            layout = QVBoxLayout(self.camera_preview)
            controls = QHBoxLayout()
            controls.addWidget(label("Camera preview", heading=True), 1)
            self.preview_mirror = QCheckBox("Mirror preview")
            self.preview_mirror.toggled.connect(self.camera_mirror.setChecked)
            controls.addWidget(self.preview_mirror)
            self.preview_start = button("Start camera", self.start_camera, True)
            self.preview_stop = button("Stop camera", self.stop_camera)
            self.preview_next = button("Next gesture", self.next_gesture)
            for control in (self.preview_start, self.preview_stop, self.preview_next):
                controls.addWidget(control)
            controls.addWidget(button("Return to workspace (Esc)", self.camera_preview.close))
            layout.addLayout(controls)
            self.preview_canvas = MediaCanvas()
            self.preview_canvas.message = self.camera_canvas.message
            layout.addWidget(self.preview_canvas, 1)
            self.preview_recognition = label(self.prediction_title.text(), heading=True)
            layout.addWidget(self.preview_recognition)
            self.preview_status = label(self.camera_status.text(), muted=True)
            layout.addWidget(self.preview_status)
        self.preview_mirror.setChecked(self.camera_mirror.isChecked())
        self.preview_canvas.set_mirrored(self.camera_mirror.isChecked())
        self.preview_canvas.set_frame(self.camera_canvas.frame)
        self.preview_recognition.setText(self.prediction_title.text())
        self.sync_camera_preview_controls()
        self.camera_preview.showFullScreen()
        self.camera_preview.raise_()
        self.camera_preview.activateWindow()

    def sync_camera_preview_controls(self):
        if self.camera_preview:
            for source, target in ((self.camera_button, self.preview_start),
                                   (self.camera_stop, self.preview_stop), (self.camera_next, self.preview_next)):
                target.setEnabled(source.isEnabled())

    def closeEvent(self, event):
        if not self.jobs.request_close(lambda: QMessageBox.question(self, "Exit FSL Studio",
            "Stop the running job and exit? Partial outputs will be kept.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes):
            event.ignore()
            return
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("layout", self.saveState())
        self.settings.setValue("workspace", self.tabs.currentIndex())
        self.settings.setValue("source_root", self.source_root.text().strip())
        self.settings.setValue("dataset_root", self.dataset_root.text().strip())
        self.settings.sync()
        self.shutdown()
        event.accept()

    def shutdown(self):
        if self.closed:
            return
        self.closed = True
        self.timer.stop()
        if self.camera_preview:
            self.camera_preview.close()
        media_workers = list(self.media_workers)
        for worker in media_workers:
            worker.stop()
        if self.camera:
            self.camera.stop()
        self.live.shutdown()
        self.jobs.shutdown()
        for worker in (*media_workers, self.camera):
            if worker:
                worker.wait()
        self.pool.waitForDone()
