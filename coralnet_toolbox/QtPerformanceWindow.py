import psutil
import GPUtil
import collections
import time
import platform

import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout, QVBoxLayout, QSizePolicy
from PyQt5.QtCore import QTimer


# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------

HISTORY_SIZE = 60
UPDATE_INTERVAL_MS = 500

# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class PerformanceWindow(QWidget):
    """
    A widget to monitor and display real-time CPU, Memory, and GPU usage.
    The monitoring starts/stops automatically when the widget is shown/hidden.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Tooltip for context if hovered
        self.setToolTip("Real-time hardware performance monitor.")

        # --- Data Storage ---
        # Use collections.deque for efficient, fixed-size data storage
        self.history_size = HISTORY_SIZE  # Number of data points to display
        self.cpu_data = collections.deque([0] * self.history_size, maxlen=self.history_size)
        self.mem_data = collections.deque([0] * self.history_size, maxlen=self.history_size)
        self.gpu_data = collections.deque([0] * self.history_size, maxlen=self.history_size)

        # Per-core CPU data
        self.cpu_cores = psutil.cpu_count(logical=True)
        self.per_core_data = []
        for _ in range(self.cpu_cores):
            self.per_core_data.append(collections.deque([0] * self.history_size, maxlen=self.history_size))

        # Disk I/O data
        self.disk_read_data = collections.deque([0] * self.history_size, maxlen=self.history_size)
        self.disk_write_data = collections.deque([0] * self.history_size, maxlen=self.history_size)
        self.prev_disk_io = psutil.disk_io_counters()
        self.prev_time = time.monotonic()
        
        # Network traffic data
        self.net_sent_data = collections.deque([0] * self.history_size, maxlen=self.history_size)
        self.net_recv_data = collections.deque([0] * self.history_size, maxlen=self.history_size)
        self.prev_net_io = psutil.net_io_counters()
        
        # X-axis data (time steps)
        self.x_axis = list(range(self.history_size))

        # --- Initialize UI ---
        self.setup_ui()

        # --- Timer for Real-Time Updates ---
        # Set up a QTimer to trigger the update_plots method every UPDATE_INTERVAL_MS
        # We do NOT start the timer here. It will be started by the showEvent.
        self.timer = QTimer()
        self.timer.setInterval(UPDATE_INTERVAL_MS)
        self.timer.timeout.connect(self.update_plots)

    def setup_ui(self):
        """
        Initializes the user interface, setting up the layout and plots.
        """
        # --- Layout ---
        # Apply the layout directly to 'self'
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # --- Styling (light background with bold black text; neon graph colors) ---
        light_bg_hex = '#F8F9FA'  # rgba(248,249,250)
        pg.setConfigOption('background', light_bg_hex)
        pg.setConfigOption('foreground', '#000000')
        # Set widget background to match theme
        self.setStyleSheet("background-color: rgba(248,249,250,1); color: #000000;")

        # --- Compact Header (CPU / Memory / GPU) ---
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(12)

        # hardware info, small labels
        cpu_info = platform.processor() or "CPU"
        total_memory = round(psutil.virtual_memory().total / (1024**3), 1)
        try:
            gpus = GPUtil.getGPUs()
            gpu_info = gpus[0].name if gpus else "No GPU"
        except Exception:
            gpu_info = "No GPU"

        info_label_style = (
            "color: #000000;"
            "font-size: 10pt;"
            "font-family: 'Consolas', 'Courier New', monospace;"
            "font-weight: bold;"
        )

        self.cpu_info_label = QLabel(f"{cpu_info}")
        self.cpu_info_label.setStyleSheet(info_label_style)
        self.mem_info_label = QLabel(f"{total_memory:.1f} GB")
        self.mem_info_label.setStyleSheet(info_label_style)
        self.gpu_info_label = QLabel(f"{gpu_info}")
        self.gpu_info_label.setStyleSheet(info_label_style)

        # Helper for compact metric widget
        def make_compact_metric(name, initial_value, color):
            w = QWidget()
            v = QVBoxLayout(w)
            v.setContentsMargins(4, 2, 4, 2)
            v.setSpacing(2)
            title = QLabel(name)
            title.setStyleSheet("font-weight:700; font-size:10pt; color: #000000; font-family: 'Consolas', monospace;")
            val = QLabel(initial_value)
            val.setStyleSheet("font-weight:800; font-size:12pt; color: #000000; font-family: 'Consolas', monospace;")

            # tiny sparkline
            spark = pg.PlotWidget()
            spark.setBackground(light_bg_hex)
            spark.setFixedHeight(36)
            spark.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            spark.getPlotItem().hideAxis('left')
            spark.getPlotItem().hideAxis('bottom')
            spark.setMouseEnabled(False, False)
            spark.setMenuEnabled(False)
            # curve pen color will be provided by caller via `color` argument
            curve = spark.plot([0] * self.history_size, pen=pg.mkPen(color, width=1.5))

            v.addWidget(title)
            v.addWidget(val)
            v.addWidget(spark)
            return w, val, curve

        # Graph colors: CPU=yellow, Memory=cyan, GPU=magenta
        cpu_color = '#00A8E6' # FFD400'   # yellow
        mem_color = '#00A8E6' # 00E5FF'   # cyan
        gpu_color = '#00A8E6' # FF00CC'   # magenta

        cpu_widget, self.cpu_value_label, self.cpu_curve = make_compact_metric("CPU", "0%", cpu_color)
        mem_widget, self.mem_value_label, self.mem_curve = make_compact_metric("Memory", "0%", mem_color)
        gpu_widget, self.gpu_value_label, self.gpu_curve = make_compact_metric("GPU", "0%", gpu_color)

        header_layout.addWidget(cpu_widget)
        header_layout.addWidget(mem_widget)
        header_layout.addWidget(gpu_widget)

        # No advanced area: keep UI minimal (CPU / Memory / GPU only)
        layout.addWidget(header)

    def update_plots(self):
        """
        Fetches new system stats and updates the plot data.
        This method is called by the QTimer.
        """
        curr_time = time.monotonic()
        time_delta = curr_time - self.prev_time
        
        # --- Get CPU, Memory and GPU percentages ---
        # CPU usage
        cpu_percent = psutil.cpu_percent()
        self.cpu_data.append(cpu_percent)
        self.cpu_value_label.setText(f"{cpu_percent:.1f}%")
        # update sparkline
        try:
            self.cpu_curve.setData(list(self.cpu_data))
        except Exception:
            pass

        # Per-core CPU usage (kept in memory but not shown in minimal UI)
        # Still update internal per-core data so it can be exposed later if needed
        per_core = psutil.cpu_percent(percpu=True)
        for i, usage in enumerate(per_core):
            if i < len(self.per_core_data):
                self.per_core_data[i].append(usage)

        # Memory usage
        mem_percent = psutil.virtual_memory().percent
        self.mem_data.append(mem_percent)
        self.mem_value_label.setText(f"{mem_percent:.1f}%")
        try:
            self.mem_curve.setData(list(self.mem_data))
        except Exception:
            pass

        # GPU usage
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_percent = gpus[0].load * 100
                self.gpu_value_label.setText(f"{gpu_percent:.1f}%")
            else:
                gpu_percent = 0
                self.gpu_value_label.setText("0%")
        except Exception:
            gpu_percent = 0
            self.gpu_value_label.setText("0%")

        self.gpu_data.append(gpu_percent)
        try:
            self.gpu_curve.setData(list(self.gpu_data))
        except Exception:
            pass
        
        # Disk I/O and Network counters still tracked internally but not rendered
        curr_disk_io = psutil.disk_io_counters()
        if curr_disk_io and self.prev_disk_io and time_delta > 0:
            read_bytes = curr_disk_io.read_bytes - self.prev_disk_io.read_bytes
            write_bytes = curr_disk_io.write_bytes - self.prev_disk_io.write_bytes
            read_mb_per_sec = (read_bytes / time_delta) / (1024**2)
            write_mb_per_sec = (write_bytes / time_delta) / (1024**2)
            self.disk_read_data.append(read_mb_per_sec)
            self.disk_write_data.append(write_mb_per_sec)
        self.prev_disk_io = curr_disk_io

        curr_net_io = psutil.net_io_counters()
        if curr_net_io and self.prev_net_io and time_delta > 0:
            sent_bytes = curr_net_io.bytes_sent - self.prev_net_io.bytes_sent
            recv_bytes = curr_net_io.bytes_recv - self.prev_net_io.bytes_recv
            sent_mb_per_sec = (sent_bytes / time_delta) / (1024**2)
            recv_mb_per_sec = (recv_bytes / time_delta) / (1024**2)
            self.net_sent_data.append(sent_mb_per_sec)
            self.net_recv_data.append(recv_mb_per_sec)
        self.prev_net_io = curr_net_io

        self.prev_time = curr_time

    # --- Event Handlers to Control Monitoring ---
    
    def showEvent(self, event):
        """
        Overrides the QWidget's showEvent.
        Starts the timer when the window is shown.
        """
        super().showEvent(event)
        if not self.timer.isActive():
            self.timer.start()

    def hideEvent(self, event):
        """
        Overrides the QWidget's hideEvent.
        Stops the timer when the window is hidden (e.g., minimized).
        """
        super().hideEvent(event)
        if self.timer.isActive():
            self.timer.stop()

    def closeEvent(self, event):
        """
        Overrides the QMainWindow's closeEvent.
        Ensures the timer is stopped when the window is closed.
        """
        if self.timer.isActive():
            self.timer.stop()
        super().closeEvent(event)