import psutil
import GPUtil
import collections
import time
import platform

import pyqtgraph as pg
from PyQt5.QtWidgets import QMainWindow, QGridLayout, QWidget, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

from coralnet_toolbox.Icons import get_icon


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class SystemMonitor(QMainWindow):
    """
    A PyQt5 application to monitor and display real-time CPU, Memory, and GPU usage.
    The monitoring starts/stops automatically when the window is shown/hidden.
    """
    def __init__(self):
        super().__init__()

        # --- Window Properties ---
        self.setWindowTitle("System Monitor")
        self.setWindowIcon(get_icon("system_monitor.png"))
        self.setGeometry(100, 100, 800, 900)

        # --- Data Storage ---
        # Use collections.deque for efficient, fixed-size data storage
        self.history_size = 100  # Number of data points to display
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
        self.setup_layout()

        # --- Timer for Real-Time Updates ---
        # Set up a QTimer to trigger the update_plots method every 1000 ms (1 second)
        # We do NOT start the timer here. It will be started by the showEvent.
        self.timer = QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.update_plots)

    def setup_layout(self):
        """
        Initializes the user interface, setting up the layout and plots.
        """
        # --- Central Widget and Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QGridLayout(central_widget)
        layout.setVerticalSpacing(20)  # Add more vertical spacing between plots

        # --- Styling ---
        pg.setConfigOption('background', 'w')  # White background for graphs
        pg.setConfigOption('foreground', '#000000')  # Black foreground elements in graphs
        plot_label_style = {"color": "#000000", "font-size": "14pt"}  # Black text for plot labels
        title_style = {"color": "#000000", "font-size": "16pt", "font-weight": "bold"}  # Black text

        # --- Stats Labels - Only Hardware Info ---
        self.stats_widget = QWidget()
        stats_layout = QGridLayout(self.stats_widget)
        stats_layout.setHorizontalSpacing(20)  # Add horizontal spacing between items
        stats_layout.setContentsMargins(10, 10, 10, 20)  # Add margins around the widget

        # Get system hardware information
        cpu_info = platform.processor() or "CPU"
        total_memory = round(psutil.virtual_memory().total / (1024**3), 1)
        
        try:
            gpus = GPUtil.getGPUs()
            gpu_info = gpus[0].name if gpus else "No GPU detected"
        except Exception:
            gpu_info = "GPU information unavailable"
        
        # Hardware info labels with consistent styling and alignment
        info_label_style = "color: black; font-size: 11pt; font-weight: bold; padding: 5px;"

        self.cpu_info_label = QLabel(f"CPU: {cpu_info}")
        self.cpu_info_label.setStyleSheet(info_label_style)
        self.cpu_info_label.setAlignment(Qt.AlignCenter)
        stats_layout.addWidget(self.cpu_info_label, 0, 0)
        
        self.mem_info_label = QLabel(f"Memory: {total_memory:.1f} GB")
        self.mem_info_label.setStyleSheet(info_label_style)
        self.mem_info_label.setAlignment(Qt.AlignCenter)
        stats_layout.addWidget(self.mem_info_label, 0, 1)
        
        self.gpu_info_label = QLabel(f"GPU: {gpu_info}")
        self.gpu_info_label.setStyleSheet(info_label_style)
        self.gpu_info_label.setAlignment(Qt.AlignCenter)
        stats_layout.addWidget(self.gpu_info_label, 0, 2)
        
        # Set column stretching to distribute space evenly
        stats_layout.setColumnStretch(0, 1)
        stats_layout.setColumnStretch(1, 1)
        stats_layout.setColumnStretch(2, 1)

        layout.addWidget(self.stats_widget, 0, 0, 1, 2)

        # Helper function to add bottom space for x-axis and enable auto-range
        def configure_plot(plot_widget, fixed_y_range=None):
            plot_widget.getPlotItem().getAxis('bottom').setHeight(40)
            plot_widget.getPlotItem().layout.setContentsMargins(10, 10, 10, 30)
            
            # If fixed_y_range is None, enable auto-ranging, otherwise set fixed range
            if fixed_y_range is None:
                plot_widget.enableAutoRange(axis='y')
            else:
                min_val, max_val = fixed_y_range
                plot_widget.setYRange(min_val, max_val, padding=0.05)
                
            return plot_widget
            
        # Helper function to create a value label for a plot
        def create_value_label(plot_widget, initial_text="Current: N/A"):
            text_item = pg.TextItem(text=initial_text, color='#000000', anchor=(0, 0))
            font = QFont()
            font.setBold(True)
            text_item.setFont(font)
            plot_widget.addItem(text_item)
            # Position at top-left corner with some margin
            text_item.setPos(10, 10)
            return text_item

        # --- CPU Plot --- (fixed 0-100 range)
        self.cpu_plot_widget = pg.PlotWidget()
        self.cpu_plot_widget.setTitle("CPU Usage (%)", **title_style)
        self.cpu_plot_widget.setLabel("left", "Usage", units="%", **plot_label_style)
        self.cpu_plot_widget.setLabel("bottom", "Time (s)", **plot_label_style)
        self.cpu_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.cpu_curve = self.cpu_plot_widget.plot(self.x_axis, list(self.cpu_data), pen=pg.mkPen('#00A3FF', width=3))
        self.cpu_value_label = create_value_label(self.cpu_plot_widget, "Current: 0%")
        configure_plot(self.cpu_plot_widget, fixed_y_range=(0, 100))
        layout.addWidget(self.cpu_plot_widget, 1, 0)

        # --- Per-Core CPU Plot --- (fixed 0-100 range)
        self.core_plot_widget = pg.PlotWidget()
        self.core_plot_widget.setTitle("Per-Core CPU Usage (%)", **title_style)
        self.core_plot_widget.setLabel("left", "Usage", units="%", **plot_label_style)
        self.core_plot_widget.setLabel("bottom", "Time (s)", **plot_label_style)
        self.core_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        self.per_core_curves = []
        for i in range(self.cpu_cores):
            color = pg.intColor(i, hues=self.cpu_cores)
            self.per_core_curves.append(
                self.core_plot_widget.plot(
                    self.x_axis, 
                    list(self.per_core_data[i]), 
                    pen=pg.mkPen(color, width=2)
                )
            )
        configure_plot(self.core_plot_widget, fixed_y_range=(0, 100))
        layout.addWidget(self.core_plot_widget, 1, 1)

        # --- Memory Plot --- (fixed 0-100 range)
        self.mem_plot_widget = pg.PlotWidget()
        self.mem_plot_widget.setTitle("Memory Usage (%)", **title_style)
        self.mem_plot_widget.setLabel("left", "Usage", units="%", **plot_label_style)
        self.mem_plot_widget.setLabel("bottom", "Time (s)", **plot_label_style)
        self.mem_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.mem_curve = self.mem_plot_widget.plot(self.x_axis, list(self.mem_data), pen=pg.mkPen('#FF5733', width=3))
        self.mem_value_label = create_value_label(self.mem_plot_widget, "Current: 0%")
        configure_plot(self.mem_plot_widget, fixed_y_range=(0, 100))
        layout.addWidget(self.mem_plot_widget, 2, 0)

        # --- Disk I/O Plot --- (auto-ranging)
        self.disk_plot_widget = pg.PlotWidget()
        self.disk_plot_widget.setTitle("Disk I/O (MB/s)", **title_style)
        self.disk_plot_widget.setLabel("left", "Transfer Rate", units="MB/s", **plot_label_style)
        self.disk_plot_widget.setLabel("bottom", "Time (s)", **plot_label_style)
        self.disk_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        self.disk_read_curve = self.disk_plot_widget.plot(
            self.x_axis, 
            list(self.disk_read_data), 
            pen=pg.mkPen('#FFDD33', width=3)
        )
        self.disk_write_curve = self.disk_plot_widget.plot(
            self.x_axis, 
            list(self.disk_write_data), 
            pen=pg.mkPen('#33DDFF', width=3)
        )
        
        self.disk_value_label = create_value_label(self.disk_plot_widget, "R: 0 MB/s, W: 0 MB/s")
        configure_plot(self.disk_plot_widget, fixed_y_range=None)  # Auto-range
        layout.addWidget(self.disk_plot_widget, 2, 1)

        # --- GPU Plot --- (fixed 0-100 range)
        self.gpu_plot_widget = pg.PlotWidget()
        self.gpu_plot_widget.setTitle("GPU Usage (%)", **title_style)
        self.gpu_plot_widget.setLabel("left", "Usage", units="%", **plot_label_style)
        self.gpu_plot_widget.setLabel("bottom", "Time (s)", **plot_label_style)
        self.gpu_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.gpu_curve = self.gpu_plot_widget.plot(self.x_axis, list(self.gpu_data), pen=pg.mkPen('#33FF57', width=3))
        self.gpu_value_label = create_value_label(self.gpu_plot_widget, "Current: 0%")
        configure_plot(self.gpu_plot_widget, fixed_y_range=(0, 100))
        layout.addWidget(self.gpu_plot_widget, 3, 0)

        # --- Network Traffic Plot --- (auto-ranging)
        self.net_plot_widget = pg.PlotWidget()
        self.net_plot_widget.setTitle("Network Traffic (MB/s)", **title_style)
        self.net_plot_widget.setLabel("left", "Transfer Rate", units="MB/s", **plot_label_style)
        self.net_plot_widget.setLabel("bottom", "Time (s)", **plot_label_style)
        self.net_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        self.net_sent_curve = self.net_plot_widget.plot(
            self.x_axis, 
            list(self.net_sent_data), 
            pen=pg.mkPen('#FF33DD', width=3)
        )
        self.net_recv_curve = self.net_plot_widget.plot(
            self.x_axis, 
            list(self.net_recv_data), 
            pen=pg.mkPen('#DD33FF', width=3)
        )
        
        self.net_value_label = create_value_label(self.net_plot_widget, "Up: 0 MB/s, Down: 0 MB/s")
        configure_plot(self.net_plot_widget, fixed_y_range=None)  # Auto-range
        layout.addWidget(self.net_plot_widget, 3, 1)

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
        self.cpu_value_label.setText(f"Current: {cpu_percent:.1f}%")
        
        # Per-core CPU usage
        per_core = psutil.cpu_percent(percpu=True)
        for i, usage in enumerate(per_core):
            if i < len(self.per_core_data):
                self.per_core_data[i].append(usage)
                self.per_core_curves[i].setData(self.x_axis, list(self.per_core_data[i]))

        # Memory usage
        mem_percent = psutil.virtual_memory().percent
        self.mem_data.append(mem_percent)
        self.mem_value_label.setText(f"Current: {mem_percent:.1f}%")

        # GPU usage
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_percent = gpus[0].load * 100
                self.gpu_value_label.setText(f"Current: {gpu_percent:.1f}%")
            else:
                gpu_percent = 0
                self.gpu_value_label.setText("Current: 0%")
        except Exception:
            gpu_percent = 0
            self.gpu_value_label.setText("Current: 0%")

        self.gpu_data.append(gpu_percent)
        
        # Disk I/O
        curr_disk_io = psutil.disk_io_counters()
        
        # Calculate read/write rates in MB/s
        if curr_disk_io and self.prev_disk_io:
            read_bytes = curr_disk_io.read_bytes - self.prev_disk_io.read_bytes
            write_bytes = curr_disk_io.write_bytes - self.prev_disk_io.write_bytes
            read_mb_per_sec = (read_bytes / time_delta) / (1024**2)
            write_mb_per_sec = (write_bytes / time_delta) / (1024**2)
            
            self.disk_read_data.append(read_mb_per_sec)
            self.disk_write_data.append(write_mb_per_sec)
            
            # Update the value label with current values
            self.disk_value_label.setText(f"R: {read_mb_per_sec:.1f} MB/s, W: {write_mb_per_sec:.1f} MB/s")
            
            # Let pyqtgraph handle auto-ranging automatically
            self.disk_plot_widget.enableAutoRange(axis='y')
    
        self.prev_disk_io = curr_disk_io
        
        # Network traffic
        curr_net_io = psutil.net_io_counters()
        if curr_net_io and self.prev_net_io:
            sent_bytes = curr_net_io.bytes_sent - self.prev_net_io.bytes_sent
            recv_bytes = curr_net_io.bytes_recv - self.prev_net_io.bytes_recv
            
            sent_mb_per_sec = (sent_bytes / time_delta) / (1024**2)
            recv_mb_per_sec = (recv_bytes / time_delta) / (1024**2)
            
            self.net_sent_data.append(sent_mb_per_sec)
            self.net_recv_data.append(recv_mb_per_sec)
            
            # Update the value label with current values
            self.net_value_label.setText(f"Up: {sent_mb_per_sec:.1f} MB/s, Down: {recv_mb_per_sec:.1f} MB/s")
            
            # Let pyqtgraph handle auto-ranging automatically
            self.net_plot_widget.enableAutoRange(axis='y')
    
        self.prev_net_io = curr_net_io

        # --- Update Plot Curves ---
        # Convert deque to list for plotting
        self.cpu_curve.setData(self.x_axis, list(self.cpu_data))
        self.mem_curve.setData(self.x_axis, list(self.mem_data))
        self.gpu_curve.setData(self.x_axis, list(self.gpu_data))
        self.disk_read_curve.setData(self.x_axis, list(self.disk_read_data))
        self.disk_write_curve.setData(self.x_axis, list(self.disk_write_data))
        self.net_sent_curve.setData(self.x_axis, list(self.net_sent_data))
        self.net_recv_curve.setData(self.x_axis, list(self.net_recv_data))

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