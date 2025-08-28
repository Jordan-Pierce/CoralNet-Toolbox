import psutil
import GPUtil
import collections
import time

import pyqtgraph as pg
from PyQt5.QtWidgets import QMainWindow, QGridLayout, QWidget, QLabel
from PyQt5.QtCore import QTimer, Qt
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
        self.setGeometry(100, 100, 1200, 900)

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
        self.prev_time = time.time()
        
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
        pg.setConfigOption('background', '#1F1F1F')
        pg.setConfigOption('foreground', 'w')
        label_style = {"color": "#CCCCCC", "font-size": "14pt"}
        title_style = {"color": "#FFFFFF", "font-size": "16pt", "font-weight": "bold"}

        # --- Stats Labels ---
        self.stats_widget = QWidget()
        stats_layout = QGridLayout(self.stats_widget)
        
        # CPU absolute values
        self.cpu_label = QLabel("CPU: N/A")
        self.cpu_label.setStyleSheet("color: white; font-size: 12pt;")
        stats_layout.addWidget(self.cpu_label, 0, 0)
        
        # Memory absolute values
        self.mem_label = QLabel("Memory: N/A")
        self.mem_label.setStyleSheet("color: white; font-size: 12pt;")
        stats_layout.addWidget(self.mem_label, 0, 1)
        
        # GPU absolute values
        self.gpu_label = QLabel("GPU: N/A")
        self.gpu_label.setStyleSheet("color: white; font-size: 12pt;")
        stats_layout.addWidget(self.gpu_label, 0, 2)
        
        layout.addWidget(self.stats_widget, 0, 0, 1, 2)

        # Helper function to add bottom space for x-axis
        def configure_plot(plot_widget):
            plot_widget.getPlotItem().getAxis('bottom').setHeight(40)
            plot_widget.getPlotItem().layout.setContentsMargins(10, 10, 10, 30)
            return plot_widget
            
        # Helper function to create a value label for a plot
        def create_value_label(plot_widget, initial_text="Current: N/A"):
            text_item = pg.TextItem(text=initial_text, color='w', anchor=(0, 0))
            font = QFont()
            font.setBold(True)
            text_item.setFont(font)
            plot_widget.addItem(text_item)
            # Position at top-right corner with some margin
            text_item.setPos(80, 10)
            return text_item

        # --- CPU Plot ---
        self.cpu_plot_widget = pg.PlotWidget()
        self.cpu_plot_widget.setTitle("CPU Usage (%)", **title_style)
        self.cpu_plot_widget.setLabel("left", "Usage", units="%", **label_style)
        self.cpu_plot_widget.setLabel("bottom", "Time (s)", **label_style)
        self.cpu_plot_widget.setYRange(0, 100, padding=0.05)
        self.cpu_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.cpu_curve = self.cpu_plot_widget.plot(self.x_axis, list(self.cpu_data), pen=pg.mkPen('#00A3FF', width=3))
        self.cpu_value_label = create_value_label(self.cpu_plot_widget, "Current: 0%")
        configure_plot(self.cpu_plot_widget)
        layout.addWidget(self.cpu_plot_widget, 1, 0)

        # --- Per-Core CPU Plot ---
        self.core_plot_widget = pg.PlotWidget()
        self.core_plot_widget.setTitle("Per-Core CPU Usage (%)", **title_style)
        self.core_plot_widget.setLabel("left", "Usage", units="%", **label_style)
        self.core_plot_widget.setLabel("bottom", "Time (s)", **label_style)
        self.core_plot_widget.setYRange(0, 100, padding=0.05)
        self.core_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        self.per_core_curves = []
        for i in range(self.cpu_cores):
            color = pg.intColor(i, hues=self.cpu_cores)
            self.per_core_curves.append(
                self.core_plot_widget.plot(
                    self.x_axis, 
                    list(self.per_core_data[i]), 
                    pen=pg.mkPen(color, width=2), 
                    name=f"Core {i}"
                )
            )
        # Add a legend for core numbers
        legend = self.core_plot_widget.addLegend()
        for i, curve in enumerate(self.per_core_curves):
            legend.addItem(curve, f"Core {i}")
            
        configure_plot(self.core_plot_widget)
        layout.addWidget(self.core_plot_widget, 1, 1)

        # --- Memory Plot ---
        self.mem_plot_widget = pg.PlotWidget()
        self.mem_plot_widget.setTitle("Memory Usage (%)", **title_style)
        self.mem_plot_widget.setLabel("left", "Usage", units="%", **label_style)
        self.mem_plot_widget.setLabel("bottom", "Time (s)", **label_style)
        self.mem_plot_widget.setYRange(0, 100, padding=0.05)
        self.mem_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.mem_curve = self.mem_plot_widget.plot(self.x_axis, list(self.mem_data), pen=pg.mkPen('#FF5733', width=3))
        self.mem_value_label = create_value_label(self.mem_plot_widget, "Current: 0 GB")
        configure_plot(self.mem_plot_widget)
        layout.addWidget(self.mem_plot_widget, 2, 0)

        # --- Disk I/O Plot ---
        self.disk_plot_widget = pg.PlotWidget()
        self.disk_plot_widget.setTitle("Disk I/O (MB/s)", **title_style)
        self.disk_plot_widget.setLabel("left", "Transfer Rate", units="MB/s", **label_style)
        self.disk_plot_widget.setLabel("bottom", "Time (s)", **label_style)
        self.disk_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        self.disk_read_curve = self.disk_plot_widget.plot(
            self.x_axis, 
            list(self.disk_read_data), 
            pen=pg.mkPen('#FFDD33', width=3), 
            name="Read"
        )
        self.disk_write_curve = self.disk_plot_widget.plot(
            self.x_axis, 
            list(self.disk_write_data), 
            pen=pg.mkPen('#33DDFF', width=3), 
            name="Write"
        )
        
        # Add a legend
        legend = self.disk_plot_widget.addLegend()
        legend.addItem(self.disk_read_curve, "Read")
        legend.addItem(self.disk_write_curve, "Write")
        
        self.disk_value_label = create_value_label(self.disk_plot_widget, "R: 0 MB/s, W: 0 MB/s")
        configure_plot(self.disk_plot_widget)
        layout.addWidget(self.disk_plot_widget, 2, 1)

        # --- GPU Plot ---
        self.gpu_plot_widget = pg.PlotWidget()
        self.gpu_plot_widget.setTitle("GPU Usage (%)", **title_style)
        self.gpu_plot_widget.setLabel("left", "Usage", units="%", **label_style)
        self.gpu_plot_widget.setLabel("bottom", "Time (s)", **label_style)
        self.gpu_plot_widget.setYRange(0, 100, padding=0.05)
        self.gpu_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.gpu_curve = self.gpu_plot_widget.plot(self.x_axis, list(self.gpu_data), pen=pg.mkPen('#33FF57', width=3))
        self.gpu_value_label = create_value_label(self.gpu_plot_widget, "Current: 0%")
        configure_plot(self.gpu_plot_widget)
        layout.addWidget(self.gpu_plot_widget, 3, 0)

        # --- Network Traffic Plot ---
        self.net_plot_widget = pg.PlotWidget()
        self.net_plot_widget.setTitle("Network Traffic (MB/s)", **title_style)
        self.net_plot_widget.setLabel("left", "Transfer Rate", units="MB/s", **label_style)
        self.net_plot_widget.setLabel("bottom", "Time (s)", **label_style)
        self.net_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        self.net_sent_curve = self.net_plot_widget.plot(
            self.x_axis, 
            list(self.net_sent_data), 
            pen=pg.mkPen('#FF33DD', width=3), 
            name="Upload"
        )
        self.net_recv_curve = self.net_plot_widget.plot(
            self.x_axis, 
            list(self.net_recv_data), 
            pen=pg.mkPen('#DD33FF', width=3), 
            name="Download"
        )
        
        # Add a legend
        net_legend = self.net_plot_widget.addLegend()
        net_legend.addItem(self.net_sent_curve, "Upload")
        net_legend.addItem(self.net_recv_curve, "Download")
        
        self.net_value_label = create_value_label(self.net_plot_widget, "Up: 0 MB/s, Down: 0 MB/s")
        configure_plot(self.net_plot_widget)
        layout.addWidget(self.net_plot_widget, 3, 1)

    def update_plots(self):
        """
        Fetches new system stats and updates the plot data.
        This method is called by the QTimer.
        """
        curr_time = time.time()
        time_delta = curr_time - self.prev_time
        
        # --- Absolute Value Updates ---
        # CPU absolute values
        cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            cpu_freq_current = cpu_freq.current
        else:
            cpu_freq_current = 0
        
        # Memory absolute values (in GB)
        mem_info = psutil.virtual_memory()
        total_mem = mem_info.total / (1024**3)  # Convert to GB
        used_mem = mem_info.used / (1024**3)
        
        # GPU info
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_name = gpus[0].name
                gpu_temp = gpus[0].temperature
                gpu_mem_total = gpus[0].memoryTotal / 1024  # Convert to GB
                gpu_mem_used = gpus[0].memoryUsed / 1024  # Convert to GB
                gpu_info_str = f"GPU: {gpu_name} | "
                gpu_info_str += f"Temp: {gpu_temp}°C | "
                gpu_info_str += f"Memory: {gpu_mem_used:.1f}/{gpu_mem_total:.1f} GB"
            else:
                gpu_info_str = "GPU: Not available"
        except Exception:
            gpu_info_str = "GPU: Not available"
        
        # Update info labels
        self.cpu_label.setText(f"CPU: {cpu_count} cores | {cpu_freq_current:.0f} MHz")
        self.mem_label.setText(f"Memory: {used_mem:.1f}/{total_mem:.1f} GB ({mem_info.percent:.1f}%)")
        self.gpu_label.setText(gpu_info_str)
        
        # --- Get New Data ---
        # CPU usage
        cpu_percent = psutil.cpu_percent()
        self.cpu_data.append(cpu_percent)
        self.cpu_value_label.setText(f"Current: {cpu_percent:.1f}% ({cpu_freq_current:.0f} MHz)")
        
        # Per-core CPU usage
        per_core = psutil.cpu_percent(percpu=True)
        for i, usage in enumerate(per_core):
            if i < len(self.per_core_data):  # Ensure we don't exceed array bounds
                self.per_core_data[i].append(usage)
                self.per_core_curves[i].setData(self.x_axis, list(self.per_core_data[i]))

        # Memory usage
        mem_percent = psutil.virtual_memory().percent
        self.mem_data.append(mem_percent)
        self.mem_value_label.setText(f"Current: {mem_percent:.1f}% ({used_mem:.1f} GB)")

        # GPU usage (handle cases where no NVIDIA GPU is found)
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_percent = gpus[0].load * 100  # Get load of the first GPU
                gpu_mem_used = gpus[0].memoryUsed / 1024  # Convert to GB
                self.gpu_value_label.setText(f"Current: {gpu_percent:.1f}% ({gpu_mem_used:.1f} GB)")
            else:
                gpu_percent = 0  # Default to 0 if no GPU found
                self.gpu_value_label.setText("Current: 0% (No GPU)")
        except Exception:
            gpu_percent = 0  # Default to 0 on error
            self.gpu_value_label.setText("Current: 0% (No GPU)")
        
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
            
            # Auto-scale Y-axis based on max value
            max_io = max(max(self.disk_read_data), max(self.disk_write_data)) * 1.2
            if max_io > 0:
                self.disk_plot_widget.setYRange(0, max_io)
        
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
            
            # Auto-scale Y-axis based on max value
            max_net = max(max(self.net_sent_data), max(self.net_recv_data)) * 1.2
            if max_net > 0:
                self.net_plot_widget.setYRange(0, max_net)
        
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