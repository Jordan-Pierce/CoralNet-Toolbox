import sys
import os
import threading
import http.server
import socketserver
import argparse
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl

# --- 1. The HTML Template for the WebGL Viewer ---
# We use the open-source @mkkellogg/gaussian-splats-3d viewer via a CDN
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gaussian Splat Viewer</title>
    <style>
        body { margin: 0; overflow: hidden; background-color: #000000; }
        canvas { display: block; width: 100vw; height: 100vh; }
    </style>
</head>
<body>
    <script type="module">
        // esm.sh automatically resolves 'three.js' and other dependencies 
        // behind the scenes so we don't need an import map!
        import * as GaussianSplats3D from 'https://esm.sh/@mkkellogg/gaussian-splats-3d@1.1.2';

        // Initialize the viewer
        const viewer = new GaussianSplats3D.Viewer({
            'cameraUp': [0, -1, -0.6],
            'initialCameraPosition': [0, 0, -5],
            'initialCameraLookAt': [0, 0, 0],
            'dynamicScene': false
        });

        // Load the local PLY file served by our Python backend
        viewer.addSplatScene('/model.ply', {
            'splatAlphaCrop': 0.1,
            'showLoadingUI': true
        }).then(() => {
            viewer.start();
        });
    </script>
</body>
</html>
"""

# --- 2. Local HTTP Server to bypass CORS and serve the PLY ---
class SplatServerHandler(http.server.SimpleHTTPRequestHandler):
    ply_path = ""
    
    def do_GET(self):
        # Serve the HTML UI
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))
        
        # Serve the local PLY file directly to the JS engine
        elif self.path == '/model.ply':
            if not os.path.exists(self.ply_path):
                self.send_error(404, "PLY file not found")
                return
            
            self.send_response(200)
            self.send_header('Content-type', 'application/octet-stream')
            self.send_header('Content-Length', str(os.path.getsize(self.ply_path)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            with open(self.ply_path, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404, "File not found")

    # Suppress console logging for the web server
    def log_message(self, format, *args):
        pass

# --- 3. The PyQt5 Application ---
class MainWindow(QMainWindow):
    def __init__(self, ply_path):
        super().__init__()
        self.setWindowTitle("Web-Based Gaussian Splat Viewer")
        self.resize(1024, 768)

        # Setup the UI
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        if not ply_path or not os.path.exists(ply_path):
            error_label = QLabel(f"Error: Could not find PLY file at '{ply_path}'\nPlease provide a valid --input argument.")
            layout.addWidget(error_label)
            return

        # Start the background web server
        self.port = 8080
        SplatServerHandler.ply_path = ply_path
        self.server = socketserver.TCPServer(("", self.port), SplatServerHandler)
        
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True # Kills the server when the Qt app closes
        self.server_thread.start()

        # Initialize the WebEngine View and point it to our local server
        self.browser = QWebEngineView()
        self.browser.setUrl(QUrl(f"http://localhost:{self.port}"))
        layout.addWidget(self.browser)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="View a Gaussian splatting model using PyQtWebEngine.")
    parser.add_argument("--input", required=True, help="Path to the input PLY file.")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = MainWindow(args.input)
    window.show()
    sys.exit(app.exec_())