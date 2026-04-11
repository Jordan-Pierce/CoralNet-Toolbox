import sys
import os
import threading
import http.server
import socketserver
import argparse
import shutil
import urllib.request
import re
import webbrowser
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

THREE_VERSION = "0.178.0"
JS_DEPS = {
    "three.module.js":  f"https://unpkg.com/three@{THREE_VERSION}/build/three.module.js",
    "three.core.js":    f"https://unpkg.com/three@{THREE_VERSION}/build/three.core.js",
    "three.webgpu.js":  f"https://unpkg.com/three@{THREE_VERSION}/build/three.webgpu.js",
    "spark.module.js":  "https://sparkjs.dev/releases/spark/0.1.10/spark.module.js",
    "OrbitControls.js": f"https://unpkg.com/three@{THREE_VERSION}/examples/jsm/controls/OrbitControls.js",
}

def download_file(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)

def patch_bare_specifiers(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    patched = content
    patched = re.sub(r'from\s*["\']three["\']',              'from "/three.module.js"', patched)
    patched = re.sub(r'import\s*\(\s*["\']three["\']\s*\)',  'import("/three.module.js")', patched)
    patched = re.sub(r'from\s*["\'][^"\']*three\.core\.js["\']',   'from "/three.core.js"',   patched)
    patched = re.sub(r'from\s*["\'][^"\']*three\.webgpu\.js["\']', 'from "/three.webgpu.js"', patched)
    if patched != content:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(patched)
        print(f"[SparkViewer] Patched {os.path.basename(filepath)}")

def ensure_js_deps():
    for filename, url in JS_DEPS.items():
        dest = os.path.join(SCRIPT_DIR, filename)
        if not os.path.exists(dest):
            print(f"[SparkViewer] Downloading {filename} ...")
            try:
                download_file(url, dest)
                print(f"[SparkViewer] Saved {filename}")
            except Exception as e:
                print(f"[SparkViewer] ERROR: {e}")
                sys.exit(1)
        patch_bare_specifiers(dest)

class SparkServerHandler(http.server.SimpleHTTPRequestHandler):
    ply_path = ""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SCRIPT_DIR, **kwargs)
    def do_GET(self):
        if self.path.split('?')[0].lower().endswith(('.ply', '.splat')):
            self.send_response(200)
            self.send_header('Content-type', 'application/octet-stream')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            with open(self.ply_path, 'rb') as f:
                shutil.copyfileobj(f, self.wfile)
        else:
            super().do_GET()
    def log_message(self, *args): pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"[SparkViewer] ERROR: File not found: {args.input}")
        sys.exit(1)

    ensure_js_deps()

    SparkServerHandler.ply_path = args.input
    server = socketserver.TCPServer(("", args.port), SparkServerHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    url = f"http://localhost:{args.port}/viewer.html"
    print(f"[SparkViewer] Serving on {url}")
    print(f"[SparkViewer] Model: {args.input}")
    print(f"[SparkViewer] Press Ctrl+C to stop.")

    time.sleep(0.5)
    webbrowser.open(url)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[SparkViewer] Shutting down.")
        server.shutdown()