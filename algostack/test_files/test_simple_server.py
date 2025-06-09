#!/usr/bin/env python3
"""
Simple HTTP server to test WSL networking
"""

import http.server
import socket
import socketserver


def get_ip():
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    return local_ip


PORT = 8000


class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            html = """
            <html>
            <head><title>AlgoStack Test Server</title></head>
            <body>
            <h1>✅ WSL Networking is Working!</h1>
            <p>If you can see this, your WSL networking is fine.</p>
            <p>The dashboard should work too.</p>
            <hr>
            <p>Now try running: <code>./setup_and_run_dashboard.sh</code></p>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
        else:
            super().do_GET()


if __name__ == "__main__":
    with socketserver.TCPServer(("0.0.0.0", PORT), MyHandler) as httpd:
        ip = get_ip()
        print(f"🧪 Test server running on port {PORT}")
        print("\nTry these URLs:")
        print(f"  • http://localhost:{PORT}")
        print(f"  • http://127.0.0.1:{PORT}")
        print(f"  • http://{ip}:{PORT}")
        print("\nPress Ctrl+C to stop")
        httpd.serve_forever()
