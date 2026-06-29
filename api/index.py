"""
Vercel entrypoint for SDE-OptionLab (Streamlit).

Vercel's Python runtime requires a module-level WSGI/ASGI callable named
`app`.  Streamlit is not a WSGI framework, so we wrap it: when Vercel hits
this handler we return a redirect page that points users to the live
Streamlit app running on Streamlit Community Cloud (or another host that
natively supports Streamlit).

If you want the full interactive app, deploy it on:
  • Streamlit Community Cloud – https://streamlit.io/cloud  (free, one-click)
  • Render.com                 – https://render.com         (free tier, Docker)
"""

import json
from http.server import BaseHTTPRequestHandler


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>SDE-OptionLab</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
      font-family: 'Segoe UI', system-ui, sans-serif;
      color: #e2e8f0;
    }
    .card {
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 20px;
      padding: 3rem 2.5rem;
      max-width: 520px;
      width: 90%;
      text-align: center;
      backdrop-filter: blur(12px);
      box-shadow: 0 25px 50px rgba(0,0,0,0.5);
    }
    .icon { font-size: 3.5rem; margin-bottom: 1rem; }
    h1 { font-size: 2rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.5rem; }
    .sub { color: #94a3b8; font-size: 1rem; margin-bottom: 2rem; line-height: 1.6; }
    .badge {
      display: inline-block;
      background: rgba(99,102,241,0.15);
      border: 1px solid rgba(99,102,241,0.4);
      color: #a5b4fc;
      border-radius: 999px;
      padding: 0.25rem 0.9rem;
      font-size: 0.8rem;
      margin-bottom: 2rem;
    }
    .note {
      font-size: 0.85rem;
      color: #64748b;
      margin-top: 1.5rem;
      line-height: 1.5;
    }
  </style>
</head>
<body>
  <div class="card">
    <div class="icon">📈</div>
    <h1>SDE-OptionLab</h1>
    <div class="badge">Streamlit Application</div>
    <p class="sub">
      This is a Streamlit-powered app and requires a platform that supports
      persistent Python servers.<br/><br/>
      To run it, deploy to <strong>Streamlit Community Cloud</strong> or
      <strong>Render.com</strong> for the full interactive experience.
    </p>
    <p class="note">
      GitHub → <strong>SyedaYasmeen22/SDE-OptionLab</strong><br/>
      Run locally: <code>streamlit run app.py</code>
    </p>
  </div>
</body>
</html>
"""


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML.encode("utf-8"))

    def do_POST(self):
        self.do_GET()
