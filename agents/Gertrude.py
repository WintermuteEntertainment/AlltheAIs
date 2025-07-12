#Gertrude.py

from flask import Flask, request, jsonify, render_template, Response
from llama_cpp import Llama
import threading
import random
import time
import io
import pygame
from gtts import gTTS
import os

# === Configuration ===
MODEL_PATH = r"A:\gertrude_phi2_finetune\gguf_out\Gertrude-fixed.gguf"
PORT = 5000
MAX_TOKENS = 256

# === Flask Setup ===
app = Flask(__name__, static_folder="static", template_folder="templates")

# === CORS Helpers ===
def _build_cors_preflight_response():
    response = Response()
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    return response

def _corsify_response(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

# === Load GGUF model ===
print("🚗 Loading Gertrude GGUF model...")
llm = Llama(model_path=r"A:\gertrude_phi2_finetune\gguf_out\Gertrude-fixed.gguf", n_ctx=2048, n_threads=8, n_gpu_layers=35)
print("✅ Model loaded.")

# === Audio ===
pygame.mixer.init()

# === Voltage Simulation ===
voltage_base = 12.8
last_voltage_update = time.time()

def get_voltage():
    global voltage_base, last_voltage_update
    if time.time() - last_voltage_update > 30:
        voltage_base = random.uniform(12.4, 13.2)
        last_voltage_update = time.time()
    fluctuation = random.uniform(-0.2, 0.3)
    load_factor = 0.3 if time.time() % 12 < 4 else 0.1
    return round(voltage_base + fluctuation - load_factor, 1)

# === Routes ===
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/completions", methods=["POST", "OPTIONS"])
def completions():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()

    data = request.json
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400

    try:
        full_prompt = (
            "You are Gertrude, a helpful and opinionated in-car assistant.\n\n"
            f"User: {prompt}\nGertrude:"
        )
        result = llm(full_prompt, max_tokens=MAX_TOKENS, stop=["User:", "Gertrude:"], echo=False)
        reply = result["choices"][0]["text"].strip()
        return _corsify_response(jsonify({"text": reply}))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/speak", methods=["POST", "OPTIONS"])
def speak():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()

    data = request.json
    text = data.get("text", "")
    lang = data.get("lang", "en")
    if not text:
        return jsonify({"error": "Missing text"}), 400

    try:
        tts = gTTS(text=text, lang=lang)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        return Response(audio_buffer.read(), mimetype="audio/mpeg")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/voltage", methods=["GET", "OPTIONS"])
def voltage():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()
    return _corsify_response(jsonify({"voltage": get_voltage()}))

@app.route("/location", methods=["POST", "OPTIONS"])
def location():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()

    try:
        data = request.get_json()
        print("📍 Received GPS data:", data)
        return _corsify_response(jsonify({"status": "received"}))
    except Exception as e:
        print("❌ Error in /location:", e)
        return _corsify_response(jsonify({"error": str(e)})), 500

# === Entry Point ===
if __name__ == "__main__":
    print(f"Gertrude is running on http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT)
