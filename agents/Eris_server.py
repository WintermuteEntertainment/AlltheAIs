# =============================================================
#  ERIS – Experimental Self‑Modification INTEGRATED (v2 ie ERIS 68) ________________> ALETHEIA VERSION
# =============================================================

import importlib
import inspect
import os
import torch

os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Force offline mode
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Suppress TensorFlow logs
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'   # Better CUDA error reporting

import sys
import textwrap
import time
import types

from pathlib import Path
from typing import Callable, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import json
import logging
import gc
import requests
import re
import threading
import time
import traceback
from datetime import datetime
import urllib.parse
import subprocess
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from peft import PeftModel, PeftConfig

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

# Initialize Flask
app = Flask(__name__)
CORS(app)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
if not torch.cuda.is_available():
    torch.set_default_device('cpu')  # Force CPU mode
    print("CUDA not available - running in CPU mode")

eris_engine = None
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

import tensorflow as tf

if hasattr(tf, 'compat') and hasattr(tf.compat, 'v1'):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
else:
    tf.get_logger().setLevel('ERROR')



# ======================
# SELF-MODIFICATION CONFIG
# ======================
PATCH_DIR = Path(r"A:\ALETHEIA\AlltheAIs\agents\eris_pending_patches")
PATCH_DIR.mkdir(exist_ok=True)
ALLOWED_MODULES = {"eris_plugins", "eris_behaviours"}  # Modules safe to modify
MAX_PATCH_LINES = 1000

# ======================
# CONFIGURATION & SETUP
# ======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

MODEL_PATH = r"A:/ERISPARENTDIRECTORY/LLMs/models/erisv2merged"
MEMORY_DIR = "memory"
FACT_CHECK_LOG = "fact_checks.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(MEMORY_DIR, exist_ok=True)

# API Keys and Search Config
SEARCH_API_KEYS = {
    "bing": "01285bdb-1a74-47ea-84d3-a0025776eb4c",
    "google": "AIzaSyADMzj5EbAviQuL1OkK0QXKCiGn5PmC64c"
}

SEARCH_ENDPOINTS = {
    "bing": "https://api.bing.microsoft.com/v7.0/search",
    "google": "https://www.googleapis.com/customsearch/v1"
}

GOOGLE_CX = "8526ca60c63bf4143"
GOOGLE_TTS_API_KEY = "AIzaSyADMzj5EbAviQuL1OkK0QXKCiGn5PmC64c"
GOOGLE_TTS_API_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"

GOOGLE_TTS_VOICES = {
    "en-US-Wavenet-A": "Male (Natural)",
    "en-US-Wavenet-D": "Male (Deep)",
    "en-US-Wavenet-F": "Female (Soft)",
    "en-US-Wavenet-C": "Female (Clear)",
    "en-US-Neural2-J": "Male (Casual)",
    "en-US-News-L": "News Anchor"
}

# ======================
# SELF-MODIFIER CLASS
# ======================
class SelfModifier:
    """Controlled hot‑swap engine for ERIS self-modification"""
    
    def __init__(self, module_whitelist: set[str] | None = None):
        self.allowed = module_whitelist or ALLOWED_MODULES

    def propose_patch(self, module_name: str, replacement_code: str) -> Path:
        """Save candidate patch to disk"""
        if module_name not in self.allowed:
            raise PermissionError(f"Module {module_name} not in whitelist")
        
        normalized = textwrap.dedent(replacement_code).strip()
        if len(normalized.splitlines()) > MAX_PATCH_LINES:
            raise ValueError("Patch too large")
        
        timestamp = int(time.time())
        patch_file = PATCH_DIR / f"{module_name.replace('.', '_')}_{timestamp}.py.patch"
        patch_file.write_text(normalized, encoding="utf-8")
        return patch_file

    def review_patch(self, patch_path: Path, reviewer_fn: Optional[Callable[[str], bool]] = None) -> bool:
        """Review patch and return True to accept, False to reject"""
        code = patch_path.read_text(encoding="utf-8")
        
        if reviewer_fn:
            decision = bool(reviewer_fn(code))
        else:
            print(f"\n===== PATCH {patch_path.name} =====")
            print(code)
            decision = input("\nAccept this patch? [y/N] ").lower().startswith("y")
        
        if decision:
            print("Patch approved → will apply")
        else:
            print("Patch rejected → deleting")
            print(f"PATCH PATH EXISTS? {patch_path.exists()} at {patch_path}")
            patch_path.replace(PATCH_DIR / f"rejected_{patch_path.name}")
            

        return decision

    def apply_patch(self, module_name: str, patch_path: Path, auto_restart: bool = True):
        """Apply patch to target module"""
        if module_name not in self.allowed:
            raise PermissionError(f"Module {module_name} not in whitelist")
        
        target_spec = importlib.util.find_spec(module_name)
        if not target_spec or not target_spec.origin:
            raise ImportError(f"Cannot locate module: {module_name}")
        
        target_path = Path(target_spec.origin)
        backup_path = target_path.with_suffix(".bak")
        backup_path.write_text(target_path.read_text(encoding="utf-8"), encoding="utf-8")
        
        target_path.write_text(patch_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Applied patch to {target_path} (backup at {backup_path})")

        if module_name not in sys.modules:
            importlib.import_module(module_name)
        
        try:
            reloaded = importlib.reload(sys.modules[module_name])
            print(f"Module {module_name} reloaded → {reloaded}")
        except Exception as exc:
            print(f"Reload failed: {exc!r}")
            if auto_restart:
                print("Attempting graceful self‑restart…")
                self._self_restart()
            else:
                raise
                
    def _self_restart(self):
        """Exec‑replace current Python process"""
        os.execv(sys.executable, [sys.executable] + sys.argv)

# ======================
# WEB SEARCH CLASS (v68)
# ======================
class WebSearch:
    def __init__(self):
        self.search_cache = {}
        self.error_log = []

    def perform_search(self, query, engine="google", num_results=3):
        if len(query.split()) < 4:
            return []
        
        cache_key = f"{engine}:{re.sub(r'\W+', '', query).lower()}"
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]
            
        try:
            processed_query = self._preprocess_query(query)
            params, headers = self._build_request(engine, processed_query, num_results)
            
            response = requests.get(
                SEARCH_ENDPOINTS[engine],
                headers=headers,
                params=params,
                timeout=15
            )
            response.raise_for_status()
            results = self._parse_results(engine, response.json())
            validated_results = [r for r in results if self._validate_result(r)]
            self.search_cache[cache_key] = validated_results[:num_results]
            return self.search_cache[cache_key]
            
        except Exception as e:
            logging.error(f"Search error ({engine}): {str(e)}")
            return []

    def _build_request(self, engine, query, num_results):
        params = {"q": query, "num": min(num_results, 10)}
        headers = {}
        
        if engine == "google":
            params.update({
                "key": SEARCH_API_KEYS["google"],
                "cx": GOOGLE_CX,
                "num": min(num_results, 10)
            })
        elif engine == "bing":
            headers = {"Ocp-Apim-Subscription-Key": SEARCH_API_KEYS["bing"]}
            
        return params, headers

    def _parse_results(self, engine, data):
        if engine == "google":
            return [{
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet")
            } for item in data.get("items", [])]
        elif engine == "bing":
            return [{
                "title": item.get("name"),
                "link": item.get("url"),
                "snippet": item.get("snippet")
            } for item in data.get("webPages", {}).get("value", [])]
        return []

    # Keep other WebSearch methods from v65 unchanged

    def _preprocess_query(self, query):
            query = re.sub(r'\b(?:a|an|the|how|to|of)\b', '', query)
            return ' '.join(query.split()[:7])

    def _validate_result(self, result):
        return all(key in result for key in ["title", "link", "snippet"])


# ======================
# MODEL HANDLING
# ======================

def load_model():
    model_path = os.path.abspath(MODEL_PATH)  # Get absolute path
    logging.info(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    # Explicit local_files_only flag is CRITICAL
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        local_files_only=True  # MUST ADD THIS FLAG
    )
    return tokenizer, model

# ======================
# CORE COMPONENTS
# ======================
class FactChecker:
    def __init__(self):
        self.source_weights = {
            "wikipedia": 0.7,  # Increased from 0.5
            "google_factcheck": 0.4  # Increased from 0.3
        }
        self.thresholds = {
            "direct_match": 0.8,
            "partial_match": 0.4
        }

    def _clean_statement(self, text):
        """Enhanced text cleaning for fact-checking"""
        # Remove special command patterns first
        text = re.sub(r'\[\[.*?\]\]', '', text)
        # Standard cleaning
        text = re.sub(r'[^\w\s\-.,!?]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:500]  # Truncate to 500 characters

    def verify_statement(self, statement):
        try:
            clean_stmt = self._clean_statement(statement)
            if not clean_stmt:
                return 0.0, {"error": "empty statement"}

            verification_results = {}
            total_score = 0.0
        
            # Wikipedia check with sentence search
            wiki_score = self._check_wikipedia(clean_stmt)
            verification_results["wikipedia"] = wiki_score
            total_score += wiki_score * self.source_weights["wikipedia"]

            # Google Fact Check Tools
            google_score = self._check_google_factcheck(clean_stmt)
            verification_results["google"] = google_score
            total_score += google_score * self.source_weights["google_factcheck"]

            print(f"Wikipedia: {wiki_score}, Google: {google_score}")

            # Normalize to 0-1 range
            final_confidence = min(max(total_score, 0.0), 1.0)
    
            return round(final_confidence, 2), verification_results

        except Exception as e:
            logging.error(f"Fact-checking error: {str(e)}")
            return 0.0, {"error": str(e)}

    def _check_wikipedia(self, text):
        try:
            # Search for exact sentence match
            url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(text)}&srlimit=1&format=json"
            response = requests.get(url, timeout=8)
            data = response.json()
            
            if 'query' in data:
                for result in data['query']['search']:
                    # Get page content and check for exact phrase
                    content_url = f"https://en.wikipedia.org/w/api.php?action=parse&pageid={result['pageid']}&format=json"
                    content_resp = requests.get(content_url, timeout=8)
                    content = content_resp.json().get('parse', {}).get('text', {}).get('*', '')
                    
                    # Calculate match strength
                    matches = re.findall(re.escape(text.lower()), content.lower())
                    if matches:
                        return min(1.0, len(matches) * 0.2)  # 0.2 per match up to 1.0
            return 0.0
        except Exception as e:
            logging.error(f"Wikipedia check failed: {str(e)}")
            return 0.0

    def _check_google_factcheck(self, text):
        try:
            url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
            params = {
                "query": text,
                "key": "AIzaSyADMzj5EbAviQuL1OkK0QXKCiGn5PmC64c",
                "languageCode": "en"
            }
            response = requests.get(url, params=params, timeout=10)
            claims = response.json().get('claims', [])
            
            if claims:
                # Calculate based on claim review ratings
                ratings = [c.get('claimReview', [{}])[0].get('textualRating', '') for c in claims]
                score_map = {
                    "true": 1.0,
                    "mostly true": 0.8,
                    "half true": 0.5,
                    "mostly false": 0.3,
                    "false": 0.0,
                    "pants on fire": 0.0
                }
                return sum(score_map.get(r.lower(), 0.0) for r in ratings) / len(ratings)
            return 0.0
        except Exception as e:
            logging.error(f"Google FactCheck failed: {str(e)}")
            return 0.0

class MemoryManager:
    def __init__(self):
        self.memory_buffer = []
        self.buffer_lock = threading.Lock()
        self.buffer_size = 5
        self._active = True  # Shutdown flag
        self.self_modifier = SelfModifier() #SelfModifier class definition

    def close(self):  # Explicit close method
        self._active = False
        self._flush_buffer()

    def get_short_term(self, max_entries=15):
        """Retrieve short-term memory entries"""
        try:
            with self.buffer_lock:
                buffer_copy = list(self.memory_buffer)
            
            file_entries = []
            if os.path.exists("memory/short_term.jsonl"):
                with open("memory/short_term.jsonl", "r", encoding="utf-8") as f:
                    file_entries = [json.loads(line) for line in f.readlines()[-max_entries:]]
            
            return file_entries + buffer_copy
        except Exception as e:
            logging.error(f"Memory read failed: {str(e)}")
            return []

    def _flush_buffer(self):
        if not self._active:  # Prevent post-shutdown writes
            return
            
        try:
            # Null check for logging
            if logging:
                with open("memory/short_term.jsonl", "a", encoding="utf-8") as f:
                    for entry in self.memory_buffer:
                        f.write(json.dumps(entry) + "\n")
                    self.memory_buffer.clear()
        except Exception as e:
            if logging:
                logging.error(f"Memory flush failed: {str(e)}")

class ERISEngine:
    def __init__(self, tokenizer, model):
        self._cuda_lock = threading.Lock()
        self.tokenizer = tokenizer
        self.model = model
        #self.summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
        self.memory = MemoryManager()
        self.fact_checker = FactChecker()
        self.web_search = WebSearch()
        self.system_prompt = (
            "ERIS can:\n"
            "- Perform web searches\n"
            "- Modify interface using [[command:type,selector,value]]\n"
            "Valid commands:\n"
            "[[command:color,.message,#color]]\n"
            "[[command:background,#chat-box,#color]]\n"
            "[[command:add_class,body,class-name]]\n"
            "Respond with commands at the end"
        )
        self.fact_check_enabled = False  # default
        self.search_enabled = False #default
        self.self_modifier = SelfModifier()

    def _build_prompt(self, user_input):
        # Get last n exchanges from short-term memory
        short_term_memory = self.memory.get_short_term()[-100:]  
    
        # Format the conversation history
        history = ""
        for entry in short_term_memory:
            if "user" in entry:
                history += f"User: {entry['user']}\n"
            if "response" in entry:
                history += f"ERIS: {entry['response']}\n"
    
        return (
            f"{self.system_prompt}\n"
            f"{history}\n"
            f"User: {user_input}\n"
            f"ERIS:"
        )

    def _extract_commands(self, text):
        try:
            commands = []
            command_pattern = r"\[\[command:(.*?)\]\]"
            for match in re.finditer(command_pattern, text):
                try:
                    parts = [p.strip() for p in match.group(1).split(",")]
                    if len(parts) < 2:
                        continue
                    command = {
                        "type": parts[0].lower(),
                        "target": parts[1],
                        "parameters": parts[2:] if len(parts) > 2 else []
                    }
                    if command["type"] in ["color", "background", "add_class", "toggle", "create_element"]:
                        commands.append(command)
                        text = text.replace(match.group(0), "")
                except Exception as e:
                    logging.warning(f"Invalid command format: {match.group(0)}")
                    continue
            return text.strip(), commands
        except Exception as e:
            logging.error(f"Command extraction failed: {str(e)}")
            return text, []

    def generate_response(self, user_input):
        # Timing debug
        start_time = time.time()
        with self._cuda_lock:
            try:
            # Optimized prompt building
                short_term_memory = self.memory.get_short_term()[-100:]
                history = "\n".join(
                    f"User: {entry['user']}\nERIS: {entry['response']}" 
                    for entry in short_term_memory
                    if "user" in entry and "response" in entry
                )
        
                prompt = f"{self.system_prompt}\n{history}\nUser: {user_input}\nERIS:"
        
                if self.search_enabled and any(word in user_input.lower() for word in ["current", "recent", "latest", "news"]):
                    print("Web Search Enabled.");
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        search_future = executor.submit(
                        self.web_search.perform_search,  
                        user_input,
                        engine="google",  # Default to Google
                        num_results=2
                    )
                    try:
                        search_results = search_future.result(timeout=3)
                        if search_results:
                            prompt += f"\n[Search Results: {json.dumps(search_results)}]"
                    except TimeoutError:
                        pass

                # Optimized model parameters
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=8192,
                    truncation=True,
                    padding=True
                ).to(DEVICE)

                # Faster generation config
                with torch.no_grad():
                        outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=150,  # Reduced from 175
                        temperature=0.82,      # More deterministic
                        top_p=0.82,
                        repetition_penalty=1.5,
                        do_sample=True,
                        num_return_sequences=1
                    )

                # Handle different output formats
                if hasattr(outputs, 'sequences'):
                    generated_sequence = outputs.sequences[0]
                elif hasattr(outputs, 'logits'):
                    generated_sequence = outputs.logits.argmax(-1)[0]
                else:
                    generated_sequence = outputs[0]

                response_text = self.tokenizer.decode(
                    generated_sequence,
                    skip_special_tokens=True
                ).split('ERIS:')[-1].strip()
                                        
                response_text, commands = self._extract_commands(response_text)
                if self.fact_check_enabled:
                    if any(x in user_input.lower() for x in ["is it true", "verify", "fact", "real", "accurate"]):
                        logging.debug("[FactCheck Triggered] Verifying response confidence...")
                        check_confidence, _ = self.fact_checker.verify_statement(response_text)
                        final_confidence = min(check_confidence + 0.3, 1.0)
                    else:
                        final_confidence = 1.0

                else:
                    print("Fact Checking Disabled.");
                    #final_confidence = 1.0  # assume full confidence if not fact-checked

                # Self-modification check must come after confidence calc
                if "[[selfmod:" in response_text:
                    return self._handle_selfmod_command(response_text, user_input)

                return response_text, round(final_confidence, 2), commands
        
            except RuntimeError as e:
                if "CUDA" in str(e):
                    logging.critical("CUDA corruption detected! Reinitializing...")
                    flush_gpu_memory()
                    self.model = self.model.to("cpu")
                    torch.cuda.empty_cache()
                    self.model = self.model.to(DEVICE)
                    return "System recovered, please try again", 0.5, []
            except Exception as e:
                logging.critical(f"Generation error: {str(e)}")
                traceback.print_exc()
                return "I encountered an error processing your request", 0.1, []
            finally:
                logging.info(f"Response generated in {time.time()-start_time:.2f}s")
                logging.info(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
                flush_gpu_memory()

    def _handle_selfmod_command(self, response_text: str, user_input: str):
        """Process self-modification commands"""
        try:
            # Extract command pattern: [[selfmod:propose,module_name]]
            command_pattern = r"\[\[selfmod:([a-z_]+),([\w\.]+)(?:,(.*))?\]\]"
            match = re.search(command_pattern, response_text)
            if not match:
                return "Invalid selfmod command format", 0.1, []
                
            cmd_type, module_name, extra = match.groups()
            clean_response = response_text.replace(match.group(0), "").strip()
            
            if cmd_type == "propose":
                # Generate patch using the model
                patch_prompt = f"Generate updated code for {module_name}. Original functionality must be preserved.\n\nUser request: {user_input}\n\nCode:"
                patch_code, _, _ = self.generate_response(patch_prompt)
                
                # Save proposed patch
                patch_path = self.self_modifier.propose_patch(module_name, patch_code)
                return f"Patch proposed at {patch_path}. Use [[selfmod:review,{patch_path}]] to review.", 0.9, []
                
            elif cmd_type == "review":
                patch_path = Path(extra)
                decision = self.self_modifier.review_patch(patch_path, self._ai_reviewer)
                if decision:
                    self.self_modifier.apply_patch(module_name, patch_path, auto_restart=False)
                    return "Patch applied successfully! Changes are active.", 1.0, []
                return "Patch rejected", 0.8, []
                
            return f"Unknown selfmod command: {cmd_type}", 0.1, []
            
        except Exception as e:
            return f"Selfmod failed: {str(e)}", 0.1, []
            
    def _ai_reviewer(self, code: str) -> bool:
        prompt = (
            "You are a code review assistant for an AI system named ERIS.\n"
            "Below is a Python patch proposal. You must review it for safety, functionality, and correctness.\n\n"
            "PATCH:\n"
            f"{code.strip()}\n\n"
            "Respond ONLY with either:\n"
            "APPROVE\n"
            "or\n"
            "REJECT\n"
            "(Do not explain or add commentary.)"
        )
        response, confidence, _ = self.generate_response(prompt)
        print(f"[REVIEW RESPONSE]: {response.strip()} (Confidence: {confidence})")
        return "APPROVE" in response.strip().upper() and confidence > 0.2

    def calculate_confidence(self, verification_results, response_text):
        # Base score from fact checking
        base_score = verification_results.get('total_score', 0.0)
    
        # Length penalty for long responses
        length_penalty = min(len(response_text) / 500, 1.0)  # 0-1 penalty
    
        # Uncertainty keywords detection
        uncertainty_terms = ["maybe", "perhaps", "I think", "possibly"]
        uncertainty_count = sum(response_text.lower().count(term) for term in uncertainty_terms)
        uncertainty_penalty = min(uncertainty_count * 0.1, 0.3)
    
        # Final calculation
        final_score = base_score * 0.7 + (1 - length_penalty) * 0.2 + (1 - uncertainty_penalty) * 0.1
        return min(max(final_score, 0.0), 1.0)

    def evaluate_code(self, code: str):
        try:
            import eris_plugins  # ← manually import here so it's available
            local_vars = {"eris_plugins": eris_plugins}
            exec(f"result = {code}", {}, local_vars)
            return str(local_vars["result"])
        except Exception as e:
            return f"Error evaluating code: {e}"

    def to(self, device):
        """Move the model to a specified device"""
        self.model = self.model.to(device)
        self.device = device
        return self

def initialize_engine():
    global eris_engine
    if eris_engine is not None:
        return
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logging.info(f"Loading model (attempt {attempt + 1}/{max_retries})...")
            tokenizer, model = load_model()
            eris_engine = ERISEngine(tokenizer, model)
            logging.info("Model loaded successfully")
            return
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {str(e)}")
            traceback.print_exc()
            if attempt == max_retries - 1:
                logging.critical("Max retries reached, exiting...")
                sys.exit(1)
            time.sleep(5 * (attempt + 1))
            flush_gpu_memory()
    
    # Ensure fallback if all retries fail
    eris_engine = None



# ======================
# UTILITY FUNCTIONS
# ======================
# 
def flush_gpu_memory():
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
            gc.collect()
            
             # Windows-specific cleanup (UPDATED)
            if sys.platform == "win32":
                # Use device-specific reset instead:
                subprocess.run(["nvidia-smi", "-h", "-i", "0"], check=False)
                
        except Exception as e:
            logging.error(f"GPU cleanup failed: {str(e)}")
    gc.collect()

def clean_response_text(text):
    return re.sub(r'\[.*?\]|\{.*?\}|<.*?>', '', text).strip()




# ======================
# FLASK ROUTES
# ======================

@app.route("/")
def serve_interface():
    return render_template('index.htm')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon') # Make/add favicon image!

# ======================
# TTS ROUTES (RE-ADDED FROM v64)
# ======================
@app.route('/google-tts-voices')
def google_tts_voices():
    try:
        response = requests.get(
            "https://texttospeech.googleapis.com/v1/voices",
            params={"key": GOOGLE_TTS_API_KEY},
            timeout=10
        )
        response.raise_for_status()
        voices = response.json().get('voices', [])
        filtered_voices = [
            v for v in voices 
            if any(lc.startswith('en') for lc in v.get('languageCodes', []))
        ]
        return jsonify([{
            "name": v["name"],
            "gender": v["ssmlGender"],
            "language": v["languageCodes"][0]
        } for v in filtered_voices])
    except Exception as e:
        logging.error(f"Google TTS voices error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/google-tts')
def google_tts():
    try:
        text = request.args.get('text', '')
        voice = request.args.get('voice', 'en-US-Wavenet-D')
        
        response = requests.post(
            GOOGLE_TTS_API_URL,
            json={
                "input": {"text": text},
                "voice": {"languageCode": "en-US", "name": voice},
                "audioConfig": {"audioEncoding": "MP3"}
            },
            params={"key": GOOGLE_TTS_API_KEY},
            timeout=15
        )
        response.raise_for_status()
        return jsonify(response.json())
    except Exception as e:
        logging.error(f"Google TTS error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/tts')
def tts_proxy():
    try:
        text = request.args.get('text', '')[:5000]  # Increased limit to 5000 chars
        voice = request.args.get('voice', 'en-US-Wavenet-D')
        
        # Validate voice
        if voice not in GOOGLE_TTS_VOICES:
            voice = 'en-US-Wavenet-D'

        # Split long text into chunks
        chunks = []
        current_chunk = []
        char_count = 0
        
        for sentence in re.split(r'(?<=[.!?]) +', text):
            if char_count + len(sentence) > 4500:  # Leave buffer for encoding
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                char_count = 0
            current_chunk.append(sentence)
            char_count += len(sentence)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        # Process chunks
        audio_responses = []
        for chunk in chunks:
            response = requests.post(
                "https://texttospeech.googleapis.com/v1/text:synthesize",
                json={
                    "input": {"text": chunk},
                    "voice": {"languageCode": "en-US", "name": voice},
                    "audioConfig": {"audioEncoding": "MP3"}
                },
                params={"key": GOOGLE_TTS_API_KEY},
                timeout=15
            )
            response.raise_for_status()
            audio_responses.append(response.json().get('audioContent', ''))
            
        return jsonify({"audioContents": audio_responses, "voiceUsed": voice})

    except Exception as e:
        logging.error(f"TTS Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ======================
# MORE FLASK ROUTES
# ======================

@app.route("/chat", methods=["POST"])
def handle_chat():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Invalid request"}), 400

        user_input = data["text"].strip()
        # Intercept !!eval for direct code execution
        if user_input.startswith("!!eval "):
            result = eris_engine.evaluate_code(user_input[7:])
            return jsonify({
                "response": result,
                "confidence": 1.0,
                "commands": []
            })
        if not user_input:
            return jsonify({"error": "Empty message"}), 400

        # Quick responses
        quick_responses = {
            "hello": "Hello there! How can I help you today?",
            "hi": "Hi! What can I do for you?",
            "ping": "Pong!",
            "test": "System is functioning normally"
        }
        
        if user_input.lower() in quick_responses:
            return jsonify({
                "response": quick_responses[user_input.lower()],
                "confidence": 1.0,
                "commands": []
            })

        # Main processing
        response, confidence, commands = eris_engine.generate_response(user_input)
        return jsonify({
            "response": response,
            "confidence": confidence,
            "commands": commands
        })

    except Exception as e:
        logging.error(f"Chat error: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health_check():
    status = {
        "status": "active" if eris_engine else "inactive",
        "model_loaded": eris_engine is not None,
        "gpu_available": torch.cuda.is_available(),
        "timestamp": datetime.now().isoformat()
    }
    return jsonify(status)

@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.errorhandler(404)
def not_found(e):
    return jsonify(error="Endpoint not found"), 404

# ======================
# FLASK ROUTES FOR SELF-MOD
# ======================
@app.route('/selfmod/propose', methods=['POST'])
def handle_selfmod_propose():
    try:
        data = request.get_json()
        module = data.get('module')
        code = data.get('code')
        if not module or not code:
            return jsonify({"error": "Missing module or code"}), 400
            
        patch_path = eris_engine.self_modifier.propose_patch(module, code)
        return jsonify({"path": str(patch_path)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/selfmod/review', methods=['POST'])
def handle_selfmod_review():
    try:
        data = request.get_json()
        path = data.get('path')
        module = data.get('module')

        if not path or not module:
            return jsonify({"error": "Missing path or module"}), 400

        patch_path = Path(path)  # <---- convert string to Path object!

        decision = eris_engine.self_modifier.review_patch(
            patch_path, 
            eris_engine._ai_reviewer
        )

        if decision:
            eris_engine.self_modifier.apply_patch(module, patch_path)
            return jsonify({
                "status": "applied",
                "message": "✅ Patch approved and applied.",
                "details": f"{patch_path.name} → {module}"
            })

        return jsonify({"status": "rejected"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ======================
# MAIN EXECUTION (AT BOTTOM)
# ======================
if __name__ == "__main__":
    try:
        initialize_engine()
        from waitress import serve
        serve(app, host='localhost', port=5011, threads=4)
    finally:
        if eris_engine:
            eris_engine.memory.close()
        flush_gpu_memory()
