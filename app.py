import os
import pickle
import sqlite3
import uuid
import torch

from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_session import Session
from werkzeug.utils import secure_filename

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from rag import generate_rag_response


# =========================
# APP CONFIG
# =========================
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev_key")

DB_PATH = "nivada.db"

app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = False
Session(app)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


# =========================
# DEVICE SETUP (OPTIMIZED)
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# LOAD MODEL SAFELY
# =========================
MODEL_PATH = "fast_model/"
ENCODER_PATH = "label_encoder.pkl"

try:
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

    model.to(device)
    model.eval()

    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

except Exception as e:
    print("MODEL LOAD ERROR:", e)
    tokenizer = None
    model = None
    label_encoder = None


# =========================
# DB CONNECTION
# =========================
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            email TEXT UNIQUE,
            password TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            email TEXT UNIQUE,
            password TEXT,
            department TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            complaint_text TEXT,
            category TEXT,
            department TEXT,
            address TEXT,
            latitude TEXT,
            longitude TEXT,
            image_path TEXT,
            status TEXT DEFAULT 'pending',
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


init_db()


# =========================
# HELPERS
# =========================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_image(image):
    if not image or image.filename == "":
        return None

    if not allowed_file(image.filename):
        return None

    folder = "static/uploads"
    os.makedirs(folder, exist_ok=True)

    filename = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(folder, filename)

    image.save(path)
    return path


# =========================
# AUTH DECORATORS
# =========================
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "admin_id" not in session:
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return wrapper


# =========================
# ML PREDICTION (OPTIMIZED)
# =========================
def predict_complaint(text):
    if not model or not tokenizer:
        return "Other"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

        confidence, pred = torch.max(probs, dim=1)

    confidence = confidence.item()
    pred = pred.item()

    if confidence < 0.60:
        return "Other"

    return label_encoder.inverse_transform([pred])[0]


# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email").lower().strip()
        password = generate_password_hash(request.form.get("password"))

        conn = get_db()
        cur = conn.cursor()

        cur.execute("SELECT id FROM users WHERE email=?", (email,))
        if cur.fetchone():
            flash("Email already exists", "error")
            return redirect(url_for("register"))

        cur.execute(
            "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
            (username, email, password)
        )

        conn.commit()
        conn.close()

        flash("Registered successfully", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email").lower().strip()
        password = request.form.get("password")

        conn = get_db()
        cur = conn.cursor()

        cur.execute("SELECT * FROM users WHERE email=?", (email,))
        user = cur.fetchone()

        conn.close()

        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("submit_complaint"))

        flash("Invalid credentials", "error")

    return render_template("login.html")


@app.route("/submit", methods=["GET", "POST"])
@login_required
def submit_complaint():
    if request.method == "POST":

        text = request.form.get("complaint_text", "").strip()
        address = request.form.get("address")
        lat = request.form.get("latitude")
        lng = request.form.get("longitude")
        image = request.files.get("complaint_image")

        if not text and not image:
            flash("Enter complaint text or image", "error")
            return redirect(url_for("submit_complaint"))

        category = predict_complaint(text if text else "image complaint")

        rag = generate_rag_response(text if text else "image complaint")

        department = "municipal"

        session["rag_solution"] = rag["final_solution"]
        session["rag_cases"] = rag["similar_cases"]

        image_path = save_image(image)

        conn = get_db()
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO complaints
            (user_id, complaint_text, category, department, address, latitude, longitude, image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session["user_id"],
            text,
            category,
            department,
            address,
            lat,
            lng,
            image_path
        ))

        conn.commit()
        conn.close()

        flash(f"Complaint submitted: {category}", "success")
        return redirect(url_for("dashboard"))

    return render_template("index.html")


@app.route("/dashboard")
@login_required
def dashboard():
    conn = get_db()
    cur = conn.cursor()

    cur.execute(
        "SELECT * FROM complaints WHERE user_id=? ORDER BY timestamp DESC",
        (session["user_id"],)
    )
    cur.execute("SELECT COUNT(*) as total FROM complaints WHERE user_id=?", (session['user_id'],))
    total = cur.fetchone()["total"]

    cur.execute("SELECT COUNT(*) as pending FROM complaints WHERE user_id=? AND status='pending'", (session['user_id'],))
    pending = cur.fetchone()["pending"]

    cur.execute("SELECT COUNT(*) as progress FROM complaints WHERE user_id=? AND status='in progress'", (session['user_id'],))
    progress = cur.fetchone()["progress"]

    cur.execute("SELECT COUNT(*) as resolved FROM complaints WHERE user_id=? AND status='resolved'", (session['user_id'],))
    resolved = cur.fetchone()["resolved"]
    complaints = cur.fetchall()
    conn.close()

    return render_template(
        'dashboard.html',
        complaints=complaints,
        rag_solution=session.get("rag_solution"),
        rag_cases=session.get("rag_cases"),
        username=session.get("username"),
        user={"username": session.get("username")},
        total=total,
        pending=pending,
        progress=progress,
        resolved=resolved
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)