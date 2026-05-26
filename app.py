import os
import sqlite3
import uuid
import pickle
import torch

from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_session import Session
from werkzeug.utils import secure_filename

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from rag import generate_rag_response


app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev_key")

DB_PATH = "nivada.db"

app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = False
Session(app)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

CATEGORY_TO_DEPT = {
    "Road": "municipal",
    "Pothole": "municipal",
    "Street Light": "electricity",
    "Power Outage": "electricity",
    "Electricity": "electricity",
    "Water Leakage": "water",
    "Water Supply": "water",
    "Drainage": "sanitation",
    "Garbage": "sanitation",
    "Sanitation": "sanitation",
    "Traffic": "transport",
    "Accident": "police",
    "Theft": "police",
    "Noise": "municipal",
    "Other": "municipal"
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            password TEXT,
            phone TEXT
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
            assigned_admin_id INTEGER,
            address TEXT,
            latitude TEXT,
            longitude TEXT,
            image_path TEXT,
            rag_solution TEXT,
            rag_cases TEXT,
            status TEXT DEFAULT 'pending',
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    cur.execute("PRAGMA table_info(complaints)")
    columns = [c[1] for c in cur.fetchall()]

    if "submitted_complaints" not in columns:
        cur.execute("ALTER TABLE complaints ADD COLUMN submitted_complaints TEXT")

    if "submitted_complaint_types" not in columns:
        cur.execute("ALTER TABLE complaints ADD COLUMN submitted_complaint_types TEXT")

    if "rag_solution" not in columns:
        cur.execute("ALTER TABLE complaints ADD COLUMN rag_solution TEXT")

    conn.commit()
    conn.close()

def normalize_category(cat: str):
    if not cat:
        return "Other"
def safe_category(category: str):
    if not category:
        return "Other"

    category = category.strip()

    return category if category in CATEGORY_TO_DEPT else "Other"
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_image(image):
    if not image or image.filename == "":
        return None

    if not allowed_file(image.filename):
        return None

    os.makedirs("static/uploads", exist_ok=True)

    ext = image.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"

    path = os.path.join("static", "uploads", filename)
    image.save(path)

    return path

def get_assigned_admin(department):
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        SELECT id FROM admins
        WHERE LOWER(department)=LOWER(?)
        LIMIT 1
    """, (department,))

    admin = cur.fetchone()
    conn.close()

    return admin["id"] if admin else None

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


@app.route("/")
def home():
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email").lower().strip()
        password = generate_password_hash(request.form.get("password"))
        phone = request.form.get("phone")

        conn = get_db()
        cur = conn.cursor()

        cur.execute("SELECT id FROM users WHERE email=?", (email,))
        if cur.fetchone():
            flash("Email already exists", "error")
            return redirect(url_for("register"))

        cur.execute("""
            INSERT INTO users (username, email, password, phone)
            VALUES (?, ?, ?, ?)
        """, (username, email, password, phone))

        conn.commit()
        conn.close()

        flash("Registered successfully", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].lower().strip()
        password = request.form["password"]

        conn = get_db()
        cur = conn.cursor()

        cur.execute("SELECT * FROM users WHERE email=?", (email,))
        user = cur.fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("dashboard"))

        flash("Invalid credentials", "error")

    return render_template("login.html")


@app.route("/profile")
@login_required
def profile():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT id, username, email, phone FROM users WHERE id=?", (session["user_id"],))
    user = cur.fetchone()

    conn.close()
    return render_template("profile.html", user=user)


@app.route("/update_profile", methods=["POST"])
@login_required
def update_profile():
    username = request.form.get("username")
    email = request.form.get("email")
    phone = request.form.get("phone")

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        UPDATE users
        SET username=?, email=?, phone=?
        WHERE id=?
    """, (username, email, phone, session["user_id"]))

    conn.commit()
    conn.close()

    session["username"] = username

    flash("Profile updated successfully", "success")
    return redirect(url_for("profile"))


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        email = request.form["email"].lower().strip()
        password = request.form["password"]

        conn = get_db()
        cur = conn.cursor()

        cur.execute("SELECT * FROM admins WHERE email=?", (email,))
        admin = cur.fetchone()
        conn.close()

        if admin and check_password_hash(admin["password"], password):
            session.clear()
            session["admin_id"] = admin["id"]
            session["admin_name"] = admin["username"]
            session["department"] = admin["department"]
            return redirect(url_for("admin_dashboard"))

        flash("Invalid admin credentials", "error")

    return render_template("admin_login.html")

@app.route("/admin/register", methods=["GET", "POST"])
def admin_register():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email").lower().strip()
        password = generate_password_hash(request.form.get("password"))
        department = request.form.get("department")

        conn = get_db()
        cur = conn.cursor()

        cur.execute("SELECT id FROM admins WHERE email=?", (email,))
        if cur.fetchone():
            flash("Admin already exists", "error")
            return redirect(url_for("admin_register"))

        cur.execute("""
            INSERT INTO admins (username, email, password, department)
            VALUES (?, ?, ?, ?)
        """, (username, email, password, department))

        conn.commit()
        conn.close()

        flash("Admin registered successfully", "success")

        # ✅ FORCE CORRECT ROUTE
        return redirect(url_for("admin_login"))

    return render_template("admin_register.html")
@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        SELECT * FROM complaints
        WHERE LOWER(department)=LOWER(?)
        ORDER BY timestamp DESC
    """, (session["department"],))

    complaints = cur.fetchall()
    conn.close()

    return render_template(
        "admin_dashboard.html",
        complaints=complaints,
        admin_name=session.get("admin_name")
    )


@app.route("/submit", methods=["GET", "POST"])
@login_required
def submit_complaint():
    if request.method == "POST":

        text = request.form.get("complaint_text", "").strip()
        address = request.form.get("address")
        latitude = request.form.get("latitude")
        longitude = request.form.get("longitude")
        image = request.files.get("complaint_image")

        if not text and not image:
            flash("Enter complaint text or image", "error")
            return redirect(url_for("submit_complaint"))

        query = text if text else "image complaint"

        category = predict_complaint(query)
        category = safe_category(category)

        department = CATEGORY_TO_DEPT.get(category, "municipal")

        assigned_admin_id = get_assigned_admin(department)

        rag = generate_rag_response(query)

        image_path = save_image(image)

        conn = get_db()
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO complaints (
                user_id, complaint_text, category, department,
                assigned_admin_id, address, latitude, longitude,
                image_path, rag_solution, rag_cases
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session["user_id"],
            text,
            category,
            department,
            assigned_admin_id,
            address,
            latitude,
            longitude,
            image_path,
            rag.get("final_solution", ""),
            str(rag.get("similar_cases", []))
        ))

        conn.commit()
        conn.close()

        return redirect(url_for("dashboard"))

    return render_template("index.html")


@app.route("/dashboard")
@login_required
def dashboard():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        SELECT * FROM complaints
        WHERE user_id=?
        ORDER BY timestamp DESC
    """, (session["user_id"],))

    complaints = cur.fetchall()

    cur.execute("""
        SELECT COUNT(*) FROM complaints
        WHERE user_id=?
    """, (session["user_id"],))

    total = cur.fetchone()[0]

    conn.close()

    return render_template(
        "dashboard.html",
        complaints=complaints,
        total=total,
        username=session.get("username")
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

if __name__ == "__main__":
    init_db() 
    app.run(debug=True, host="0.0.0.0", port=5000)