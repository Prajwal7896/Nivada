import os
import pickle
import sqlite3
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_session import Session
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from rag import generate_rag_response

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev_key")
DB_PATH = "nivada.db"
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
Session(app)

MODEL_PATH = "fast_model/"
ENCODER_PATH = "label_encoder.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

except Exception:
    model = None
    tokenizer = None

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

        confidence = torch.max(probs).item()
        prediction_idx = torch.argmax(probs, dim=1).item()

    if confidence < 0.6:
        return "Other"

    return label_encoder.inverse_transform([prediction_idx])[0]


CATEGORY_TO_DEPT = {
    "Theft": "police",
    "Robbery": "police",
    "Chain Snatching": "police",
    "Cyber Fraud": "police",
    "Online Scam": "police",
    "Identity Theft": "police",
    "Assault": "police",
    "Domestic Violence": "police",
    "Harassment": "police",
    "Child Abuse": "police",
    "Drug Trafficking": "police",
    "Illegal Weapons": "police",
    "Kidnapping": "police",
    "Missing Person": "police",
    "Police Misconduct": "police",
    "Power Outage": "electricity",
    "Low Voltage": "electricity",
    "Frequent Power Cuts": "electricity",
    "Transformer Failure": "electricity",
    "Street Light Not Working": "electricity",
    "Electric Shock Hazard": "electricity",
    "Meter Fault": "electricity",
    "High Electricity Bill": "electricity",
    "Fuse Issue": "electricity",
    "Substation Failure": "electricity",
    "Water Leakage": "water",
    "Drinking Water Shortage": "water",
    "Contaminated Water": "water",
    "Pipeline Burst": "water",
    "Low Water Pressure": "water",
    "Drain Overflow": "water",
    "Sewer Blockage": "water",
    "Water Tank Issue": "water",
    "Illegal Water Connection": "water",
    "Water Supply Delay": "water",
    "Potholes": "transport",
    "Road Damage": "transport",
    "Traffic Jam": "transport",
    "Signal Failure": "transport",
    "Bus Delay": "transport",
    "Auto Overcharging": "transport",
    "Accident Hazard": "transport",
    "Parking Issue": "transport",
    "Road Construction Delay": "transport",
    "Public Transport Misconduct": "transport",
    "Hospital Negligence": "health",
    "Ambulance Delay": "health",
    "Medicine Shortage": "health",
    "Doctor Absence": "health",
    "Overcharging Hospital": "health",
    "Poor Hygiene Hospital": "health",
    "Vaccination Issue": "health",
    "Emergency Delay": "health",
    "Blood Bank Issue": "health",
    "Medical Staff Misconduct": "health",
    "Garbage Collection Issue": "municipal",
    "Street Cleaning": "municipal",
    "Public Toilet Issue": "municipal",
    "Stray Animals": "municipal",
    "Illegal Construction": "municipal",
    "Water Logging": "municipal",
    "Street Light Maintenance": "municipal",
    "Park Maintenance": "municipal",
    "Dead Animal Removal": "municipal",
    "Air Pollution": "environment",
    "Water Pollution": "environment",
    "Noise Pollution": "environment",
    "Tree Cutting": "environment",
    "Illegal Dumping": "environment",
    "Industrial Waste": "environment",
    "Deforestation": "environment",
    "River Contamination": "environment",
    "Plastic Ban Violation": "environment",
    "Climate Hazard": "environment",
    "Internet Down": "telecom",
    "Slow Internet": "telecom",
    "Mobile Network Issue": "telecom",
    "Call Drop": "telecom",
    "SIM Activation Issue": "telecom",
    "Billing Issue": "telecom",
    "Recharge Failure": "telecom",
    "Fiber Cut": "telecom",
    "Tower Fault": "telecom",
    "Spam Calls": "telecom",

}

def get_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row 
        return conn
    except Exception as e:
        print("DB CONNECTION ERROR:", e)
        return None


def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            email TEXT UNIQUE,
            password TEXT
        )
    ''')

    cur.execute('''
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT UNIQUE,
            password TEXT,
            department TEXT
        )
    ''')

    cur.execute('''
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
    ''')

    conn.commit()
    cur.close()
    conn.close()

init_db()

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrapper

def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'admin_id' not in session:
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return wrapper

from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

import uuid

def save_image(image):
    if not image or image.filename == "":
        return None

    if not allowed_file(image.filename):
        return None

    upload_folder = "static/uploads"
    os.makedirs(upload_folder, exist_ok=True)

    ext = image.filename.rsplit('.', 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"

    file_path = os.path.join(upload_folder, filename)
    image.save(file_path)

    return file_path

@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('submit_complaint'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password_raw = request.form.get('password')

        if not username or not email or not password_raw:
            flash("All fields required", "error")
            return redirect(url_for('register'))
        email = email.lower().strip()
        password = generate_password_hash(password_raw)

        try:
            conn = get_db()
            cur = conn.cursor()

            cur.execute("SELECT * FROM users WHERE email=?", (email,))
            existing = cur.fetchone()

            if existing:
                flash("Email already exists", "error")
                cur.close()
                conn.close()
                return redirect(url_for('register'))

            cur.execute(
                "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                (username, email, password)
            )

            conn.commit()
            cur.close()
            conn.close()

            flash("Registered successfully", "success")
            return redirect(url_for('login'))

        except Exception as e:
            print(e)
            flash("Error occurred", "error")

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if not email or not password:
            flash("Enter email and password", "error")
            return redirect(url_for('login'))

        email = email.lower().strip()  # ✅ normalize email

        try:
            conn = get_db()
            cur = conn.cursor()

            cur.execute(
                "SELECT * FROM users WHERE email=?",
                (email,)
            )
            user = cur.fetchone()

            if user and check_password_hash(user['password'], password):
                session['user_id'] = user['id']
                session['username'] = user['username']

                cur.close()
                conn.close()

                return redirect(url_for('submit_complaint'))

            cur.close()
            conn.close()

            flash("Invalid login", "error")

        except Exception as e:
            print(e)
            flash("Server error. Try again.", "error")

    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    try:
        conn = get_db()
        cur = conn.cursor()

        cur.execute(
            "SELECT * FROM complaints WHERE user_id=? ORDER BY timestamp DESC",
            (session['user_id'],)
        )
        complaints = cur.fetchall()

        cur.close()
        conn.close()

    except Exception as e:
        print(e)
        complaints = []

    rag_solution = session.pop("rag_solution", None)
    rag_cases = session.pop("rag_cases", None)

    return render_template(
        'dashboard.html',
        complaints=complaints,
        rag_solution=rag_solution,
        rag_cases=rag_cases,
        username=session.get("username"),  
        user={"username": session.get("username")} 
    )

@app.route('/submit', methods=['GET', 'POST'])
@login_required
def submit_complaint():
    if request.method == 'POST':
        text = request.form.get('complaint_text', '').strip()
        address = request.form.get('address')
        lat = request.form.get('latitude')
        lng = request.form.get('longitude')
        image = request.files.get('complaint_image')

        if not text and not image:
            flash("Please provide complaint text or image", "error")
            return redirect(url_for('submit_complaint'))

        category = predict_complaint(text if text else "image complaint")
        rag_data = generate_rag_response(text if text else "image complaint")

        solution = rag_data["final_solution"]
        similar_cases = rag_data["similar_cases"]

        department = CATEGORY_TO_DEPT.get(category, "municipal")

        session["rag_solution"] = solution
        session["rag_cases"] = similar_cases

        image_path = save_image(image)

        try:
            conn = get_db()
            cur = conn.cursor()

            cur.execute(
                """
                INSERT INTO complaints 
                (user_id, complaint_text, category, department, address, latitude, longitude, image_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session['user_id'],
                    text,
                    category,
                    department,
                    address,
                    lat,
                    lng,
                    image_path
                )
            )

            conn.commit()
            cur.close()
            conn.close()

        except Exception as e:
            print("DB ERROR:", e)
            flash("Error occurred while submitting complaint", "error")
            return redirect(url_for('submit_complaint'))

        flash(f"Submitted successfully! Category: {category}", "success")
        return redirect(url_for('dashboard'))

    return render_template('index.html')


@app.route('/admin_register', methods=['GET', 'POST'])
def admin_register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password_raw = request.form.get('password')
        department = request.form.get('department')

        if not username or not email or not password_raw or not department:
            flash("All fields are required", "error")
            return redirect(url_for('admin_register'))

        password = generate_password_hash(password_raw)
        email = email.lower().strip() 

        try:
            conn = get_db()
            cur = conn.cursor()

            # Check existing admin
            cur.execute("SELECT * FROM admins WHERE email=?", (email,))
            existing = cur.fetchone()

            if existing:
                flash("Email already exists", "error")
                cur.close()
                conn.close()
                return redirect(url_for('admin_register'))

            # Insert admin
            cur.execute(
                "INSERT INTO admins (username, email, password, department) VALUES (?, ?, ?, ?)",
                (username, email, password, department)
            )

            conn.commit()
            cur.close()
            conn.close()

            flash("Admin registered successfully", "success")
            return redirect(url_for('admin_login'))

        except Exception as e:
            print(e)
            flash("Something went wrong", "error")

    return render_template('admin_register.html')

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        department = request.form.get('department')

        if not email or not password or not department:
            flash("All fields required", "error")
            return redirect(url_for('admin_login'))

        email = email.lower().strip() 

        try:
            conn = get_db()
            cur = conn.cursor()

            cur.execute(
                "SELECT * FROM admins WHERE email=? AND department=?",
                (email, department)
            )
            admin = cur.fetchone()

            if admin and check_password_hash(admin['password'], password):
                session['admin_id'] = admin['id']
                session['department'] = admin['department']
                session['admin_name'] = admin['username']

                cur.close()
                conn.close()

                return redirect(url_for('admin_dashboard'))

            cur.close()
            conn.close()

            flash("Invalid login credentials", "error")

        except Exception as e:
            print(e)
            flash("Server error. Try again.", "error")

    return render_template('admin_login.html')

@app.route('/admin_dashboard')
@admin_required
def admin_dashboard():
    dept = session['department']

    try:
        conn = get_db()
        cur = conn.cursor()

        
        cur.execute(
            "SELECT * FROM complaints WHERE department=? ORDER BY timestamp DESC",
            (dept,)
        )
        complaints = cur.fetchall()

        cur.execute("""
            SELECT category, COUNT(*) as count
            FROM complaints
            WHERE department=?
            GROUP BY category
        """, (dept,))
        stats = cur.fetchall()

        cur.close()
        conn.close()

    except Exception as e:
        print(e)
        complaints = []
        stats = []
        flash("Error loading dashboard", "error")

    return render_template(
        'admin_dashboard.html',
        complaints=complaints,
        stats=stats
    )

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(debug=True)