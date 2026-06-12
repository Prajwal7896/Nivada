import os
import uuid
import pickle
import torch

from fastapi import FastAPI, Request, Form, UploadFile, File, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from monitoring_service import track_prediction
from starlette.middleware.sessions import SessionMiddleware

from sqlalchemy import create_engine, Column, Integer, String, Text, TIMESTAMP
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.sql import func

from werkzeug.security import generate_password_hash, check_password_hash

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from rag import generate_rag_response


# =========================
# APP SETUP
# =========================
app = FastAPI()

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY", "dev_key")
)

templates = Jinja2Templates(directory="templates")


# =========================
# DATABASE
# =========================
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:sage7896@127.0.0.1:5432/nivada"
)

engine = create_engine(
    DATABASE_URL,
    echo=True,
    pool_pre_ping=True
)
SessionLocal = sessionmaker(bind=engine, autoflush=False)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String)
    email = Column(String, unique=True)
    password = Column(String)
    phone = Column(String)


class Admin(Base):
    __tablename__ = "admins"
    id = Column(Integer, primary_key=True)
    username = Column(String)
    email = Column(String, unique=True)
    password = Column(String)
    department = Column(String)


class Complaint(Base):
    __tablename__ = "complaints"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)

    complaint_text = Column(Text)
    category = Column(String)
    department = Column(String)

    assigned_admin_id = Column(Integer)

    address = Column(Text)
    latitude = Column(String)
    longitude = Column(String)

    image_path = Column(Text)

    rag_solution = Column(Text)
    rag_cases = Column(Text)

    status = Column(String, default="pending")
    timestamp = Column(TIMESTAMP, server_default=func.now())


Base.metadata.create_all(bind=engine)


# =========================
# DB SESSION
# =========================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =========================
# CATEGORY MAP
# =========================
DEPT_MAPPING = {
    # Electricity group
    "Electricity": "electricity",
    "Power Outage": "electricity",
    "Network": "electricity",

    # Water group
    "Water": "water",
    "Water Supply": "water",
    "Water Leakage": "water",

    # Municipal / Infrastructure
    "Road": "municipal",
    "Infrastructure": "municipal",
    "Sanitation": "municipal",

    # Transport
    "Transport": "transport",

    # Police
    "Crime": "police",
    "Theft": "police",

    # Health
    "Health": "health",

    # Environment
    "Environment": "environment",
}

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# ML MODEL
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
# HELPERS
# =========================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_image(image):
    if not image or not image.filename:
        return None

    if not allowed_file(image.filename):
        return None

    os.makedirs("static/uploads", exist_ok=True)

    ext = image.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join("static/uploads", filename)

    with open(path, "wb") as f:
        f.write(image.file.read())

    return path


def predict_complaint(text):
    if not model or not tokenizer:
        return "Other"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    if confidence.item() < 0.60:
        return "Other"

    return label_encoder.inverse_transform([pred.item()])[0]


def get_assigned_admin(department, db):
    admin = db.query(Admin).filter(Admin.department.ilike(department)).first()
    return admin.id if admin else None


# =========================
# ROUTES
# =========================

@app.get("/")
def home():
    return RedirectResponse("/login")


# -------------------------
# REGISTER
# -------------------------
@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@app.post("/register")
def register(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    phone: str = Form(...),
    db: Session = Depends(get_db)
):
    email = email.lower().strip()

    if db.query(User).filter(User.email == email).first():
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error": "Email already exists"
        })

    user = User(
        username=username,
        email=email,
        password=generate_password_hash(password),
        phone=phone
    )

    db.add(user)
    db.commit()

    return templates.TemplateResponse("login.html", {
        "request": request,
        "success": "Account created successfully. Please login."
    })


# -------------------------
# LOGIN
# -------------------------

@app.post("/login")
def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.email == email.lower().strip()).first()

    if user and check_password_hash(user.password, password):
        request.session["user_id"] = user.id
        request.session["username"] = user.username
        return RedirectResponse("/dashboard", status_code=302)

    return RedirectResponse("/login?registered=1", status_code=302)
@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

# -------------------------
# PROFILE
# -------------------------
@app.get("/profile", response_class=HTMLResponse)
def profile(request: Request, db: Session = Depends(get_db)):
    user_id = request.session.get("user_id")

    if not user_id:
        return RedirectResponse("/login")

    user = db.query(User).filter(User.id == user_id).first()

    return templates.TemplateResponse("profile.html", {
        "request": request,
        "user": user
    })


@app.post("/update_profile")
def update_profile(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    phone: str = Form(...),
    db: Session = Depends(get_db)
):
    user_id = request.session.get("user_id")

    if not user_id:
        return RedirectResponse("/login", status_code=302)

    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        return RedirectResponse("/login", status_code=302)

    # update
    user.username = username
    user.email = email.lower().strip()
    user.phone = phone

    db.commit()

    request.session["username"] = username

    return RedirectResponse("/profile", status_code=302)


# -------------------------
# ADMIN
# -------------------------
@app.get("/admin/login", response_class=HTMLResponse)
def admin_login_page(request: Request):
    return templates.TemplateResponse("admin_login.html", {"request": request})


@app.post("/admin/login")
def admin_login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    admin = db.query(Admin).filter(Admin.email == email.lower().strip()).first()

    if admin and check_password_hash(admin.password, password):
        request.session.clear()
        request.session["admin_id"] = admin.id
        request.session["admin_name"] = admin.username
        request.session["department"] = admin.department
        return RedirectResponse("/admin/dashboard", status_code=302)

    return templates.TemplateResponse("admin_login.html", {
        "request": request,
        "error": "Invalid credentials"
    })


@app.post("/admin/register")
def admin_register(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    department: str = Form(...),
    db: Session = Depends(get_db)
):
    if db.query(Admin).filter(Admin.email == email).first():
        return templates.TemplateResponse("admin_login.html", {
            "request": request,
            "error": "Admin already exists"
        })

    admin = Admin(
        username=username,
        email=email,
        password=generate_password_hash(password),
        department=department
    )

    db.add(admin)
    db.commit()

    return RedirectResponse("/admin/login", status_code=302)

@app.get("/admin/register", response_class=HTMLResponse)
def admin_register_page(request: Request):
    return templates.TemplateResponse("admin_register.html", {
        "request": request
    })

@app.get("/admin/dashboard", response_class=HTMLResponse)
def admin_dashboard(request: Request, db: Session = Depends(get_db)):

    admin_id = request.session.get("admin_id")

    if not admin_id:
        return RedirectResponse("/admin/login")

    complaints = db.query(Complaint).filter(
        Complaint.assigned_admin_id == admin_id
    ).order_by(Complaint.timestamp.desc()).all()

    total = len(complaints)

    pending = db.query(Complaint).filter(
        Complaint.assigned_admin_id == admin_id,
        Complaint.status == "pending"
    ).count()

    resolved = db.query(Complaint).filter(
        Complaint.assigned_admin_id == admin_id,
        Complaint.status == "resolved"
    ).count()

    in_progress = db.query(Complaint).filter(
        Complaint.assigned_admin_id == admin_id,
        Complaint.status == "in progress"
    ).count()

    return templates.TemplateResponse("admin_dashboard.html", {
        "request": request,
        "complaints": complaints,
        "admin_name": request.session.get("admin_name"),
        "total": total,
        "pending": pending,
        "resolved": resolved,
        "in_progress": in_progress
    })


# -------------------------
# COMPLAINT
# -------------------------
@app.get("/submit", response_class=HTMLResponse)
def submit_page(request: Request):
    user_id = request.session.get("user_id")

    if not user_id:
        return RedirectResponse("/login", status_code=302)

    return templates.TemplateResponse("index.html", {
        "request": request
    })
@app.post("/submit")
def submit_complaint(
    request: Request,
    complaint_text: str = Form(""),
    address: str = Form(""),
    latitude: str = Form(""),
    longitude: str = Form(""),
    complaint_image: UploadFile = File(None),
    db: Session = Depends(get_db),
):
    user_id = request.session.get("user_id")

    if not user_id:
        return RedirectResponse("/login")

    if not complaint_text and not complaint_image:
        return RedirectResponse("/dashboard")

    query = complaint_text if complaint_text else "image complaint"

    category = predict_complaint(query)
    category = category.strip().title()

    department = DEPT_MAPPING.get(category, "municipal")
    assigned_admin_id = get_assigned_admin(department, db)

    rag = generate_rag_response(query)

    track_prediction(
        confidence=rag.get("confidence", 0.5),
        extra={
            "category": category,
            "department": department,
            "text_length": len(query)
        }
    )

    image_path = save_image(complaint_image)

    complaint = Complaint(
        user_id=user_id,
        complaint_text=complaint_text,
        category=category,
        department=department,
        assigned_admin_id=assigned_admin_id,
        address=address,
        latitude=latitude,
        longitude=longitude,
        image_path=image_path,
        rag_solution=rag.get("final_solution", ""),
        rag_cases=str(rag.get("similar_cases", []))
    )

    db.add(complaint)
    db.commit()

    return RedirectResponse("/dashboard", status_code=302)

import json

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)):
    user_id = request.session.get("user_id")

    if not user_id:
        return RedirectResponse("/login")

    complaints = db.query(Complaint).filter(
        Complaint.user_id == user_id
    ).order_by(Complaint.timestamp.desc()).all()

    total = len(complaints)

    pending = db.query(Complaint).filter(
        Complaint.user_id == user_id,
        Complaint.status == "pending"
    ).count()

    progress = db.query(Complaint).filter(
        Complaint.user_id == user_id,
        Complaint.status == "in progress"
    ).count()

    resolved = db.query(Complaint).filter(
        Complaint.user_id == user_id,
        Complaint.status == "resolved"
    ).count()

    # OPTIONAL: attach latest RAG (safe parse)
    rag_solution = None
    rag_cases = []

    if complaints:
        last = complaints[0]

        rag_solution = last.rag_solution

        try:
            rag_cases = json.loads(last.rag_cases) if last.rag_cases else []
        except:
            rag_cases = []

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "complaints": complaints,
        "total": total,
        "pending": pending,
        "progress": progress,
        "resolved": resolved,
        "rag_solution": rag_solution,
        "rag_cases": rag_cases,
        "username": request.session.get("username")
    })


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login")