from flask import Flask, render_template, request, redirect, url_for, flash, session, abort
from flask_sqlalchemy import SQLAlchemy
import bcrypt, joblib
from flask_migrate import Migrate
import numpy as np
import os
import secrets
import sqlite3
import time
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from datetime import timedelta, datetime
from sqlalchemy.exc import IntegrityError
from werkzeug.utils import secure_filename


MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
try:
    with open(MODEL_PATH, "rb") as file:
        model = joblib.load(file)
except Exception as exc:
    raise RuntimeError(
        "Failed to load model.joblib. Install compatible dependencies from requirements.txt "
        "(especially scikit-learn==1.3.2) and retry."
    ) from exc

app = Flask(__name__)
db_path = os.path.join(app.instance_path, 'database.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{db_path}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'secret_key')
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = os.path.join(app.instance_path, 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MAX_LOGIN_ATTEMPTS = 5
LOGIN_WINDOW_SECONDS = 15 * 60
LOCKOUT_SECONDS = 15 * 60
LOAN_CATEGORIES = [
    'Newlywed Grants',
    'Student Loans',
    'Personal Loan',
    'Real Estate',
    'Unemployment Grants',
    'Mortgage',
    'Assets Accumulation',
]
DEFAULT_ADMIN_EMAILS = {
    "admin1@loanhub.local",
    "admin2@loanhub.local",
}


def _admin_email_from_env(env_key):
    return (os.environ.get(env_key, "") or "").strip().lower()


ADMIN_ALLOWED_EMAILS = set(DEFAULT_ADMIN_EMAILS)
for _env_key in ("ADMIN_EMAIL_1", "ADMIN_EMAIL_2"):
    _email = _admin_email_from_env(_env_key)
    if _email:
        ADMIN_ALLOWED_EMAILS.add(_email)
DOCUMENT_TYPES = [
    "Identity Proof",
    "Address Proof",
    "Income Proof",
    "Employment Proof",
    "Property Document",
]
ALLOWED_DOC_EXTENSIONS = {"pdf", "jpg", "jpeg", "png"}
FAQ_ITEMS = [
    {
        "question": "Why was my loan application not approved?",
        "answer": "Approval depends on multiple factors such as credit score, income, loan amount, tenure, and profile consistency.",
    },
    {
        "question": "What does confidence mean in the decision?",
        "answer": "Confidence shows how certain the model is about the predicted outcome. Higher confidence means stronger model certainty.",
    },
    {
        "question": "Can I apply again after a rejection?",
        "answer": "Yes. You can improve key factors such as income-to-loan ratio, credit profile, and tenure choice, then re-apply.",
    },
    {
        "question": "Which documents are required for validation?",
        "answer": "Identity proof, address proof, income proof, employment proof, and property documents (if applicable).",
    },
    {
        "question": "Can admin users see all applications?",
        "answer": "Yes. Admin users can access the admin panel and review all submitted applications.",
    },
]

db = SQLAlchemy(app)
migrate = Migrate(app, db)


login_manager = LoginManager(app)
login_manager.login_view = 'login'  # Specify the login view endpoint


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def generate_csrf_token():
    token = session.get('_csrf_token')
    if not token:
        token = secrets.token_hex(16)
        session['_csrf_token'] = token
    return token


app.jinja_env.globals['csrf_token'] = generate_csrf_token


@app.before_request
def protect_post_requests():
    if request.method == 'POST':
        token = request.form.get('_csrf_token') or request.headers.get('X-CSRFToken')
        if not token or token != session.get('_csrf_token'):
            abort(400, description='Invalid CSRF token')


def clean_recent_attempts(attempts, now_epoch):
    return [ts for ts in attempts if now_epoch - ts < LOGIN_WINDOW_SECONDS]


def validate_choice(value, valid_options):
    return value in valid_options


def allowed_document_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_DOC_EXTENSIONS


def is_admin_account(user):
    return bool(
        user
        and getattr(user, "is_authenticated", False)
        and (user.email or "").strip().lower() in ADMIN_ALLOWED_EMAILS
    )


@app.context_processor
def inject_asset_version():
    style_path = os.path.join(app.static_folder, 'style.css')
    try:
        style_version = int(os.path.getmtime(style_path))
    except OSError:
        style_version = 1
    return {
        'style_version': style_version,
        'is_admin_account': is_admin_account,
    }


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    role = db.Column(db.String(), default='User')
    authenticated = db.Column(db.Boolean, default=False)
    def __init__(self, email, password, name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))


def sync_admin_access():
    """Enforce that only whitelisted emails have Admin role."""
    try:
        users = User.query.all()
        changed = False
        for user in users:
            email = (user.email or "").strip().lower()
            should_be_admin = email in ADMIN_ALLOWED_EMAILS
            target_role = "Admin" if should_be_admin else "User"
            if user.role != target_role:
                user.role = target_role
                changed = True
        if changed:
            db.session.commit()
    except Exception:
        db.session.rollback()

@app.route('/')
def home():
    labels = LOAN_CATEGORIES

    data = [0, 10, 15, 8, 22, 18, 25]

    # Log the length of data array to the console for debugging
    print(f"Data array length: {len(data)}")

    # Return the components to the HTML template
    return render_template(
        template_name_or_list='dashboard.html',
        data=data,
        labels=labels,
    )

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        if not name or not email or not password:
            flash('Please fill all required fields.', 'error')
            return redirect(url_for('register'))
        if len(password) < 8:
            flash('Password must be at least 8 characters.', 'error')
            return redirect(url_for('register'))
        if email in ADMIN_ALLOWED_EMAILS:
            flash('This email is reserved for system administrators.', 'error')
            return redirect(url_for('register'))

        new_user = User(name=name, email=email, password=password)
        new_user.role = 'User'
        db.session.add(new_user)
        try:
            db.session.commit()
            flash('Registration successful. Please login.', 'success')
            return redirect(url_for('login'))
        except IntegrityError:
            db.session.rollback()
            flash('This email is already registered.', 'error')
            return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        sync_admin_access()
        now_epoch = int(time.time())
        lockout_until = session.get('lockout_until', 0)
        if lockout_until and now_epoch < lockout_until:
            remaining = lockout_until - now_epoch
            wait_minutes = max(1, remaining // 60)
            flash(f'Too many failed attempts. Try again in {wait_minutes} minute(s).', 'error')
            return redirect(url_for('login'))

        attempts = clean_recent_attempts(session.get('login_attempts', []), now_epoch)
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            # Keep role aligned with admin whitelist at login time.
            expected_role = 'Admin' if (user.email or '').strip().lower() in ADMIN_ALLOWED_EMAILS else 'User'
            if user.role != expected_role:
                user.role = expected_role
            session['login_attempts'] = []
            session['lockout_until'] = 0
            user.authenticated = True
            db.session.commit()

            login_user(user)
            flash('Login successful!', 'success')

            next_page = request.args.get('next')
            return redirect(next_page or url_for('userprofile'))
        else:
            attempts.append(now_epoch)
            session['login_attempts'] = attempts
            if len(attempts) >= MAX_LOGIN_ATTEMPTS:
                session['lockout_until'] = now_epoch + LOCKOUT_SECONDS
                flash('Too many failed attempts. Account temporarily locked.', 'error')
            else:
                remaining = MAX_LOGIN_ATTEMPTS - len(attempts)
                flash(f'Incorrect password or invalid email. {remaining} attempt(s) left.', 'error')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    current_user.authenticated = False
    db.session.commit()

    logout_user()
    flash('Logout successful!', 'success')

    return redirect(url_for('login'))

@app.route('/userprofile')
@login_required
def userprofile():
    applications = LoanApplication.query.filter_by(loan_applicant_userid=current_user.id).all()
    decision_explanation = normalize_explanation_text(session.get('last_decision_explanation'))
    session['last_decision_explanation'] = decision_explanation
    return render_template(
        'user-profile.html',
        user=current_user,
        applications=applications,
        decision_explanation=decision_explanation,
    )


@app.route('/admin_profile')
@login_required
def admin_profile():
    sync_admin_access()
    if not is_admin_account(current_user):
        flash('Admin access required.', 'error')
        return redirect(url_for('userprofile'))
    applications = LoanApplication.query.all()
    return render_template('index.html', user=current_user, applications=applications)


@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    decision_explanation = normalize_explanation_text(session.get('last_decision_explanation'))
    session['last_decision_explanation'] = decision_explanation
    return render_template(
        "prediction.html",
        decision_explanation=decision_explanation,
        loan_categories=LOAN_CATEGORIES,
    )


@app.route('/analysis')
@login_required
def analysis():
    if is_admin_account(current_user):
        applications = LoanApplication.query.all()
    else:
        applications = LoanApplication.query.filter_by(loan_applicant_userid=current_user.id).all()

    applications = sorted(applications, key=lambda x: x.id)

    app_labels = [f"App #{app.id}" for app in applications]
    loan_amounts = [float(app.loan_amount or 0) for app in applications]
    terms = [float(app.loan_term_length or 0) for app in applications]
    credit_scores = [float(app.credit_score or 0) for app in applications]

    category_counts = {}
    location_counts = {"Urban": 0, "Semiurban": 0, "Rural": 0, "Unknown": 0}
    uncategorized_count = 0
    for app in applications:
        category = (app.Loan_Categories or "").strip()
        if not category:
            uncategorized_count += 1
        else:
            category_counts[category] = category_counts.get(category, 0) + 1
        location = app.property_location or "Unknown"
        if location not in location_counts:
            location_counts["Unknown"] += 1
        else:
            location_counts[location] += 1

    summary = {
        "total_applications": len(applications),
        "avg_loan_amount": round(sum(loan_amounts) / len(loan_amounts), 2) if loan_amounts else 0,
        "avg_term": round(sum(terms) / len(terms), 2) if terms else 0,
        "avg_credit": round(sum(credit_scores) / len(credit_scores), 4) if credit_scores else 0,
    }

    chart_data = {
        "app_labels": app_labels,
        "loan_amounts": loan_amounts,
        "terms": terms,
        "credit_scores": credit_scores,
        "category_labels": list(category_counts.keys()),
        "category_values": list(category_counts.values()),
        "location_labels": [k for k, v in location_counts.items() if v > 0],
        "location_values": [v for v in location_counts.values() if v > 0],
        "uncategorized_count": uncategorized_count,
    }

    return render_template("analysis.html", summary=summary, chart_data=chart_data)


@app.route('/about')
@login_required
def about():
    return render_template("about.html")


@app.route('/faq')
@login_required
def faq():
    query = request.args.get('q', '').strip().lower()
    if query:
        faq_items = [
            item for item in FAQ_ITEMS
            if query in item["question"].lower() or query in item["answer"].lower()
        ]
    else:
        faq_items = FAQ_ITEMS
    return render_template("faq.html", faq_items=faq_items, query=query)


@app.route('/documents', methods=['GET', 'POST'])
@login_required
def documents():
    if request.method == 'POST':
        document_type = request.form.get('document_type', '').strip()
        file = request.files.get('document_file')

        if not validate_choice(document_type, set(DOCUMENT_TYPES)):
            flash('Please select a valid document type.', 'error')
            return redirect(url_for('documents'))
        if not file or not file.filename:
            flash('Please choose a file to upload.', 'error')
            return redirect(url_for('documents'))

        original_name = file.filename.strip()
        validation_errors = []

        if not allowed_document_file(original_name):
            validation_errors.append('Invalid file format. Allowed: PDF, JPG, JPEG, PNG.')

        file.stream.seek(0, os.SEEK_END)
        file_size = file.stream.tell()
        file.stream.seek(0)

        if file_size <= 0:
            validation_errors.append('File appears to be empty.')
        if file_size > app.config['MAX_CONTENT_LENGTH']:
            validation_errors.append('File is too large. Maximum size is 5 MB.')

        extension = original_name.rsplit(".", 1)[1].lower() if "." in original_name else ""
        mime = (file.mimetype or "").lower()
        mime_expect = {
            "pdf": {"application/pdf"},
            "jpg": {"image/jpeg"},
            "jpeg": {"image/jpeg"},
            "png": {"image/png"},
        }
        if extension in mime_expect and mime and mime not in mime_expect[extension]:
            validation_errors.append('File content type does not match file extension.')

        is_valid = len(validation_errors) == 0
        safe_name = secure_filename(original_name)
        stored_name = None
        message = "Document is valid and uploaded successfully."

        if is_valid:
            stored_name = f"{current_user.id}_{int(time.time())}_{safe_name}"
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], stored_name)
            file.save(save_path)
        else:
            message = " ".join(validation_errors)

        upload_record = DocumentUpload(
            user_id=current_user.id,
            document_type=document_type,
            original_filename=original_name,
            stored_filename=stored_name,
            file_size=file_size,
            is_valid=is_valid,
            validation_message=message,
        )
        db.session.add(upload_record)
        db.session.commit()

        flash(message, 'success' if is_valid else 'error')
        return redirect(url_for('documents'))

    uploads = DocumentUpload.query.filter_by(user_id=current_user.id).order_by(DocumentUpload.id.desc()).all()
    valid_uploaded_types = {
        doc.document_type
        for doc in uploads
        if doc.is_valid and doc.document_type in DOCUMENT_TYPES
    }
    completed = len(valid_uploaded_types)
    total_required = len(DOCUMENT_TYPES)
    completion_percent = int((completed / total_required) * 100) if total_required else 0

    return render_template(
        "documents.html",
        uploads=uploads,
        document_types=DOCUMENT_TYPES,
        valid_uploaded_types=valid_uploaded_types,
        completed=completed,
        total_required=total_required,
        completion_percent=completion_percent,
    )


@app.route('/improve-score')
@login_required
def improve_score():
    return render_template("improve_score.html")


def build_decision_explanation(approved, confidence, credit, applicant_income, coapplicant_income, loan_amount, term_months, education, employed, area):
    total_income = max(0.0, applicant_income + coapplicant_income)
    income_to_loan = (total_income / loan_amount) if loan_amount > 0 else 0.0

    strengths = []
    risks = []

    if credit >= 0.84:
        strengths.append("Credit history is strong (high credit score).")
    else:
        risks.append("Credit score is in the low or medium range, which increases default risk.")

    if total_income >= 5000:
        strengths.append("Total household income is healthy.")
    else:
        risks.append("Total household income is relatively low.")

    if income_to_loan >= 0.15:
        strengths.append("Loan amount is manageable compared to total income.")
    else:
        risks.append("Loan amount is high compared to total income.")

    if term_months >= 180:
        strengths.append("Longer tenure can reduce monthly repayment pressure.")
    else:
        risks.append("Short tenure can increase monthly repayment pressure.")

    if employed == "Yes":
        strengths.append("Applicant is employed, which is a positive stability signal.")
    else:
        risks.append("Employment status indicates lower income stability.")

    if education == "Graduate":
        strengths.append("Graduate profile is generally a positive signal in the model.")

    if area == "Urban":
        strengths.append("Urban property location aligns with favorable training patterns.")

    if confidence >= 0.75:
        confidence_band = "High confidence"
    elif confidence >= 0.55:
        confidence_band = "Medium confidence"
    else:
        confidence_band = "Low confidence"

    return {
        "decision": "Approved" if approved else "Not Approved",
        "confidence_pct": f"{confidence:.2%}",
        "confidence_band": confidence_band,
        "strengths": strengths[:4],
        "risks": risks[:4],
    }


def normalize_explanation_text(explanation):
    if not explanation:
        return explanation

    replacements = {
        "Credit history strong hai (high credit score).": "Credit history is strong (high credit score).",
        "Total household income achha hai.": "Total household income is healthy.",
        "Income ke comparison me loan amount manageable hai.": "Loan amount is manageable compared to total income.",
        "Loan amount income ke comparison me high lag raha hai.": "Loan amount is high compared to total income.",
        "Short tenure monthly repayment pressure badha sakti hai.": "Short tenure can increase monthly repayment pressure.",
        "Applicant employed hai, income stability positive signal hai.": "Applicant is employed, which is a positive stability signal.",
        "Graduate profile model me generally positive signal deta hai.": "Graduate profile is generally a positive signal in the model.",
        "Urban property profile training pattern me acceptable hai.": "Urban property location aligns with favorable training patterns.",
    }

    for key in ("strengths", "risks"):
        items = explanation.get(key, [])
        explanation[key] = [replacements.get(item, item) for item in items]
    return explanation


class LoanApplication(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50), nullable=False)
    middle_initial = db.Column(db.String(10))
    last_name = db.Column(db.String(50), nullable=False)
    gender= db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(50))
    contact_number = db.Column(db.String(50), nullable=False)
    home_address= db.Column(db.String(50), nullable=False)
    education = db.Column(db.String(50), nullable=False)
    Employment_status = db.Column(db.String(50), nullable=False)
    Employer_name= db.Column(db.String(50), nullable=False)
    Salary= db.Column(db.Integer(), nullable=False)
    credit_score= db.Column(db.Numeric(3, 4),nullable=False)
    marital_status = db.Column(db.String(50), nullable=False)
    spouse_name = db.Column(db.String(50))
    spouse_employment_status= db.Column(db.String(50))
    spouse_salary = db.Column(db.Integer())
    spouse_employer = db.Column(db.String(50))
    dependents = db.Column(db.String(50), nullable=False)
    property_location = db.Column(db.String(50), nullable=False)
    loan_amount = db.Column(db.Integer(), nullable=False)
    loan_term_length= db.Column(db.Integer(), nullable=False)
    declaration= db.Column(db.String(50))
    Loan_Categories = db.Column(db.String(100))
    loan_applicant_userid = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('loan_applications', lazy=True))


class DocumentUpload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    document_type = db.Column(db.String(50), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    stored_filename = db.Column(db.String(255))
    file_size = db.Column(db.Integer, default=0)
    is_valid = db.Column(db.Boolean, default=False)
    validation_message = db.Column(db.String(255), default="")
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)


def ensure_loan_categories_column():
    """Ensure legacy SQLite DB has Loan_Categories column."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='loan_application'")
        if not cur.fetchone():
            return
        cur.execute("PRAGMA table_info(loan_application)")
        columns = [row[1] for row in cur.fetchall()]
        if "Loan_Categories" not in columns:
            cur.execute("ALTER TABLE loan_application ADD COLUMN Loan_Categories VARCHAR(100)")
            conn.commit()
    finally:
        conn.close()


def ensure_document_upload_table():
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS document_upload (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                document_type VARCHAR(50) NOT NULL,
                original_filename VARCHAR(255) NOT NULL,
                stored_filename VARCHAR(255),
                file_size INTEGER DEFAULT 0,
                is_valid BOOLEAN DEFAULT 0,
                validation_message VARCHAR(255) DEFAULT '',
                uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES user(id)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


with app.app_context():
    ensure_loan_categories_column()
    ensure_document_upload_table()
    sync_admin_access()
    
@app.route('/loan_application', methods=['POST'])
@login_required
def loan_application():
    if request.method == 'POST':
        try:
            first_name = request.form.get('first_name', '').strip()
            middle_initial = request.form.get('middle_initial', '').strip()
            last_name = request.form.get('last_name', '').strip()
            gender = request.form.get('gender', '').strip()
            email = request.form.get('email', '').strip()
            contact_number = request.form.get('telephone', '').strip()
            home_address = request.form.get('home_address', '').strip()
            education = request.form.get('education', '').strip()
            salary = float(request.form.get('ApplicantIncome', 0))
            employment_status = request.form.get('employed', '').strip()
            employer_name = request.form.get('company_name', '').strip()
            credit_score = float(request.form.get('credit', 0))
            marital_status = request.form.get('married', '').strip()
            spouse_name = request.form.get('spouse', '').strip()
            spouse_employment_status = request.form.get('spouse_employed', '').strip()
            spouse_salary = float(request.form.get('CoapplicantIncome', 0))
            spouse_employer = request.form.get('spouse_company_name', '').strip()
            dependents = request.form.get('dependents', '').strip()
            property_location = request.form.get('area', '').strip()
            loan_amount = float(request.form.get('LoanAmount', 0))
            loan_term_length = float(request.form.get('Loan_Amount_Term', 0))
            declaration = request.form.get('declaration', '').strip()
            loan_category = request.form.get('loan_category', '').strip()
        except ValueError:
            flash('Invalid numeric values in the form.', 'error')
            return redirect(url_for('predict'))

        if not first_name or not last_name:
            flash('First and last name are required.', 'error')
            return redirect(url_for('predict'))
        if not validate_choice(gender, {'Male', 'Female'}):
            flash('Please select a valid gender.', 'error')
            return redirect(url_for('predict'))
        if not validate_choice(marital_status, {'Yes', 'No'}):
            flash('Please select a valid marital status.', 'error')
            return redirect(url_for('predict'))
        if not validate_choice(dependents, {'0', '1', '2', '3+'}):
            flash('Please select valid dependents.', 'error')
            return redirect(url_for('predict'))
        if not validate_choice(education, {'Graduate', 'Not Graduate'}):
            flash('Please select valid education.', 'error')
            return redirect(url_for('predict'))
        if not validate_choice(employment_status, {'Yes', 'No'}):
            flash('Please select a valid employment status.', 'error')
            return redirect(url_for('predict'))
        if not validate_choice(property_location, {'Urban', 'Semiurban', 'Rural'}):
            flash('Please select a valid property location.', 'error')
            return redirect(url_for('predict'))
        if not declaration:
            flash('Declaration is required.', 'error')
            return redirect(url_for('predict'))
        if not validate_choice(loan_category, set(LOAN_CATEGORIES)):
            flash('Please select a valid loan category.', 'error')
            return redirect(url_for('predict'))

        approved, confidence = predict_loan_approval(
            gender, marital_status, dependents, education, employment_status,
            credit_score, property_location, salary, spouse_salary, loan_amount,
            loan_term_length
        )
        session['last_decision_explanation'] = build_decision_explanation(
            approved=approved,
            confidence=confidence,
            credit=credit_score,
            applicant_income=salary,
            coapplicant_income=spouse_salary,
            loan_amount=loan_amount,
            term_months=loan_term_length,
            education=education,
            employed=employment_status,
            area=property_location,
        )

        if approved:
            new_application = LoanApplication(
                first_name=first_name, middle_initial=middle_initial, last_name=last_name,
                gender=gender, email=email, contact_number=contact_number,
                home_address=home_address, education=education, Salary=salary,
                Employment_status=employment_status, Employer_name=employer_name,
                credit_score=credit_score, marital_status=marital_status,
                spouse_name=spouse_name, spouse_employment_status=spouse_employment_status,
                spouse_salary=spouse_salary, spouse_employer=spouse_employer,
                dependents=dependents, property_location=property_location,
                loan_amount=loan_amount, loan_term_length=loan_term_length,
                declaration=declaration,
                Loan_Categories=loan_category,
                loan_applicant_userid=current_user.id
            )

            db.session.add(new_application)
            db.session.commit()
            flash(f'Loan application approved and saved! Confidence: {confidence:.2%}', 'success')
            return redirect(url_for('userprofile'))

        flash(f'Loan application not approved. Confidence: {confidence:.2%}', 'info')
        return redirect(url_for('predict'))

    flash('Invalid request method.', 'error')
    return redirect(url_for('userprofile'))

def predict_loan_approval(gender, married, dependents, education, employed, credit, area, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term):
    # Guard against invalid log inputs.
    if ApplicantIncome <= 0 or LoanAmount <= 0 or Loan_Amount_Term <= 0 or (ApplicantIncome + CoapplicantIncome) <= 0:
        return False, 0.0

    male = 1 if gender == "Male" else 0
    married_yes = 1 if married == "Yes" else 0

    if dependents == '1':
        dependents_1, dependents_2, dependents_3 = 1, 0, 0
    elif dependents == '2':
        dependents_1, dependents_2, dependents_3 = 0, 1, 0
    elif dependents == "3+":
        dependents_1, dependents_2, dependents_3 = 0, 0, 1
    else:
        dependents_1, dependents_2, dependents_3 = 0, 0, 0

    not_graduate = 1 if education == "Not Graduate" else 0
    employed_yes = 1 if employed == "Yes" else 0

    if area == "Semiurban":
        semiurban, urban = 1, 0
    elif area == "Urban":
        semiurban, urban = 0, 1
    else:
        semiurban, urban = 0, 0

    ApplicantIncomelog = np.log(ApplicantIncome)
    totalincomelog = np.log(ApplicantIncome + CoapplicantIncome)
    LoanAmountlog = np.log(LoanAmount)
    Loan_Amount_Termlog = np.log(Loan_Amount_Term)

    prediction = model.predict([[credit, ApplicantIncomelog, LoanAmountlog,
                                 Loan_Amount_Termlog, totalincomelog, male,
                                 married_yes, dependents_1, dependents_2,
                                 dependents_3, not_graduate, employed_yes,
                                 semiurban, urban]])

    approved = str(prediction[0]).strip().upper() == "Y"
    confidence = 0.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([[credit, ApplicantIncomelog, LoanAmountlog,
                                              Loan_Amount_Termlog, totalincomelog, male,
                                              married_yes, dependents_1, dependents_2,
                                              dependents_3, not_graduate, employed_yes,
                                              semiurban, urban]])[0]
        confidence = float(np.max(probabilities))
    return approved, confidence


@app.route('/view_applications', methods=['GET'])
@login_required
def view_applications():
    if is_admin_account(current_user):
        user_applications = LoanApplication.query.all()
        return render_template('index.html', user=current_user, applications=user_applications)
    else:
        user_applications = LoanApplication.query.filter_by(loan_applicant_userid=current_user.id).all()
        return render_template('user-profile.html', user=current_user, applications=user_applications)


@app.errorhandler(400)
def bad_request(error):
    flash(getattr(error, 'description', 'Bad request.'), 'error')
    return render_template('dashboard.html', data=[0, 0, 0, 0], labels=['Bad', 'Request', 'Check', 'Input']), 400


@app.errorhandler(404)
def not_found(_error):
    return render_template('dashboard.html', data=[0, 0, 0, 0], labels=['Not', 'Found', 'Page', '404']), 404


@app.errorhandler(500)
def server_error(_error):
    return render_template('dashboard.html', data=[0, 0, 0, 0], labels=['Server', 'Error', 'Try', 'Again']), 500


@app.errorhandler(413)
def payload_too_large(_error):
    flash('File is too large. Maximum upload size is 5 MB.', 'error')
    return redirect(url_for('documents'))

 
if __name__ == "__main__":
    app.run(debug=True)





