# Loan Prediction System (Flask)

A Flask web application for **loan eligibility pre-screening** with user authentication, role-based dashboards, and persisted loan applications.

## Project Details

This project combines a machine learning model with a traditional Flask app workflow:

- Users can register, log in, and submit a loan application form.
- The backend converts submitted form values into model-ready features.
- A trained model (`model.joblib`) predicts whether the application is preliminarily eligible.
- If predicted as eligible, the application is saved to SQLite for review in admin views.

The app demonstrates how to integrate:

- **Authentication & sessions**: Flask-Login + bcrypt password hashing.
- **Persistence**: Flask-SQLAlchemy models + Alembic/Flask-Migrate migrations.
- **ML inference in request flow**: model loading with Joblib and prediction during form submission.

## Main Features

- User registration and login/logout.
- Role-aware app experience (user/admin-oriented pages).
- Loan application form with personal, employment, and loan attributes.
- Prediction-based approval gate before database persistence.
- Admin-facing application listing.

## Tech Stack

- Python
- Flask
- Flask-Login
- Flask-SQLAlchemy
- Flask-Migrate
- bcrypt
- NumPy
- Joblib
- SQLite

## Repository Structure

- `app.py` — main Flask app, routes, SQLAlchemy models, and prediction logic.
- `templates/` — HTML templates for auth, dashboards, profile, and prediction form.
- `static/` — CSS/JS and image assets.
- `migrations/` — Alembic migration setup and versions.
- `model.joblib` — serialized ML model used for loan prediction.
- `train.csv`, `test.csv` — dataset artifacts.

## Getting Started

### 1) Clone and enter the project

```bash
git clone <your-repo-url>
cd loan-prediction-system
```

### 2) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate   # Windows (PowerShell)
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Initialize database (if needed)

```bash
flask db upgrade
```

### 5) Run the application

```bash
python app.py
```

Open: `http://127.0.0.1:5000`

## Typical User Flow

1. Register an account.
2. Log in.
3. Open the prediction/application page.
4. Submit loan details.
5. Receive approval/not-approved feedback.
6. Approved applications are stored and visible in admin listings.

## Notes

- `app.py` currently contains both routing and model logic in a single file. As a next step, you can split this into blueprints/services for better maintainability.
- The Flask secret key is hardcoded in the sample code and should be replaced with an environment variable for production.
- SQLite is suitable for local development; consider PostgreSQL/MySQL for production deployments.
