"""
auth.py
=======
Authentication helpers for the AI Road Safety & Traffic Monitoring System.
"""

from functools import wraps

from flask import session, redirect, url_for

from werkzeug.security import generate_password_hash, check_password_hash

from src.analytics.statistics import (
    create_user,
    get_user_by_username,
    get_user_by_email,
)


# ---------------------------------------------------------------------------
# Password helpers
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    """Create a secure password hash."""
    return generate_password_hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its stored hash."""
    return check_password_hash(password_hash, password)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def authenticate_user(login_value: str, password: str):
    """
    Authenticate a user using either username or email.

    Returns
    -------
    dict | None
        User record when credentials are correct.
    """

    # Try username first
    user = get_user_by_username(login_value)

    # If username is not found, try email
    if user is None:
        user = get_user_by_email(login_value)

    if user is None:
        return None

    if not verify_password(password, user["password_hash"]):
        return None

    return user


def login_user(user: dict):
    """Store authenticated user information in the Flask session."""

    session.clear()

    session["user_id"] = user["id"]
    session["username"] = user["username"]
    session["role"] = user["role"]


def logout_user():
    """Clear the current user session."""
    session.clear()


# ---------------------------------------------------------------------------
# Route protection
# ---------------------------------------------------------------------------

def login_required(view):
    """Require an authenticated user before accessing a route."""

    @wraps(view)
    def wrapped_view(*args, **kwargs):

        if "user_id" not in session:
            return redirect(url_for("login"))

        return view(*args, **kwargs)

    return wrapped_view


def admin_required(view):
    """Require an authenticated administrator."""

    @wraps(view)
    def wrapped_view(*args, **kwargs):

        if "user_id" not in session:
            return redirect(url_for("login"))

        if session.get("role") != "admin":
            return redirect(url_for("index"))

        return view(*args, **kwargs)

    return wrapped_view
