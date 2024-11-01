from flask import Blueprint, render_template, redirect, url_for, request, flash
from . import db
from .models import User
from flask_login import login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

auth = Blueprint('auth', __name__)

@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get("email")
        password = request.form.get("password")

        #checking if the user exists
        user = User.query.filter_by(email=email).first()
        if user:
            #checking if the password is correct
            if check_password_hash(user.password, password):
                flash("Logged in successfully!", category="success")
                #logging in session
                login_user(user, remember=True)
                return redirect(url_for('views.home'))
            else:
                flash("Incorrect password", category="error")
        else:
            flash("Email does not exist", category="error")

    

    return render_template('login.html')

@auth.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get("email")
        username = request.form.get("username")
        password1 = request.form.get("password1")
        password2 = request.form.get("password2")

        #in case entry already exists
        email_exists = User.query.filter_by(email=email).first()
        username_exists = User.query.filter_by(username=username).first()

        if email_exists:
            flash("Email already exists", category="error")
        elif username_exists:
            flash("Username already in use", category="error")
        elif password1 != password2:
            flash("Passwords don't match", category="error")
        elif len(username) < 2:
            flash("Username too short", category="error")
        elif len(password1) < 7:
            flash("Password must be at least 7 characters", category="error")
        else:
            new_user = User(email=email, username=username, password=generate_password_hash(password1, method="pbkdf2:sha256"))
            db.session.add(new_user)
            db.session.commit()
            #logging in session
            login_user(new_user, remember=True)
            flash("Account created!", category="success")
            return redirect(url_for("views.home"))


    return render_template('signup.html')

@auth.route('/logout')
@login_required
def logout():
    #logging out session
    logout_user()
    return redirect(url_for("views.home"))

