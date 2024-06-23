from flask import Flask, render_template, request, jsonify, session, redirect, flash, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail
from datetime import datetime
from flask import flash
import json
import os.path
import re
import os
# import pandas as pd
import os
import sys
import shutil
import time
# import joblib
# from sklearn.ensemble import RandomForestClassifier
import urllib.request
# from geopy.geocoders import Nominatim
from flask import Flask, jsonify, request
import numpy as np

# from flask_cors import CORS
from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import tensorflow as tf

with open('config.json', 'r') as c:
    params = json.load(c)["params"]


local_server = True
app = Flask(__name__, template_folder='templates')
app.secret_key = 'super-secret-key'
acc_num_global = {}


app.config['SQLALCHEMY_DATABASE_URI'] = params['prod_uri']
db = SQLAlchemy(app)


@app.route("/")
def home():
    return render_template('index.html', params=params)


class Contact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(50), nullable=False)
    subject = db.Column(db.String(50), nullable=False)
    message = db.Column(db.String(250), nullable=False)


class Register(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    uname = db.Column(db.String(50), nullable=False)
    mobile = db.Column(db.BIGINT, nullable=False)
    email = db.Column(db.String(50), nullable=False)
    password = db.Column(db.String(10), nullable=False)
    cpassword = db.Column(db.String(10), nullable=False)


@app.route("/about")
def about():
    return render_template('about.html', params=params)


@app.route("/logout", methods=['GET', 'POST'])
def logout():
    session.pop('email')
    return redirect(url_for('home', params=params))


@app.route("/contact",  methods=['GET', 'POST'])
def contact():
    if (request.method == 'POST'):
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')
        entry = Contact(name=name, email=email,
                        subject=subject, message=message)
        db.session.add(entry)
        db.session.commit()
    return render_template('contact.html', params=params)


@app.route("/register",  methods=['GET', 'POST'])
def register():
    if (request.method == 'POST'):
        name = request.form.get('name')
        uname = request.form.get('uname')
        mobile = request.form.get('mobile')
        email = request.form.get('email')
        password = request.form.get('password')
        cpassword = request.form.get('cpassword')

        user = Register.query.filter_by(email=email).first()
        if user:
            flash('Account already exist!Please login', 'success')
            return redirect(url_for('register'))
        if not (len(name)) > 3:
            flash('length of name is invalid', 'error')
            return redirect(url_for('register'))
        if (len(mobile)) != 10:
            flash('invalid mobile number', 'error')
            return redirect(url_for('register'))
        if (len(password)) < 8:
            flash('length of password should be greater than 7', 'error')
            return redirect(url_for('register'))
        else:
            flash('You have registtered succesfully', 'success')

        entry = Register(name=name, uname=uname, mobile=mobile,
                         email=email, password=password, cpassword=cpassword)
        db.session.add(entry)
        db.session.commit()
    return render_template('register.html', params=params)


@app.route("/login", methods=['GET', 'POST'])
def login():
    if (request.method == "GET"):
        if ('email' in session and session['email']):
            return render_template('base.html', params=params)
        else:
            return render_template("login.html", params=params)

    if (request.method == "POST"):
        email = request.form["email"]
        password = request.form["password"]

        login = Register.query.filter_by(
            email=email, password=password).first()
        if login is not None:
            session['email'] = email
            return render_template('base.html', params=params)
        else:
            flash("plz enter right password", 'error')
            return render_template('login.html', params=params)


UPLOAD_FOLDER = 'static/uploads/'
DENOISED_FOLDER = 'static/denoised/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DENOISED_FOLDER'] = DENOISED_FOLDER

# Load the trained model
model = tf.keras.models.load_model(
    "color_image_denoising_model.keras", compile=False)


def preprocess_image(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    return img


def postprocess_image(img):
    img = np.clip(img, 0, 1) * 255
    img = img.astype('uint8')
    return img


@app.route('/base', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(
                app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            input_image = preprocess_image(file_path)
            denoised_image = model.predict(
                np.expand_dims(input_image, axis=0))[0]
            denoised_image = postprocess_image(denoised_image)
            denoised_file_path = os.path.join(
                app.config['DENOISED_FOLDER'], file.filename)
            cv2.imwrite(denoised_file_path, cv2.cvtColor(
                denoised_image, cv2.COLOR_RGB2BGR))
            return render_template('base.html', original_image=file_path, denoised_image=denoised_file_path)
    return render_template('base.html', original_image=None, denoised_image=None)


if __name__ == "__main__":
    app.run(debug=True, port=8080)
