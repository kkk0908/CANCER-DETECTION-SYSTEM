from flask import Flask, render_template, request, url_for
import os

PEOPLE_FOLDER = os.path.join('static', 'image')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

@app.route('/')
@app.route('/index')
def show_index():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'img2.jpg')
    return render_template("image_upload.html", user_image = full_filename)