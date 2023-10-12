from flask import Flask, render_template, request
from Fake_Url_detect import *
import numpy as np

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/gocontact")
def go_to():
    return render_template("contact.html")
@app.route("/ser")
def service():
    return render_template("services.html")
@app.route("/gd")
def guard():
    return render_template("guard.html")
@app.route("/abt")
def about():
    return render_template("about.html")

@app.route("/result", methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        url = request.form["url"]
        obj = FeatureExtraction(url)
        s = np.array(obj.getFeaturesList()).reshape(1,30)
        y_pred = gbc.predict(s)[0]
        return render_template("contact.html", url=url, y_pred=y_pred)

if '__name__' == '__main__':
    app.run(debug=False, host='0.0.0.0')
