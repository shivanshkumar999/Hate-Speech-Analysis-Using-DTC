import pickle as pkl
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

from flask import Flask, render_template, flash, request
# import requests
app = Flask(__name__)
app.secret_key = "HateSpeechDetector"

data = pd.read_csv("HS_Data_New.csv")
x = data['tweet']
cv = CountVectorizer()
cv.fit_transform(x)
model = pkl.load(open("HateSpeechDetector.pkl",'rb'))

# inp = "hey you look are so fat fucked cow"
# print(model.predict(inp))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST','GET'])
def predict():
    if request.method =='POST':
        inp = request.form['inp_data']
        inp_data = inp
        inp = cv.transform([inp]).toarray()
        flash("".join(model.predict(inp)))
    return render_template("index.html", inp_data = inp_data)

if __name__ == "__main__":
    app.run(debug=True)

