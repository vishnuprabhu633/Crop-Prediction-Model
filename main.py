from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

flask_app= Flask(__name__)
model= pickle.load(open("model.pkl", "rb"))
@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods= ["POST"])
def predict():
    float_feature= [float(x) for x in request.form.values()]
    features= [np.array(float_feature)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text="The Predicted Crop is {}".format(prediction))

if __name__=="__main__":
    flask_app.run(debug= True)
