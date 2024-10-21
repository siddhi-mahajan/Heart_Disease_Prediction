from flask import Flask, render_template, request
import pickle

model = pickle.load(open("static/model/rf_model.pkl", 'rb'))
sc = pickle.load(open('static/model/scaler.pkl', 'rb'))

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/heart-disease-predictor", methods=['POST', 'GET'])
def heartDiseasePredictor():
    if request.method == 'POST':
        result = request.form.to_dict()
        age = int(result['age']) #age
        sex = int(result['gender']) #sex
        cp = int(result['chest-pain-type']) #cp
        trestbps = int(result['resting-blood-pressure']) #trestbps
        chol = int(result['serum-cholestrol-value']) #chol
        fbs = int(result['fasting-blood-sugar']) #fbs
        restecg = int(result['resting-ecg']) #restecg
        thalach = int(result['heart-rate-value'])  #thalach
        exang = int(result['induced-agina']) #exang
        old_peak = int(result['peak-exercise-st']) #old peak
        slop = float(result['st-depressed-value']) #slop
        try:
            data = sc.transform([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,old_peak,slop]])
        except Exception as e:
            pass
        prediction,prediction_prob = model.predict(data), model.predict_proba(data)
        if prediction_prob[0][0] > prediction_prob[0][1]:
            prediction_prob = prediction_prob[0][0] * 100
        else:
            prediction_prob = prediction_prob[0][1] * 100

        result['prediction'] = prediction[0]
        result['prediction-prob']= prediction_prob
        
        return render_template("result.html", results=result)
    return render_template("heart_disease.html")

if __name__ == "__main__":
    app.run(debug=True)
