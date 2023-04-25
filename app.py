from flask import Flask,render_template,request,jsonify
import pandas as pd
import numpy as np
from src.pipline.prediction_pipline import PredictPipline,CustomData

application = Flask(__name__)
app = application

@app.route("/",methods = ["GET","POST"])
def prediction_datapoint():
    if request.method == "GET":
        return render_template("form.html")

    else:
        data = CustomData(
            Cement = float(request.form.get("Cement")),
            Blast_Furnace_Slag = float(request.form.get("Blast_Furnace_Slag")),
            Superplasticizer = float(request.form.get("Superplasticizer")),
            Coarse_Aggregate = float(request.form.get("Coarse_Aggregate")),
            Age = float(request.form.get("Age"))
            )

        filan_data = data.get_data_as_data_frame()
        predict_piplinr = PredictPipline()
        pred = predict_piplinr.predict(filan_data)
        result = round(pred[0],2)

        return render_template("form.html",final_result = "Your Concrete Strength is: {}".format(result))


if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)
