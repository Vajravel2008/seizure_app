from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import mne
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model1 = tf.keras.models.load_model("seizure_model.h5")
model2 = tf.keras.models.load_model("seizure_chbmit_model.h5")

def process_edf(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    raw.filter(0.5, 40)
    data = raw.get_data()
    data = data[:, :2560]
    data = data.T
    data = np.expand_dims(data, axis=0)

    pred1 = model1.predict(data)[0][0]
    pred2 = model2.predict(data)[0][0]

    return pred1, pred2

@app.route("/", methods=["GET", "POST", "HEAD"])
def index():
    if request.method == "HEAD":
        return "", 200
    result = ""


    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", result="No file uploaded.")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", result="No file selected.")

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        pred1, pred2 = process_edf(file_path)

        if pred1 > 0.5 or pred2 > 0.5:
            result = f"⚠️ SEIZURE DETECTED (M1={pred1:.2f}, M2={pred2:.2f})"
        else:
            result = f"✅ NO SEIZURE (M1={pred1:.2f}, M2={pred2:.2f})"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

