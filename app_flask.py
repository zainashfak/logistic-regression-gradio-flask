from flask import Flask, render_template, request, abort
from config import DATASETS
from common import load_bundle, predict

app = Flask(__name__)
BUNDLES = {k: load_bundle(k) for k in DATASETS}

@app.route("/")
def index():
    return render_template("index.html", datasets=BUNDLES)

@app.route("/<key>", methods=["GET", "POST"])
def form(key):
    if key not in BUNDLES:
        abort(404)
    b = BUNDLES[key]
    result = probs = error = None
    values = {}
    if request.method == "POST":
        values = {c: request.form.get(c, "") for c in b["features"]}
        try:
            result, probs = predict(b, values)
        except Exception as e:
            error = f"Invalid input: {e}"
    return render_template("predict.html", key=key, b=b, values=values,
                           result=result, probs=probs, error=error)

if __name__ == "__main__":
    app.run(debug=True)
