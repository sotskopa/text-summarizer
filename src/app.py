import utils
import flask
from flask_cors import CORS

app = flask.Flask(__name__)
CORS(app)

tokenizer, model = utils.load_model("models/finetuned")


@app.route("/summarize", methods=["POST"])
def summary():
    try:
        data = flask.request.json
        text = str(data["text"])
        summary = utils.generate_summary(model, tokenizer, text)
        if isinstance(summary, list):
            summary = summary[0]
        return flask.jsonify({"summary": summary, "status": 200})
    except Exception as e:
        return flask.jsonify({"error": str(e), "status": 500})



if __name__ == "__main__":
    app.run(debug=True, port=5000)
