# Logistic Regression + Gradio + Flask

1. `pip install -r requirements.txt`
2. Put your 2 CSV files in `data/` and edit `config.py` (csv path, target column, title).
3. `python train.py`        -> trains both models, prints accuracy / report / confusion matrix
4. `python app_gradio.py`   -> Gradio UI at http://127.0.0.1:7860
5. `python app_flask.py`    -> Flask UI at http://127.0.0.1:5000
