import gradio as gr
from config import DATASETS
from common import load_bundle, predict

def make_interface(key):
    b = load_bundle(key)
    inputs = []
    for c in b["features"]:
        if c in b["numeric"]:
            inputs.append(gr.Number(label=c, value=b["numeric"][c]["median"]))
        else:
            opts = b["categorical"][c]
            inputs.append(gr.Dropdown(opts, label=c, value=opts[0]))

    def fn(*vals):
        label, probs = predict(b, dict(zip(b["features"], vals)))
        return label, probs

    return gr.Interface(
        fn, inputs,
        [gr.Textbox(label="Prediction"), gr.Label(label="Class probabilities")],
        title=b["title"],
        description=f"Logistic Regression | test accuracy: {b['accuracy']:.2%}",
        flagging_mode="never",
    )

keys = list(DATASETS)
demo = gr.TabbedInterface([make_interface(k) for k in keys],
                          [DATASETS[k]["title"] for k in keys])

if __name__ == "__main__":
    demo.launch()
