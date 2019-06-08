import datetime
import json
from io import BytesIO

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from flask import Flask, g, jsonify, render_template, request
from PIL import Image
from torchvision.models import resnet50

app = Flask(__name__)

model = resnet50(pretrained=True)
model.eval()

preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_label():
    with open("assets/imagenet_class_index.json", "r") as rf:
        class_index = json.load(rf)
    return {int(k): v for k, v in class_index.items()}


labels = load_label()


@app.before_request
def before_request():
    g.request_start_time = datetime.datetime.now()

    delta = datetime.datetime.now() - g.request_start_time
    g.request_time = int(delta.total_seconds() * 1000)


@app.route("/")
def get_index_html():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict_proba():
    if request.method == "POST":
        img = Image.open(BytesIO(request.files["file"].read())).convert("RGB")
        with torch.no_grad():
            img_tensor = preprocess(img)
            img_tensor.unsqueeze_(0)
            out = F.softmax(model(img_tensor), dim=1)
            max_id = out.argmax()
            max_prob = out.max()

            app.logger.debug(g.request_time)

            return jsonify(
                {"description": labels[int(max_id)][1], "score": float(max_prob)}
            )


if __name__ == "__main__":
    app.run(debug=True, host="::", port=5001)
