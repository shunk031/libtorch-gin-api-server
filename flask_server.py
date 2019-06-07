import torchvision.transforms as transforms
from torchvision.models import resnet50

from flask import Flask, request

app = Flask(__name__)

model = resnet50(pretrained=True)
normalize = transforms.Normalize()


@app.route("/")
def get_index_html():
    raise NotImplementedError()


@app.route("/predict")
def predict_proba():
    raise NotImplementedError()
