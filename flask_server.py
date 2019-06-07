import torchvision.transforms as transforms
from flask import Flask, render_template, request
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


@app.route("/")
def get_index_html():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_proba():
    img = request.files["file"]
    return type(img)


if __name__ == "__main__":
    app.run(debug=True, host="::", port=5001)
