import json

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet50


def main():

    with open("assets/imagenet_class_index.json", "r") as rf:
        class_index = json.load(rf)
    labels = {int(key): value for key, value in class_index.items()}

    model = resnet50(pretrained=True)
    model.eval()
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    preprocess = transforms.Compose(
        [
            transforms.Resize(316),
            transforms.CenterCrop(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    img = Image.open("assets/angora.png").convert("RGB")
    img_tensor = preprocess(img)
    img_tensor.unsqueeze_(0)
    out = model(img_tensor)
    out = out.data.numpy()

    max_id = np.argmax(out)
    max_prob = np.max(out)

    label = labels[max_id]
    print(label, max_prob)


if __name__ == "__main__":
    main()
