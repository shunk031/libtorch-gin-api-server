import torch
from torchvision.models import resnet50


def main():

    # An instance of your model.
    model = resnet50(pretrained=True)
    # !!! YOU SHOULD CHANGE MODEL TO EVAL MODE !!!
    model.eval()

    # An example input you would normally provide to your model's froward() method.
    example = torch.rand(1, 3, 224, 224)

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, example)

    # Serialize your script module to a file
    traced_script_module.save("assets/model.pt")


if __name__ == "__main__":
    main()
