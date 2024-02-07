import matplotlib.pyplot as plt
import torch
import numpy as np
import training as training
from IPython.display import display, clear_output
import ipywidgets as widgets


classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat","Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

def plot(model, testloader, classes):
    button = widgets.Button(description="Select New Image")

    def clickButton(b):
        with torch.no_grad():
            clear_output(wait=True)
            display(button)
            model.eval()
            images, labels = next(iter(testloader))
            img = images[0]
            logits = model(img.view(1, -1))
            probabilities = torch.softmax(logits, dim=1)
            ps = probabilities.numpy().squeeze()

            fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
            ax1.axis("off")
            ax1.imshow(img.squeeze(), cmap="viridis")
            ax2.barh(np.arange(len(classes)), ps)
            ax2.set_aspect(0.1)
            ax2.set_yticks(np.arange(len(classes)))
            ax2.set_yticklabels(classes)
            ax2.set_title("Class Probability")
            ax2.set_xlim(0, 1.1)
            plt.tight_layout()
            plt.show()

            model.train()

    display(button)
    button.on_click(clickButton)