import wandb
import gradio as gr
from fastai.vision.all import *
from fastai.callback.all import *
from fastai.callback.wandb import *

wandb.login()

def is_cat(x):
    return x[0].isupper()

im = PILImage.create("polyps.jpg")
im.thumbnail((192, 192))

wandb.init()

learn = load_learner("resnet18.pkl")
learn.predict(im)

categories = ("no polyps", "polyps")

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

a = classify_image(im)
print(a)

image = gr.inputs.Image(shape=(192, 192))
label = gr.outputs.Label()

examples = ["polyps.jpg", "no polyps.jpg"]

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)

wandb.log({"interface": iface}, commit=False)

m = learn.model
iface.launch()

ps = list(m.parameters())
print(ps)
