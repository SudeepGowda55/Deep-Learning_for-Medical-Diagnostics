# image classification using the FastAI library and integrating it with Gradio for creating a user interface.

import wandb
import matplotlib.pyplot as plt

from fastai import *
from fastai.metrics import error_rate
from fastai.callback.all import *
from fastai.vision.all import *
from fastai.basics import *
from fastai.imports import *
from fastai.torch_core import *
from fastai.learner import *
from fastai.callback.wandb import *

from fastbook import *

wandb.login()
wandb.init(project="health-care")

# Data Collection and Preprocessing
searches = "polyps", "no polyps"
path = Path("pop")

# Creating directories and downloading images using DuckDuckGo search
# Verify and clean downloaded images
if not path.exists():
    path.mkdir()
    for o in searches:
        dest = path / o
        dest.mkdir(exist_ok=True)
        results = search_images_ddg(f"{o} photo")
        download_images(dest, urls=results[:200])
        # resize_images(dest, max_size=0, dest=dest)
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)

# Data Loading and Preparation
# Define DataBlock for images
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(256, method="squish")],
).dataloaders(path)

dls.show_batch(max_n=16)

learn = vision_learner( dls, models.vgg16, metrics=[error_rate, accuracy], cbs=WandbCallback())
learn.fine_tune(4)

# Step 5: Model Evaluation and Interpretation
# Analyze model performance and make interpretations
interep = ClassificationInterpretation.from_learner(learn)
interep.plot_top_losses(4, figsize=(10, 11))
interep.plot_confusion_matrix()

plt.show()
learn.export("resnet18.pkl")

wandb.finish()