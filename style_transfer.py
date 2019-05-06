"""
Neural style transfer using VGG16 (arXiv:1409.1556 [cs.CV]), based on
F. Chollet's implementation in Deep Learning with Python
"""

import os
import time
import yaml
from keras import backend as K
from keras.applications import vgg16
from keras.preprocessing.image import load_img
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
from libs.utils import preprocess_image, get_dimensions, deprocess_image
from libs.losses import get_total_loss
from libs.evaluator import Evaluator
from libs.save import save_image, save_gif


# Load hyper parameters
with open("params.yml", 'r') as ymlfile:
    config = yaml.load(ymlfile)

# Data definition
content_image_path = config["content_image"]
style_image_path = config["style_image"]
target_height = config["image_height"]

# If output directory doesn't exist create it
if not os.path.isdir("output"):
    os.mkdir("output")

# Calculate constant image dimensions to use henceforth
img_dims = get_dimensions(content_image_path, target_height)

content_image = K.constant(preprocess_image(content_image_path, img_dims))
style_image = K.constant(preprocess_image(style_image_path, img_dims))
generated_image = K.placeholder((1, *img_dims, 3))

# VGG expects a single matrix with all three images
input_tensor = K.concatenate([content_image, style_image, generated_image],
                             axis=0)

# Create model using pretrained ImageNet weights and removing the last layers
model = vgg16.VGG16(input_tensor=input_tensor, weights="imagenet",
                    include_top=False)

# Compute loss
loss = get_total_loss(model, img_dims, generated_image)

# Set up a class to evaluate losses and gradients in one go
evaluator = Evaluator(generated_image, loss, img_dims)

# Start generation process
x = preprocess_image(content_image_path, img_dims)
x = x.flatten()  # Optimizer fmin_l_bfgs_b can only work with vectors
gradient_steps = config["steps_per_iteration"]
iterations = config["number_iterations"]
for i in range(iterations):
    print(f"\nStarting iteration {i}")
    start_time = time.time()
    # Run optimization for "gradient_steps" times
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x, fprime=evaluator.grads,
                                     maxfun=gradient_steps)
    print(f"Current loss value: {min_val}")
    img = x.copy().reshape((*img_dims, 3))
    img = deprocess_image(img)
    # Save current generated image
    save_name = save_image(img, content_image_path, style_image_path)
    end_time = time.time()
    print(f"Iteration {i} completed in {end_time-start_time} seconds")
# Save orignal content image resized
img = load_img(content_image_path, target_size=img_dims)
dir_path = os.path.join("output", save_name)
imsave(os.path.join(dir_path, save_name+"_0.jpg"), img)

# Save as gif
if config["generate_gif"]:
    save_gif(save_name)
