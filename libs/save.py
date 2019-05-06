import os
import subprocess
import yaml
from scipy.misc import imsave
from math import log10, floor

with open("params.yml", 'r') as ymlfile:
    config = yaml.load(ymlfile)


def save_image(img, content_path, style_path):
    if not hasattr(save_image, "iteration"):
        save_image.iteration = 0
    content_image_name = os.path.split(content_path)[1][:-4]
    style_image_name = os.path.split(style_path)[1][:-4]
    name = f"{content_image_name}_{style_image_name}"
    dir_path = os.path.join("output", name)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    zeros_padding = floor(log10(config["number_iterations"]))
    num = str(save_image.iteration+1).rjust(zeros_padding+1, '0')
    img_name = name + "_" + num + ".jpg"
    img_path = os.path.join(dir_path, img_name)
    imsave(img_path, img)
    save_image.iteration += 1
    return name


def save_gif(file_name):
    dir_path = os.path.join("output", file_name)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    os.chdir(dir_path)
    img = file_name + "*.jpg"
    gif = file_name + ".gif"
    cmd = "convert " + "-resize " + "100% " + "-delay " + "50 " + "-loop " +\
          "0 " + img + " " + gif
    subprocess.call(cmd, shell=True)
