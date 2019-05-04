# Style Transfer
Neural style transfer utility based on F. Chollet's Deep Learning with Python
and using the VGG16 network.

Generated images are created in the output folder, in a directory with the
names of the content and style images. If the corresponding setting is set to
true, a gif showing the whole process will be generated as well.

### Usage

All the necessary variables are configured in the "params.yml" file.

The content and style images contain the paths to the corresponding images.

To simplify the generation of new images, all share the same dimensions, set
by the "image_height" parameter.

At the end of the process a gif can be created showing the whole generation
process. This uses the command-line program ImageMagick, so it has to be
installed beforehand.

The style is transferred by iterations, with a new image being saved at the
end of each one. Each iteration performs a number of gradient descent steps.
These parameters basically control the duration and intensity of the process.

The content weight controls the influence of the content in the final image.
The variation weight acts as a regularization parameter to reduce pixelization.
The style weight is divided by the number of style layers and applied to each
one.

The content layer used produces different results depending on the layer used.
Lower layers tend to focus more on low-level details while the higher ones take
more context into account.

The style is taken from different layers, which are specified in the last 
parameter.
