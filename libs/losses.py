from keras import backend as K
import yaml


with open("params.yml", 'r') as ymlfile:
    config = yaml.load(ymlfile)


def get_total_loss(model, img_dims, generated_img):
    content_weight = config["content_weight"]
    style_weight = config["style_weight"]
    variation_weight = config["variation_weight"]
    content_layer = config["content_layer"]
    style_layers = config["style_layers"]

    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    total_loss = K.variable(0.)

    # Get content loss
    layer_features = outputs_dict[content_layer]
    content_features = layer_features[0, :, :, :]    # 0: content
    generated_features = layer_features[2, :, :, :]  # 2: generated
    c_loss = content_weight * get_content_loss(content_features,
                                               generated_features)
    # Get style loss
    s_loss = K.variable(0.)
    for layer in style_layers:
        layer_features = outputs_dict[layer]
        style_features = layer_features[1, :, :, :]      # 1: style
        generated_features = layer_features[2, :, :, :]  # 2: generated
        s_loss_layer = get_style_loss(style_features, generated_features,
                                      img_dims)
        s_loss += (style_weight/len(style_layers)) * s_loss_layer

    # Get variation loss
    v_loss = K.variable(0.)
    v_loss = variation_weight * get_variation_loss(generated_img, img_dims)

    total_loss = c_loss + s_loss + v_loss
    return total_loss


def get_content_loss(content, generated):
    """
    J_content = || activations_content - activations_generated || ^ 2
    """
    # return K.sum(K.square(content - generated))
    return K.sum(K.square(generated - content))


def get_style_loss(style, generated, img_dims):
    """
    J_style = || Gram_style - Gram_generated || ^ 2 Frobenius
    """
    S = gram_matrix(style)
    G = gram_matrix(generated)
    channels = 3
    size = img_dims[0] * img_dims[1]
    return K.sum(K.square(S - G)) / (4. * (channels ** 2) * (size ** 2))


def gram_matrix(x):
    # Rearrange features in required order
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def get_variation_loss(x, img_dims):
    """
    Regularization loss to reduce pixelization in generated image
    """
    img_height, img_width = img_dims
    a = K.square(x[:, :img_height - 1, :img_width - 1, :] -
                 x[:, 1:, :img_width - 1, :])
    b = K.square(x[:, :img_height - 1, :img_width - 1, :] -
                 x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))
