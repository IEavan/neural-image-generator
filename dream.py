""" Implements functionality for generating images
according to the deep dream blog post by google:
https://research.googleblog.com/2015/07/deepdream-code-example-for-visualizing.html
"""

# TODO:
    # Implement a multiscale representations -- UGLY DONE
    # Implement Laplacian pyriamid gradient normalization
    # Refactor train to include arbitrary target Variable -- DONE
    # Finish CUDA optimizations
    # Make it not ugly as fuck

# PyTorch NN Library
import torch
from torch.autograd import Variable
import torchvision

# Pretty printing
from tqdm import tqdm

# Display images
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize
#plt.ion()

#Constants
USE_CUDA = torch.cuda.is_available()
DEFAULT_MODEL = "squeezenet1_0"
SUPPORTED_MODELS = ["resnet18"]

def get_model(model=DEFAULT_MODEL):
    """ Downloads and returns the specified pretrained model.
    throws ValueError if model is not supported.
    """
    if hasattr(torchvision.models, model):
        return getattr(torchvision.models, model)(pretrained=True)
    else:
        raise ValueError("Model {} does not exist".format(model))

def get_optimizer(image, optim_version="Adam", learning_rate=1e-1):
    """ returns a torch.optim optimizer for the specified image variable
    raises ValueError if optimizer does not exist
    """
    if hasattr(torch.optim, optim_version):
        return getattr(torch.optim, optim_version)({image}, lr=learning_rate)
    else:
        raise ValueError("Given optimizer does not exist")

def roll(img_variable, max_roll=32):
    """ Mimics numpy roll on the x and y axis of an image variable """
    max_roll = min(max_roll, min(img_variable.size()[-2:]))
    x_roll, y_roll = np.random.randint(1, max_roll, size=(2))

    # roll on x single axis
    img_part1 = img_variable[:,:,:,-x_roll:]
    img_part2 = img_variable[:,:,:,:-x_roll]
    img_variable = torch.cat([img_part1, img_part2], dim=-1)

    # roll on y single axis
    img_part1 = img_variable[:,:,-y_roll:,:]
    img_part2 = img_variable[:,:,:-y_roll,:]
    img_variable = torch.cat([img_part1, img_part2], dim=-2)

    def undo(img_variable):
        # undo roll on x single axis
        img_part1 = img_variable[:,:,:,x_roll:]
        img_part2 = img_variable[:,:,:,:x_roll]
        img_variable = torch.cat([img_part1, img_part2], dim=-1)

        # undo roll on y single axis
        img_part1 = img_variable[:,:,y_roll:,:]
        img_part2 = img_variable[:,:,:y_roll,:]
        img_variable = torch.cat([img_part1, img_part2], dim=-2)

        return img_variable


    return img_variable, undo

def trim_model(model, trim_to):
    """ returns a torch.nn.Sequential that contains the given model up to
    the layer defined by trim_to
    """
    features = list(model.features.children())
    trim_to = max(0, min(len(features) - 1, trim_to))
    return torch.nn.Sequential(
            *features[:trim_to])

def get_tiled_gradient(trimed_model, base_image, tile_size=128):
    """ returns the gradient update of the image
    resulting from optimizing the output of the trimed_model
    The gradient is computed in tiles, given by the tile size.
    """
    # make sure that base_image is not a variable
    try:
        base_image = base_image.data
    except AttributeError as e:
        pass

    # Create tensors
    shifted_image, undo_roll = roll(base_image, tile_size - 1)
    gradients = torch.zeros(shifted_image.size())

    # iterate over tiles
    height, width = base_image.size()[-2:]
    for y in range(0, max(height - tile_size // 2, tile_size), tile_size):
        for x in range(0, max(width - tile_size // 2, tile_size), tile_size):
            image_tile = shifted_image[:,:,y:y + tile_size, x:x + tile_size]
            image_tile = Variable(image_tile, requires_grad=True)

            # Compute gradients
            feature_channel = trimed_model(image_tile)
            score = feature_channel.sum()
            score.backward()

            # Insert gradients into gradient tensor
            gradients[:, :, y:y + tile_size, x:x + tile_size] = image_tile.grad.data
            trimed_model.zero_grad()
    return undo_roll(gradients)

def train(trimed_model, base_image=None, iterations=5, lr_scheduler=None,
        optim_version="Adam", lr_update_time=None):
    """ returns a torch Variable trained to maximize the given classes
    as predicted by the given model """

    # If no base image was given, create a new variable filled with 0s
    # with dimensions 256x256 and 3 channels and a single batch
    if base_image is None:
        base_image = torch.randn(1, 3, 128, 128)

    # # Check that if a scheduler is given then update time is also given (and vice versa)
    # if (lr_update_time is None) ^ (lr_scheduler is None):
    #     raise ValueError("lr_update_time requires that an lr_scheduler is given\n"
    #             "and vice versa")

    # # Create an optimizer for the image
    # if lr_scheduler is None:
    #     optimizer = get_optimizer(base_image)
    # else:
    #     learning_rate = lr_scheduler.__next__()
    #     optimizer = get_optimizer(base_image,
    #             optim_version=optim_version, learning_rate=learning_rate)
    OCTAVES = 3
    SCALE = 1.4

    for octave in range(OCTAVES):
        print("Octave {}".format(octave))
        if octave > 0:
            base_image = base_image.squeeze()
            base_image = torch.from_numpy(np.transpose(imresize(
                np.transpose(np.copy(base_image.numpy()), (1,2,0)), SCALE), (2,0,1)))
            base_image = base_image.unsqueeze(0)
            base_image = base_image.float()
        for i in tqdm(range(iterations)):
            gradients = get_tiled_gradient(trimed_model, base_image)
            gradients /= gradients.std() + 1e-8
            base_image += gradients

    return base_image

    # Enter training loop
    # for step in tqdm(range(iterations)):


    #     # Update Base Image
    #     base_image.add_(base_image.grad / base_image.grad.mean().data[0])
    #     base_image.grad.data.zero_()

        # # update learning rate
        # if lr_scheduler is not None and (step + 1) % lr_update_time == 0:
        #     new_learning_rate = lr_scheduler.__next__()
        #     optimizer = get_optimizer(base_image,
        #             optim_version=optim_version, learning_rate=new_learning_rate)

        # if (step + 1) % 10 == 0:
        #     print("Loss is {}".format(loss.data[0]))

if __name__ == "__main__":
    if True:
        # test image generation
        img = train(trim_model(get_model(), 9), iterations=5).squeeze()
        img -= img.min()
        img *= 1 / (img.max() + 0.01)
        print(img)
        plt.imshow(np.transpose(img.numpy(), (1, 2, 0)), interpolation="nearest")
        plt.show()
    if False:
        # test roll and roll undo function
        img = torch.randn(1,1,5, 3)
        print(img)
        rolled_img, undo = roll(img)
        print(rolled_img)
        print(undo(rolled_img))
    if False:
        # test get_grad_tiled function
        img = torch.randn(1,3,512,512)
        trimed_model = trim_model(get_model(), 9)
        gradients = get_tiled_gradient(trimed_model, img)
        gradients -= gradients.min()
        gradients /= gradients.max()
        gradients = gradients.squeeze()
        plt.imshow(np.transpose(gradients.numpy(), (1,2,0)), interpolation="nearest")
        plt.show()
