""" Implements functionality for generating images
according to the deep dream blog post by google:
https://research.googleblog.com/2015/07/deepdream-code-example-for-visualizing.html
"""

# TODO:
    # Implement a multiscale representations
    # Implement Laplacian pyriamid gradient normalization
    # Refactor train to include arbitrary target Variable -- DONE
    # Finish CUDA optimizations

# PyTorch NN Library
import torch
from torch.autograd import Variable
import torchvision

# Pretty printing
from tqdm import tqdm

# Display images
import numpy as np
import matplotlib.pyplot as plt
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

def roll(img_variable):
    """ Mimics numpy roll on the x and y axis of an image variable """
    max_roll = min(32, min(img_variable.size()[-2:]))
    x_roll, y_roll = np.random.randint(1, max_roll, size=(2))

    # roll on x single axis
    img_part1 = img_variable[:,:,:,-x_roll:]
    img_part2 = img_variable[:,:,:,:-x_roll]
    img_variable = torch.cat([img_part1, img_part2], dim=-1)

    # roll on y single axis
    img_part1 = img_variable[:,:,-y_roll:,:]
    img_part2 = img_variable[:,:,:-y_roll,:]
    img_variable = torch.cat([img_part1, img_part2], dim=-2)

    return img_variable

def trim_model(model, trim_to):
    features = list(model.features.children())
    trim_to = max(0, min(len(features) - 1, trim_to))
    return torch.nn.Sequential(
            *features[:trim_to])


def train(trimed_model, base_image=None, iterations=10, lr_scheduler=None,
        optim_version="Adam", lr_update_time=None):
    """ returns a torch Variable trained to maximize the given classes
    as predicted by the given model """

    # If no base image was given, create a new variable filled with 0s
    # with dimensions 256x256 and 3 channels and a single batch
    if base_image is None:
        base_image = Variable(torch.randn(1, 3, 256, 256))
    base_image.requires_grad = True

    # Check that if a scheduler is given then update time is also given (and vice versa)
    if (lr_update_time is None) ^ (lr_scheduler is None):
        raise ValueError("lr_update_time requires that an lr_scheduler is given\n"
                "and vice versa")

    # Create an optimizer for the image
    if lr_scheduler is None:
        optimizer = get_optimizer(base_image)
    else:
        learning_rate = lr_scheduler.__next__()
        optimizer = get_optimizer(base_image,
                optim_version=optim_version, learning_rate=learning_rate)

    # Enter training loop
    for step in tqdm(range(iterations)):
        base_image = Variable(base_image.data, requires_grad=True)
        predictions = trimed_model(base_image)
        loss = - predictions.sum()
        loss.backward()

        # Update Base Image
        base_image.add_(base_image.grad / base_image.grad.mean().data[0])
        base_image.grad.data.zero_()

        # update learning rate
        if lr_scheduler is not None and (step + 1) % lr_update_time == 0:
            new_learning_rate = lr_scheduler.__next__()
            optimizer = get_optimizer(base_image,
                    optim_version=optim_version, learning_rate=new_learning_rate)

        if (step + 1) % 10 == 0:
            print("Loss is {}".format(loss.data[0]))

    return base_image

if __name__ == "__main__":
    # test
    img = train(trim_model(get_model(), 9), iterations=10).squeeze()
    img = img.data
    img -= img.min()
    img *= 1 / (img.max() + 0.01)
    print(img)
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)), interpolation="nearest")
    plt.show()
