#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/aurimas13/CodeAcademy-AI-Course/blob/main/Notebooks_Finished/Computer_Vision_7_L4_Demonstration_3_END.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# 
# | **Topic** | **Contents** |
# | ----- | ----- |
# | **1. Computer vision libraries in PyTorch** | PyTorch has a bunch of built-in helpful computer vision libraries, let's check them out.  |
# | **2. Load data** | To practice computer vision, we'll start with some images of different pieces of clothing from [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist). |
# | **3. Prepare data** | We've got some images, let's load them in with a [PyTorch `DataLoader`](https://pytorch.org/docs/stable/data.html) so we can use them with our training loop. |
# | **4. Model 0: Building a baseline model** | Here we'll create a multi-class classification model to learn patterns in the data, we'll also choose a **loss function**, **optimizer** and build a **training loop**. | 
# | **5. Making predictions and evaluting model 0** | Let's make some predictions with our baseline model and evaluate them. |
# | **6. Setup device agnostic code for future models** | It's best practice to write device-agnostic code, so let's set it up. |
# | **7. Model 1: Adding non-linearity** | Experimenting is a large part of machine learning, let's try and improve upon our baseline model by adding non-linear layers. |
# | **8. Model 2: Convolutional Neural Network (CNN)** | Time to get computer vision specific and introduce the powerful convolutional neural network architecture. |
# | **9. Comparing our models** | We've built three different models, let's compare them. |
# | **10. Evaluating our best model** | Let's make some predictons on random images and evaluate our best model. |
# | **11. Making a confusion matrix** | A confusion matrix is a great way to evaluate a classification model, let's see how we can make one. |
# | **12. Saving and loading the best performing model** | Since we might want to use our model for later, let's save it and make sure it loads back in correctly. |

# Import PyTorch
import torch
from torch import nn⁄⁄

# Import torchvision
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

# Import matplotlib for visualization
import matplotlib.pyplot as plt


# Setup training data
train_data = datasets.FashionMNIST(
    root="data", # where to download data to?
    train=True, # get training data
    download=True, # download data if it doesn't exist on disk
    transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
    target_transform=None # you can transform labels as well
)

# Setup testing data
test_data = datasets.FashionMNIST(
    root="data",
    train=False, # get test data
    download=True,
    transform=ToTensor()
)



# ## 3. Prepare DataLoader

from torch.utils.data import DataLoader

# Setu the batch size hyperparamete
BATCH_SIZE = 32

# Turn datasets into iterables (batches)
train_dataloader = DataLoader(train_data, # dataset to turn into iterable
      batch_size = BATCH_SIZE, # how many sample per batch?
      shuffle = True # shuffle data every epoch?
)

test_dataloader = DataLoader(test_data, 
      batch_size = BATCH_SIZE, 
      shuffle = False # don't necessarily have shuffle the testing data
)

# ## 4. Model 0: Build a baseline model

# Create a flatten layer
flatten_model = nn.Flatten() # all nn modules function as a model (can do forward pass)

# Get a simgle sample
x = train_features_batch[0]

# Flatten the sample
output = flatten_model(x) # perform forward pass

# Print out what happened
print(f"Shape before flattening: {x.shape} -> [color_channels, height, width]")
print(f"Shape after flattening: {output.shape} -> [color_channels, height*width]")

# Let's create our first model using `nn.Flatten()` as the first layer. 

from torch import nn
class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), # neural networks like their inputs in vector form
            nn.Linear(in_features=input_shape, out_features=hidden_units), # in_features = number of features in a data sample (784 pixels)
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )
    
    def forward(self, x):
        return self.layer_stack(x)

# ### 4.1 Setup loss, optimizer and evaluation metrics
# 
# Since we're working on a classification problem, let's bring in prewritten [`helper_functions.py` script](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/helper_functions.py) and subsequently the `accuracy_fn()` [from here](https://www.learnpytorch.io/02_pytorch_classification/).
# 
# > **Note:** Rather than importing and using our own accuracy function or evaluation metric(s), you could import various evaluation metrics from the [TorchMetrics package](https://torchmetrics.readthedocs.io/en/latest/).

# In[16]:


import requests
from pathlib import Path 

# Download helper functions from Learn PyTorch repo (if not already downloaded)
if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  # Note: you need the "raw" GitHub URL for this to work
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)


# In[17]:


# import accuracy  metric
from helper_functions import accuracy_fn # Note: also could use torchmetrics.Accuracy()

# Setup loss funtion and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)


# ### 4.2 Creating a function to time our experiments
# 
# Our timing function will import the [`timeit.default_timer()` function](https://docs.python.org/3/library/timeit.html#timeit.default_timer) from the Python [`timeit` module](https://docs.python.org/3/library/timeit.html).

# In[18]:


from timeit import default_timer as timer 
def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


# ### 4.3 Creating a training loop and training a model on batches of data
# 

# Import tqdm for progress bar
from tqdm.auto import tqdm

# Set the seed and start the timer
torch.manual_seed(42)
train_time_start_on_cpu = timer()

# Set the number of epochs (we'll keep this small for faster training times)
epochs = 3

# Create training and testing loop
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------")
    ### Training
    train_loss = 0
    # Add a loop to loop through training batches
    for batch, (X, y) in enumerate(train_dataloader):
        model_0.train() 
        # 1. Forward pass
        y_pred = model_0(X)

        # 2. Calculate loss (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss # accumulatively add up the loss per epoch 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Print out how many samples have been seen
        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")

    # Divide total train loss by length of train dataloader (average loss per batch per epoch)
    train_loss /= len(train_dataloader)
    
    ### Testing
    # Setup variables for accumulatively adding up loss and accuracy 
    test_loss, test_acc = 0, 0 
    model_0.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            # 1. Forward pass
            test_pred = model_0(X)
           
            # 2. Calculate loss (accumatively)
            test_loss += loss_fn(test_pred, y) # accumulatively add up the loss per epoch

            # 3. Calculate accuracy (preds need to be same as y_true)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
        
        # Calculations on test metrics need to happen inside torch.inference_mode()
        # Divide total test loss by length of test dataloader (per batch)
        test_loss /= len(test_dataloader)

        # Divide total accuracy by length of test dataloader (per batch)
        test_acc /= len(test_dataloader)

    ## Print out what's happening
    print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")

# Calculate training time      
train_time_end_on_cpu = timer()
total_train_time_model_0 = print_train_time(start=train_time_start_on_cpu, 
                                           end=train_time_end_on_cpu,
                                           device=str(next(model_0.parameters()).device))

# We can use this dictionary to compare the baseline model results to other models later on.

# ## 6. Setup device agnostic-code (for using a GPU if there is one)

# Setup device agnostic coe:
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
device


# Create a model with non-linear and linear layers
class FashionMNISTModelV1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), # flatten inputs into single vector
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)



torch.manual_seed(42)
model_1 = FashionMNISTModelV1(input_shape=784, # number of input features
                              hidden_units = 10,
                              output_shape=len(class_names) # number of output classes desired
                            ).to(device) # send model to GPU
next(model_1.parameters()).device # check model device


# ### 7.1 Setup loss, optimizer and evaluation metrics
# 
# As usual, we'll setup a loss function, an optimizer and an evaluation metric (we could do multiple evaluation metrics but we'll stick with accuracy for now).

from helper_functions import accuracy_fn # Note: also could use torchmetrics.Accuracy()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(),
                                   lr=0.1)


# ### 7.2 Functionizing training and test loops

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1)) # Go from logits -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    test_loss, test_acc = 0, 0
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode(): 
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            test_pred = model(X)
            
            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                y_pred=test_pred.argmax(dim=1) # Go from logits -> pred labels
            )
        
        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")
        


# Let's also time things to see how long our code takes to run on the GPU.

torch.manual_seed(42)

# Measure time
from timeit import default_timer as timer
train_time_start_on_gpu = timer()

epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n--------")
    train_step(data_loader=train_dataloader,
               model=model_1,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn
    )
    test_step(data_loader=test_dataloader,
              model=model_1,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn
    )

train_time_end_on_gpu = timer()
total_train_time_model_1 = print_train_time(start=train_time_start_on_gpu,
                                            end=train_time_end_on_gpu,
                                            device=device)


# Let's evaluate our trained `model_1` using our `eval_model()` function and see how it went.

# In[27]:


torch.manual_seed(42)

# Note: this will error due to eval_model() not using device agnostic code
model_1_results = eval_model(model=model_1, 
    data_loader=test_dataloader,
    loss_fn=loss_fn, 
    accuracy_fn=accuracy_fn) 
model_1_results


# <!-- Oh no! 
# 
# It looks like our `eval_model()` function errors out with:
# 
# > `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument mat1 in method wrapper_addmm)` -->
# 
# It's because we've setup our data and model to use device-agnostic code but not our evaluation function.
# 
# How about we fix that by passing a target `device` parameter to our `eval_model()` function?
# 
# Then we'll try calculating the results again.
# 

# In[28]:


# Move values to device
torch.manual_seed(42)
def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn, 
               device: torch.device = device):
    """Evaluates a given model on a given dataset.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.
        device (str, optional): Target device to compute on. Defaults to device.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Send data to the target device
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        
        # Scale loss and acc
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}

# Calculate model 1 results with device-agnostic code 
model_1_results = eval_model(model=model_1, data_loader=test_dataloader,
    loss_fn=loss_fn, accuracy_fn=accuracy_fn,
    device=device
)
model_1_results


# In[29]:


model_0_results


# ## 8. Model 2: Building a Convolutional Neural Network (CNN)

# Create a convolutional neural network 
class FashionMNISTModelV2(nn.Module):
    """
    Model architecture is TinyVGG from: 
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3,
                      stride=1, # default
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.Linear(in_features=hidden_units*7*7, 
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        # print(x.shape)
        x = self.block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x

torch.manual_seed(42)

model_2 = FashionMNISTModelV2(input_shape=1, 
                              hidden_units=10, 
                              output_shape=len(class_names)).to(device)
model_2


# ### 8.1 Stepping through `nn.Conv2d()`


# Create sample of random numbers with same as image batch
images = torch.rand(size=(32, 3, 64, 64)) # [batch_size, color_channels, height, width]
test_image = images[0] # get sample image for testing
print(f'Image batch shape: {images.shape} -> [batch_size, color_channels, height, width]')
print(f'Single image shape: {test_image.shape} -> [color_channels, height, width]')
print(f'Single image pixel values:\n{test_image}')


# Let's create an example `nn.Conv2d()` with various parameters:
# * `in_channels` (int) - Number of channels in the input image.
# * `out_channels` (int) - Number of channels produced by the convolution.
# * `kernel_size` (int or tuple) - Size of the convolving kernel/filter.
# * `stride` (int or tuple, optional) - How big of a step the convolving kernel takes at a time. Default: 1.
# * `padding` (int, tuple, str) - Padding added to all four sides of input. Default: 0.
# 
# 
# ![example of going through the different parameters of a Conv2d layer](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/03-conv2d-layer.gif)
# 
# *Example of what happens when you change the hyperparameters of a `nn.Conv2d()` layer.*


torch.manual_seed(42)

# Create convolutional layer with same dimensions as TinyVGG
conv_layer = nn.Conv2d(in_channels=3,
                       out_channels=10,
                       kernel_size=3,
                       stride=1, 
                       padding=0)

# Pass the data through convolutional layer
conv_layer(test_image) # Note if running PyTorch <1.11.0, this will erro because of shape issue (nn.Conv2d() expects a 4d tensor as input)


# If we try to pass a single image in, we get a shape mismatch error:
# 
# > `RuntimeError: Expected 4-dimensional input for 4-dimensional weight [10, 3, 3, 3], but got 3-dimensional input of size [3, 64, 64] instead`
# >
# > **Note:** If you're running PyTorch 1.11.0+, this error won't occur.
# 
# This is because our `nn.Conv2d()` layer expects a 4-dimensional tensor as input with size `(N, C, H, W)` or `[batch_size, color_channels, height, width]`.
# 
# Right now our single image `test_image` only has a shape of `[color_channels, height, width]` or `[3, 64, 64]`.
# 
# We can fix this for a single image using `test_image.unsqueeze(dim=0)` to add an extra dimension for `N`.
# 

# Add extra dimension to test image
test_image.unsqueeze(dim=0).shape


# Pass test image with extra dimension through conv_layer
conv_layer(test_image.unsqueeze(dim=0)).shape


# Hmm, notice what happens to our shape (the same shape as the first layer of TinyVGG on [CNN Explainer](https://poloclub.github.io/cnn-explainer/)), we get different channel sizes as well as different pixel sizes.

torch.manual_seed(42)

# Create a new conv_layer with different values
conv_layer_2 = nn.Conv2d(in_channels=3,
                       out_channels=10,
                       kernel_size=(5,5),
                       stride=2, 
                       padding=0)


# Pass single image through new conv_layer_2
conv_layer_2(test_image.unsqueeze(dim=0)).shape


# Woah, we get another shape change.
# 
# Now our image is of shape `[1, 10, 30, 30]` (it will be different if you use different values) or `[batch_size=1, color_channels=10, height=30, width=30]`.
# 
# What's going on here?
# 
# Behind the scenes, our `nn.Conv2d()` is compressing the information stored in the image.
# 
# It does this by performing operations on the input (our test image) against its internal parameters.
# 
# The goal of this is similar to all of the other neural networks we've been building.
# 
# Data goes in and the layers try to update their internal parameters (patterns) to lower the loss function thanks to some help of the optimizer.
# 
# The only difference is *how* the different layers calculate their parameter updates or in PyTorch terms, the operation present in the layer `forward()` method.
# 



# Check out the conv_layer_2 internal parameters
print(conv_layer_2.state_dict())


# Look at that! A bunch of random numbers for a weight and bias tensor.
# 
# The shapes of these are manipulated by the inputs we passed to `nn.Conv2d()` when we set it up.
# 
# Let's check them out.
# 



# Get shapes of weight and bias tensors within conv_layer_2
print(f'conv_layer_2 weight shape: \n{conv_layer_2.weight.shape} -> [out_channels=10, in_channels=3, kernel_size=5, kernel_size=5]')
print(f'\nconv_layer_2 bias shape: \n{conv_layer_2.bias.shape} -> [out_channels = 10]')


# > **Question:** What should we set the parameters of our `nn.Conv2d()` layers?
# >
# > That's a good one. But similar to many other things in machine learning, the values of these aren't set in stone (and recall, because these values are ones we can set ourselves, they're referred to as "**hyperparameters**"). 
# >
# > The best way to find out is to try out different values and see how they effect your model's performance.
# >
# > Or even better, find a working example on a problem similar to yours (like we've done with TinyVGG) and copy it. 
# 
# We're working with a different of layer here to what we've seen before.
# 
# But the premise remains the same: start with random numbers and update them to better represent the data.
# 

# ### 8.2 Stepping through `nn.MaxPool2d()`
# Now let's check out what happens when we move data through `nn.MaxPool2d()`.
# 
# 

# In[61]:


# Print out original imae shape without and qith unsqueezed dimension
print(f'Test image original shape: {test_image.shape}')
print(f'Test image with with unsqueezed dimension: {test_image.unsqueeze(dim=0).shape}')

# Create a sample nn.MaxPool2d() layer
max_pool_layer = nn.MaxPool2d(kernel_size=2)

# Pass data through just conv_layer
test_image_through_conv = conv_layer(test_image.unsqueeze(dim=0))
print(f'Shape after going through conv_layer() : {test_image_through_conv.shape}')

# Pass data through the max pool layer
test_image_through_conv_and_max_pool = max_pool_layer(test_image_through_conv)
print(f'Shape after going through conv_layer() and max_pool_layer(): {test_image_through_conv_and_max_pool.shape}')


# Notice the change in the shapes of what's happening in and out of a `nn.MaxPool2d()` layer.
# 
# The `kernel_size` of the `nn.MaxPool2d()` layer will effects the size of the output shape.
# 
# In our case, the shape halves from a `62x62` image to `31x31` image.
# 
# Let's see this work with a smaller tensor.
# 

# In[62]:


torch.manual_seed(42)
# Create a random tensor with a similiar number of dimensions to our images
random_tensor = torch.randn(size=(1, 1, 2, 2))
print(f"Random tensor:\n{random_tensor}")
print(f"Random tensor shape: {random_tensor.shape}")

# Create a max pool layer
max_pool_layer = nn.MaxPool2d(kernel_size=2) 

# Pass the random tensor through the max pool layer
max_pool_tensor = max_pool_layer(random_tensor)
print(f"\nMax pool tensor:\n{max_pool_tensor}")
print(f"Max pool tensor shape: {max_pool_tensor.shape}")


# Notice the final two dimensions between `random_tensor` and `max_pool_tensor`, they go from `[2, 2]` to `[1, 1]`.
# 
# In essence, they get halved.
# 
# And the change would be different for different values of `kernel_size` for `nn.MaxPool2d()`.
# 
# Also notice the value leftover in `max_pool_tensor` is the **maximum** value from `random_tensor`.
# 
# What's happening here?
# 
# This is another important piece of the puzzle of neural networks.
# 
# Essentially, **every layer in a neural network is trying to compress data from higher dimensional space to lower dimensional space**. 
# 
# In other words, take a lot of numbers (raw data) and learn patterns in those numbers, patterns that are predictive whilst also being *smaller* in size than the original values.
# 
# From an artificial intelligence perspective, you could consider the whole goal of a neural network to *compress* information.
# 
# ![each layer of a neural network compresses the original input data into a smaller representation that is (hopefully) capable of making predictions on future input data](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/03-conv-net-as-compression.png)
# 
# This means, that from the point of view of a neural network, intelligence is compression.
# 
# This is the idea of the use of a `nn.MaxPool2d()` layer: take the maximum value from a portion of a tensor and disregard the rest.
# 
# In essence, lowering the dimensionality of a tensor whilst still retaining a (hopefully) significant portion of the information.
# 
# It is the same story for a `nn.Conv2d()` layer.
# 
# Except instead of just taking the maximum, the `nn.Conv2d()` performs a conovlutional operation on the data (see this in action on the [CNN Explainer webpage](https://poloclub.github.io/cnn-explainer/)).
# 
# > **Exercise:** What do you think the [`nn.AvgPool2d()`](https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html) layer does? Try making a random tensor like we did above and passing it through. Check the input and output shapes as well as the input and output values.
# 
# > **Extra-curriculum:** Lookup "most common convolutional neural networks", what architectures do you find? Are any of them contained within the [`torchvision.models`](https://pytorch.org/vision/stable/models.html) library? What do you think you could do with these?

# ### 8.3 Setup a loss function and optimizer for `model_2`
# 
# We've stepped through the layers in our first CNN enough.
# 
# But remember, if something still isn't clear, try starting small.
# 
# Pick a single layer of a model, pass some data through it and see what happens.
# 
# Now it's time to move forward and get to training!
# 
# Let's setup a loss function and an optimizer.
# 
# We'll use the functions as before, `nn.CrossEntropyLoss()` as the loss function (since we're working with multi-class classification data).
# 
# And `torch.optim.SGD()` as the optimizer to optimize `model_2.parameters()` with a learning rate of `0.1`.
# 

# In[63]:


# Setup losss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(),
                            lr=0.1)


# ### 8.4 Training and testing `model_2` using our training and test functions
# 
# Loss and optimizer ready!
# 
# Time to train and test.
# 
# We'll use our `train_step()` and `test_step()` functions we created before.
# 
# We'll also measure the time to compare it to our other models.
# 

# In[64]:


torch.manual_seed(42)

# Measure time
from timeit import default_timer as timer
train_time_start_model_2 = timer()

# Train and test model 
epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_step(data_loader=train_dataloader, 
        model=model_2, 
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn,
        device=device
    )
    test_step(data_loader=test_dataloader,
        model=model_2,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        device=device
    )

train_time_end_model_2 = timer()
total_train_time_model_2 = print_train_time(start=train_time_start_model_2,
                                           end=train_time_end_model_2,
                                           device=device)


# Woah! Looks like the convolutional and max pooling layers helped improve performance a little.
# 
# Let's evaluate `model_2`'s results with our `eval_model()` function.
# 

# In[65]:


# Get model 2 results
model_2_results = eval_model(
    model=model_2, 
    data_loader=test_dataloader,
    loss_fn=loss_fn, 
    accuracy_fn=accuracy_fn) 
model_2_results


# ## 9. Compare model results and training time
# 
# We've trained three different models.
# 
# 1. `model_0` - our baseline model with two `nn.Linear()` layers.
# 2. `model_1` - the same setup as our baseline model except with `nn.ReLU()` layers in between the `nn.Linear()` layers.
# 3. `model_2` - our first CNN model that mimics the TinyVGG architecture on the CNN Explainer website.
# 
# This is a regular practice in machine learning.
# 
# Building multiple models and performing multiple training experiments to see which performs best.
# 
# Let's combine our model results dictionaries into a DataFrame and find out.
# 

# In[66]:


import pandas as pd
compare_results = pd.DataFrame([model_0_results, model_1_results, model_2_results])
compare_results


# Nice!
# 
# We can add the training time values too.

# In[67]:


# Add training times to results for comparison
compare_results['training_time'] = [total_train_time_model_0,
                                    total_train_time_model_1,
                                    total_train_time_model_2]

compare_results


# It looks like our CNN (`FashionMNISTModelV2`) model performed the best (lowest loss, highest accuracy) but had the longest training time.
# 
# And our baseline model (`FashionMNISTModelV0`) performed better than `model_1` (`FashionMNISTModelV1`).
# 
# ### Performance-speed tradeoff
# 
# Something to be aware of in machine learning is the **performance-speed** tradeoff.
# 
# Generally, you get better performance out of a larger, more complex model (like we did with `model_2`).
# 
# However, this performance increase often comes at a sacrifice of training speed and inference speed.
# 
# > **Note:** The training times you get will be very dependant on the hardware you use. 
# >
# > Generally, the more CPU cores you have, the faster your models will train on CPU. And similar for GPUs.
# > 
# > Newer hardware (in terms of age) will also often train models faster due to incorporating technology advances.
# 
# How about we get visual?
# 

# In[68]:


# Viduazlize our model results
compare_results.set_index('model_name')['model_acc'].plot(kind='barh')
plt.xlabel('accuracy (%)')
plt.ylabel('model');


# ## 10. Make and evaluate random predictions with best model
# 
# Alright, we've compared our models to each other, let's further evaluate our best performing model, `model_2`.
# 
# To do so, let's create a function `make_predictions()` where we can pass the model and some data for it to predict on.
# 

# In[69]:


def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # Prepare sample
            sample = torch.unsqueeze(sample, dim=0).to(device) # Add an extra dimension and send sample to device

            # Forward pass (model outputs raw logit)
            pred_logit = model(sample)

            # Get prediction probability (logit -> prediction probability)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

            # Get pred_prob off GPU for further calculations
            pred_probs.append(pred_prob.cpu())
            
    # Stack the pred_probs to turn list into a tensor
    return torch.stack(pred_probs)


# In[70]:


import random
random.seed(42)
test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)

# View the first test sample shape and label
print(f"Test sample image shape: {test_samples[0].shape}\nTest sample label: {test_labels[0]} ({class_names[test_labels[0]]})")


# And now we can use our `make_predictions()` function to predict on `test_samples`.

# In[71]:


# Make predictions on test samples with model 2
pred_probs = make_predictions(model=model_2,
                              data=test_samples)

# View first two prediction probabilities list
pred_probs[:2]


# Excellent!
# 
# And now we can go from prediction probabilities to prediction labels by taking the `torch.argmax()` of the output of the `torch.softmax()` activation function.
# 

# In[72]:


# Turn the prediction probabilities into prediction labells by taking the argmax()
pred_classes = pred_probs.argmax(dim=1)
pred_classes


# In[73]:


# Are predcition in the same form as test labels?
test_labels, pred_classes


# Now our predicted classes are in the same format as our test labels, we can compare.
# 
# Since we're dealing with image data, let's stay true to the data explorer's motto. 
# 
# "Visualize, visualize, visualize!"
# 

# In[74]:


# Plot predictions
plt.figure(figsize=(9, 9))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples):
  # Create a subplot
  plt.subplot(nrows, ncols, i+1)

  # Plot the target image
  plt.imshow(sample.squeeze(), cmap="gray")

  # Find the prediction label (in text form, e.g. "Sandal")
  pred_label = class_names[pred_classes[i]]

  # Get the truth label (in text form, e.g. "T-shirt")
  truth_label = class_names[test_labels[i]] 

  # Create the title text of the plot
  title_text = f"Pred: {pred_label} | Truth: {truth_label}"
  
  # Check for equality and change title colour accordingly
  if pred_label == truth_label:
      plt.title(title_text, fontsize=10, c="g") # green text if correct
  else:
      plt.title(title_text, fontsize=10, c="r") # red text if wrong
  plt.axis(False);


# Well, well, well, doesn't that look good!
# 
# Not bad for a couple dozen lines of PyTorch code!
# 
# 

# ## 11. Making a confusion matrix for further prediction evaluation
# 
# There are many [different evaluation metrics](https://www.learnpytorch.io/02_pytorch_classification/#9-more-classification-evaluation-metrics) we can use for classification problems. 
# 
# One of the most visual is a [confusion matrix](https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/).
# 
# A confusion matrix shows you where your classification model got confused between predicitons and true labels.
# 
# To make a confusion matrix, we'll go through three steps:
# 1. Make predictions with our trained model, `model_2` (a confusion matrix compares predictions to true labels).
# 2. Make a confusion matrix using [`torch.ConfusionMatrix`](https://torchmetrics.readthedocs.io/en/latest/references/modules.html?highlight=confusion#confusionmatrix).
# 3. Plot the confusion matrix using [`mlxtend.plotting.plot_confusion_matrix()`](http://rasbt.github.io/mlxtend/user_guide/plotting/plot_confusion_matrix/).
# 
# Let's start by making predictions with our trained model.
# 

# In[75]:


# Import tqdm for progress bar
from tqdm.auto import tqdm

# 1. Make predictions with trained model
y_preds = []
model_2.eval()
with torch.inference_mode():
  for X, y in tqdm(test_dataloader, desc="Making predictions"):
    # Send data and targets to target device
    X, y = X.to(device), y.to(device)
    # Do the forward pass
    y_logit = model_2(X)
    # Turn predictions from logits -> prediction probabilities -> predictions labels
    y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
    # Put predictions on CPU for evaluation
    y_preds.append(y_pred.cpu())
# Concatenate list of predictions into a tensor
y_pred_tensor = torch.cat(y_preds)


# Wonderful!
# 
# Now we've got predictions, let's go through steps 2 & 3:
# 2. Make a confusion matrix using [`torchmetrics.ConfusionMatrix`](https://torchmetrics.readthedocs.io/en/latest/references/modules.html?highlight=confusion#confusionmatrix).
# 3. Plot the confusion matrix using [`mlxtend.plotting.plot_confusion_matrix()`](http://rasbt.github.io/mlxtend/user_guide/plotting/plot_confusion_matrix/).
# 
# First we'll need to make sure we've got `torchmetrics` and `mlxtend` installed (these two libraries will help us make and visual a confusion matrix).
# 
# > **Note:** If you're using Google Colab, the default version of `mlxtend` installed is 0.14.0 (as of March 2022), however, for the parameters of the `plot_confusion_matrix()` function we'd like use, we need 0.19.0 or higher. 
# 

# In[76]:


# See if torchmetrics exists, if not, install it
try:
    import torchmetrics, mlxtend
    print(f"mlxtend version: {mlxtend.__version__}")
    assert int(mlxtend.__version__.split(".")[1]) >= 19, "mlxtend verison should be 0.19.0 or higher"
except:
    get_ipython().system("pip install -q torchmetrics -U mlxtend # <- Note: If you're using Google Colab, this may require restarting the runtime")
    import torchmetrics, mlxtend
    print(f"mlxtend version: {mlxtend.__version__}")


# To plot the confusion matrix, we need to make sure we've got and [`mlxtend`](http://rasbt.github.io/mlxtend/) version of 0.19.0 or higher.
# 

# In[77]:


# Import mlxtend upgraded version
import mlxtend 
print(mlxtend.__version__)
assert int(mlxtend.__version__.split(".")[1]) >= 19 # should be version 0.19.0 or higher


# `torchmetrics` and `mlxtend` installed, let's make a confusion matrix!
# 
# First we'll create a `torchmetrics.ConfusionMatrix` instance telling it how many classes we're dealing with by setting `num_classes=len(class_names)`.
# 
# Then we'll create a confusion matrix (in tensor format) by passing our instance our model's predictions (`preds=y_pred_tensor`) and targets (`target=test_data.targets`).
# 
# Finally we can plot our confision matrix using the `plot_confusion_matrix()` function from `mlxtend.plotting`.
# 

# In[78]:


from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

# 2. Setup confusion matrix instance and compare predictions to targets
confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
confmat_tensor = confmat(preds=y_pred_tensor,
                         target=test_data.targets)

# 3. Plot the confusion matrix
fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(), # matplotlib likes working with NumPy 
    class_names=class_names, # turn the row and column labels into class names
    figsize=(10, 7)
);


# Woah! Doesn't that look good?
# 
# We can see our model does fairly well since most of the dark squares are down the diagonal from top left to bottom right (and ideal model will have only values in these squares and 0 everywhere else).
# 
# The model gets most "confused" on classes that are similar, for example predicting "Pullover" for images that are actually labelled "Shirt".
# 
# And the same for predicting "Shirt" for classes that are actually labelled "T-shirt/top".
# 
# This kind of information is often more helpful than a single accuracy metric because it tells use *where* a model is getting things wrong.
# 
# It also hints at *why* the model may be getting certain things wrong.
# 
# It's understandable the model sometimes predicts "Shirt" for images labelled "T-shirt/top".
# 
# We can use this kind of information to further inspect our models and data to see how it could be improved.
# 

# ## 12. Save and load best performing model
# 
# Let's finish this demonstration off by saving and loading in our best performing model.
# 
# We can save and load a PyTorch model using a combination of:
# * `torch.save` - a function to save a whole PyTorch model or a model's `state_dict()`. 
# * `torch.load` - a function to load in a saved PyTorch object.
# * `torch.nn.Module.load_state_dict()` - a function to load a saved `state_dict()` into an existing model instance.
# 
# You can see more of these three steps in the [PyTorch saving and loading models documentation](https://pytorch.org/tutorials/beginner/saving_loading_models.html).
# 
# For now, let's save our `model_2`'s `state_dict()` then load it back in and evaluate it to make sure the save and load went correctly. 
# 

# In[79]:


from pathlib import Path

# Create models directory (if it doesn't already exist), see: https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, # create parent directories if needed
                 exist_ok=True # if models directory already exists, don't error
)

# Create model save path
MODEL_NAME = "03_pytorch_computer_vision_model_2.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_2.state_dict(), # only saving the state_dict() only saves the learned parameters
           f=MODEL_SAVE_PATH)


# Now we've got a saved model `state_dict()` we can load it back in using a combination of `load_state_dict()` and `torch.load()`.
# 
# Since we're using `load_state_dict()`, we'll need to create a new instance of `FashionMNISTModelV2()` with the same input parameters as our saved model `state_dict()`.
# 

# In[80]:


# Create a new instance of FashionMNISTModelV2 (the same class as our saved state_dict())
# Note: loading model will error if the shapes here aren't the same as the saved version
loaded_model_2 = FashionMNISTModelV2(input_shape=1, 
                                    hidden_units=10, # try changing this to 128 and seeing what happens 
                                    output_shape=10) 

# Load in the saved state_dict()
loaded_model_2.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# Send model to GPU
loaded_model_2 = loaded_model_2.to(device)


# And now we've got a loaded model we can evaluate it with `eval_model()` to make sure its parameters work similarly to `model_2` prior to saving. 
# 

# In[81]:


torch.manual_seed(42)

# Note: this will error due to eval_model() not using device agnostic code
loaded_model_2_results = eval_model(
    model=loaded_model_2, 
    data_loader=test_dataloader,
    loss_fn=loss_fn, 
    accuracy_fn=accuracy_fn
    )
 
loaded_model_2_results


# Do these results look the same as `model_2_results`?

# In[82]:


model_2_results


# We can find out if two tensors are close to each other using `torch.isclose()` and passing in a tolerance level of closeness via the parameters `atol` (absolute tolerance) and `rtol` (relative tolerance).
# 
# If our model's results are close, the output of `torch.isclose()` should be true.
# 

# In[83]:


# Check to see if results are close to each other (if they are very far away, there may be an error)
torch.isclose(torch.tensor(model_2_results["model_loss"]),
              torch.tensor(loaded_model_2_results["model_loss"]),
              atol=1e-08, # absolute tolerance
              rtol=0.0001) # relative tolerance


# ## Extra-curriculum
# 
# * **Watch:** [MIT's Introduction to Deep Computer Vision](https://www.youtube.com/watch?v=iaSUYvmCekI&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=3) lecture. This will give you a great intuition behind convolutional neural networks.
# * Spend 10-minutes clicking thorugh the different options of the [PyTorch vision library](https://pytorch.org/vision/stable/index.html), what different modules are available?
# * Lookup "most common convolutional neural networks", what architectures do you find? Are any of them contained within the [`torchvision.models`](https://pytorch.org/vision/stable/models.html) library? What do you think you could do with these?
# * For a large number of pretrained PyTorch computer vision models as well as many different extensions to PyTorch's computer vision functionalities check out the [PyTorch Image Models library `timm`](https://github.com/rwightman/pytorch-image-models/) (Torch Image Models) by Ross Wightman.
