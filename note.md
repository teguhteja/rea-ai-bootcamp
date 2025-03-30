diff --git a/image-processing/01.Computer-Vision.md b/image-processing/01.Computer-Vision.md
new file mode 100644
index 0000000..1a2b16e
--- /dev/null
+++ b/image-processing/01.Computer-Vision.md
@@ -0,0 +1,85 @@
+# Computer Vision – Mastering AI Bootcamp
+[![How we teach computers to understand pictures](https://storage.googleapis.com/rg-ai-bootcamp/cnn/computer-vision-leo.png)](https://www.youtube.com/watch?v=40riCqvRoMs)
+
+How we teach computers to understand pictures
+
+Source: [TED Talk](https://www.youtube.com/watch?v=40riCqvRoMs)
+
+Computers are very good at being consistent and processing numbers. But they’re blind, sure they can show images for us in our monitor, but it cannot directly interact with us like human-to-human interaction. Like giving us feedback or help us monitor our CCTV feed and let us know if something is suspicious. That’s because until the recent age of AI, computers are blind. There are attempts to emulate vision, which is what the field Computer Vision is. Let’s start with how we see the world.
+
+How humans see the world
+------------------------
+
+Have you ever think about what’s happening when we see with our eyes ? Look around and what do you see ? A chair, a table, a computer, a smartphone and many others. How do we know that object is indeed a chair or A table ?
+
+If we think about it, back when we were younger, our parents or other people would point out an object and call out it’s name. Sometimes it’s explicit, for example “that’s a green apple”. Sometimes it’s implicit, “let’s sit at the chair over there”. We will then observe the object, take notes and give it an appropriate label. So we identify an object and give it a label.
+
+How do computers see the world ?
+--------------------------------
+
+The story began back in 1959 where a group of neurophysiologists showed a cat an array of images, attempting to correlate a response in its brain. They discovered that it responded first to hard edges or lines, and scientifically, this meant that image processing starts with simple shapes like straight or curved edges, not objects. Interesting isn’t it ?
+
+How Computer Vision works
+-------------------------
+
+Which one is correct ? Well, apparently both of them, the field of computer vision does follow how our brain works, so we also start with the edges before the object, but it’s happening so fast that we don’t even think about it anymore.
+
+In any case, we are not going to discuss how our brain works in relation to computers, we are here to learn about the application of computer vision to enable computers to see.
+
+Two essential technologies are used to accomplish this: - A type of machine learning called deep learning, which we already covered earlier - A convolutional neural network (CNN), which we are going to learn next
+
+Convolutional Neural Network (CNN)
+----------------------------------
+
+A Convolutional Neural Network (CNN) is a type of artificial neural network (ANN), that was designed for image recognition using a special type of layer, aptly named a convolutional layer, and has proven very effective to learn from image and image-like data. Regarding image data, CNNs can be used for many different computer vision tasks, such as image processing, classification, segmentation, and object detection.
+
+In total, there are three main types of CNN layers:
+
+*   Convolutional layer
+*   Pooling layer
+*   Fully-connected (FC) layer
+
+![](https://storage.googleapis.com/rg-ai-bootcamp/cnn/convnet-layers.png)
+
+CNN Layers
+
+Source: [Research Gate](https://www.researchgate.net/figure/Simple-Model-of-Convolutional-Neural-Network_fig2_344622537)
+
+From the picture above, the input image goes through the convolution process in the convolution layer and the output is a feature map. The feature map then went through subsampling in the Pooling layer (subsampling layer) which effectively reduces the size by half, and so on, until it reaches the final layer which is a fully connected layer where the input is processed to return a probability between 0 and 1. With each layer, the CNN increases in its complexity, identifying greater portions of the image. Earlier layers focus on simple features, such as colors and edges. As the image data progresses through the layers of the CNN, it starts to recognize larger elements or shapes of the object until it finally identifies the intended object.
+
+> Note: ANN is actually the same Neural Network that we learn earlier, so we’ll use the term Neural Network or NN going forward.
+
+Application of Computer Vision
+------------------------------
+
+Source: [AI Index Stanford](https://aiindex.stanford.edu/wp-content/uploads/2023/04/HAI_AI-Index-Report-2023_CHAPTER_2.pdf)
+
+![](https://storage.googleapis.com/rg-ai-bootcamp/cnn/face-detection.png)
+
+![](https://storage.googleapis.com/rg-ai-bootcamp/cnn/human-pose-estimation.png)
+
+Pose Estimation
+
+![](https://storage.googleapis.com/rg-ai-bootcamp/cnn/semantic-segmentation.png)
+
+semantic-segmentation
+
+![](https://storage.googleapis.com/rg-ai-bootcamp/cnn/medical-image-segmentation.png)
+
+medical-image-segmentation
+
+![](https://storage.googleapis.com/rg-ai-bootcamp/cnn/object-detection.png)
+
+object-detection
+
+![](https://storage.googleapis.com/rg-ai-bootcamp/cnn/which-face-is-real.png)
+
+which-face-is-real
+
+Source: [MIT 6.S191: Convolutional Neural Networks](https://www.youtube.com/watch?v=NmLK_WQBxB4)
+
+![](https://storage.googleapis.com/rg-ai-bootcamp/cnn/self-driving-car.png)
+
+self-driving-car
+
+Back to top
\ No newline at end of file
diff --git a/image-processing/02.Hello World in Image Classification.md b/image-processing/02.Hello World in Image Classification.md
new file mode 100644
index 0000000..047fb22
--- /dev/null
+++ b/image-processing/02.Hello World in Image Classification.md	
@@ -0,0 +1,205 @@
+Title: Hello World in Image Classification – Mastering AI Bootcamp 
+
+The classic “Hello World” when discussing image classification is using CNN on the MNIST dataset. The dataset contains 60,000-item where each MNIST image is a crude 28 x 28 pixel grayscale handwritten digit from “0” to “9.”
+
+Regular NN on MNIST dataset
+---------------------------
+
+Now, can we create an image classifier using a regular neural network (without CNN) ? Yes, we can, actually we already did it back when we are studying Machine Learning/Deep Learning. Here’s the code:
+
+```
+# MNIST solver
+
+import torch
+
+# Load MNIST data
+from torchvision import datasets, transforms
+from torch.utils.data import DataLoader
+
+# Define data transformations
+transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
+
+# Load MNIST data
+mnist_train = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
+mnist_test = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
+
+# Use Data Loader
+train_loader = DataLoader(mnist_train, batch_size=100, shuffle=True)
+test_loader = DataLoader(mnist_test, batch_size=100, shuffle=False)
+
+# Train
+import torch.nn as nn
+
+# Define model
+class MnistModel(nn.Module):
+def __init__(self):
+super().__init__()
+self.linear = nn.Linear(784, 100)
+self.hidden = nn.Linear(100, 10)
+
+def forward(self, xb):
+# Flatten the image tensors using reshape
+xb = xb.reshape(-1, 784)
+out = self.linear(xb)
+out = self.hidden(out)
+return out
+
+# Instantiate the model
+model = MnistModel()
+
+# Define loss function
+loss_fn = nn.CrossEntropyLoss()
+
+# Define optimizer
+learning_rate = 1e-2
+optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
+
+# Define accuracy function
+def accuracy(outputs, labels):
+_, preds = torch.max(outputs, dim=1)
+return torch.tensor(torch.sum(preds == labels).item() / len(preds))
+
+# Train
+total_epochs = 5
+for epoch in range(total_epochs):
+for images, labels in train_loader:
+# Generate predictions
+outputs = model(images)
+loss = loss_fn(outputs, labels)
+# Perform gradient descent
+optimizer.zero_grad()
+loss.backward()
+optimizer.step()
+print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, total_epochs, loss.item()))
+
+# Evaluate
+with torch.no_grad():
+accum_acc = 0
+for images, labels in test_loader:
+outputs = model(images)
+loss = loss_fn(outputs, labels)
+acc = accuracy(outputs, labels)
+accum_acc += acc
+
+print('Test loss: {:.4f}, Test accuracy: {:.4f}'.format(loss.item(), accum_acc/len(test_loader)))
+```
+
+```
+Epoch [1/5], Loss: 0.4527
+Epoch [2/5], Loss: 0.2594
+Epoch [3/5], Loss: 0.3485
+Epoch [4/5], Loss: 0.5416
+Epoch [5/5], Loss: 0.4624
+Test loss: 0.4252, Test accuracy: 0.9089
+```
+
+Hey, it works, it also have 91% accuracy. There is no problem right ?
+
+Well, on a simple image such as the MNIST dataset, which contains only black and white colors as well as simple shapes, that’s true. However, the images we encounter in the real world are far more complex and diverse in terms of colors, textures, and objects.
+
+To tackle these challenges effectively, specialized neural network architectures like Convolutional Neural Networks (CNNs) have emerged as the preferred choice, as they are designed to capture spatial hierarchies, local features, and patterns, making them well-suited for the diverse and intricate nature of real-world images.
+
+> Note: - Spatial Hierarchies: The network can learn to recognize patterns, shapes, and structures in an image in a hierarchical manner. This involves identifying simple features (such as edges and corners) at lower levels and progressively combining them to recognize more complex and abstract objects or concepts at higher levels. - Local Features: The network can identify and focus on specific regions or elements within an image that are relevant for recognition or classification. These local features can be small patterns, textures, or details within an image that contribute to the overall understanding of what the image represents.
+
+CNN on MNIST dataset
+--------------------
+
+Let’s try converting the code above to it’s CNN version
+
+```
+import torch
+import torch.nn as nn
+import torch.optim as optim
+from torchvision import datasets, transforms
+from torch.utils.data import DataLoader
+
+# Define a CNN model for MNIST
+class CNNModel(nn.Module):
+def __init__(self):
+super(CNNModel, self).__init__()
+self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
+self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
+self.fc1 = nn.Linear(64 * 5 * 5, 128)
+self.fc2 = nn.Linear(128, 10)
+
+def forward(self, x):
+x = torch.relu(self.conv1(x))
+x = torch.max_pool2d(x, 2)
+x = torch.relu(self.conv2(x))
+x = torch.max_pool2d(x, 2)
+x = x.view(x.size(0), -1)
+x = torch.relu(self.fc1(x))
+x = self.fc2(x)
+return x
+
+# Define data transformations
+transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
+
+# Load MNIST data
+mnist_train = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
+mnist_test = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
+
+# Use Data Loader
+train_loader = DataLoader(mnist_train, batch_size=100, shuffle=True)
+test_loader = DataLoader(mnist_test, batch_size=100, shuffle=False)
+
+# Instantiate the CNN model
+cnn_model = CNNModel()
+
+# Define loss function and optimizer
+loss_fn = nn.CrossEntropyLoss()
+learning_rate = 0.01
+optimizer = optim.SGD(cnn_model.parameters(), lr=learning_rate)
+
+# Define accuracy function
+def accuracy(outputs, labels):
+_, preds = torch.max(outputs, dim=1)
+return torch.tensor(torch.sum(preds == labels).item() / len(preds))
+
+# Training loop
+total_epochs = 5
+for epoch in range(total_epochs):
+for images, labels in train_loader:
+outputs = cnn_model(images)
+loss = loss_fn(outputs, labels)
+
+optimizer.zero_grad()
+loss.backward()
+optimizer.step()
+
+print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, total_epochs, loss.item()))
+
+# Evaluation
+#cnn_model.eval()
+with torch.no_grad():
+accum_acc = 0
+for images, labels in test_loader:
+outputs = cnn_model(images)
+
+loss = loss_fn(outputs, labels)
+acc = accuracy(outputs, labels)
+accum_acc += acc
+
+print('Test loss: {:.4f}, Test accuracy: {:.4f}'.format(loss.item(), accum_acc/len(test_loader)))
+```
+
+```
+Epoch [1/5], Loss: 0.2483
+Epoch [2/5], Loss: 0.2432
+Epoch [3/5], Loss: 0.0879
+Epoch [4/5], Loss: 0.1307
+Epoch [5/5], Loss: 0.0887
+Test loss: 0.1283, Test accuracy: 0.9694
+```
+
+Wow, 97% accuracy!
+
+Our new code defines a CNN model with two convolutional layers followed by fully connected layers. It also normalizes the data to have a mean of 0.5 and a standard deviation of 0.5. Normalizing the data ensures that the pixel values have a consistent scale, usually between 0 and 1 or -1 and 1. This helps stabilize training, as large input values can lead to unstable gradients during backpropagation.
+
+Still not convinced, well you can try modifying the code above to use the CIFAR10 which you can find on Huggingface. The CIFAR10 dataset presents a more complex challenge compared to MNIST due to its color images (32x32 pixels RGB) and diverse set of object categories (including animals, vehicles, and everyday objects).
+
+We’ll skip the CIFAR10 notebook, but if you are interested in the result, you can visit this notebook: NN vs CNN CIFAR10
+
+Let’s continue to how CNN works.
+
+Back to top
\ No newline at end of file
diff --git a/image-processing/03.CIFAR10 comparison for regular Neural Network vs CNN.md b/image-processing/03.CIFAR10 comparison for regular Neural Network vs CNN.md
new file mode 100644
index 0000000..2f52fad
--- /dev/null
+++ b/image-processing/03.CIFAR10 comparison for regular Neural Network vs CNN.md	
@@ -0,0 +1,334 @@
+Title: CIFAR10 comparison for regular Neural Network vs CNN – Mastering AI Bootcamp 
+
+CIFAR-10 dataset
+----------------
+
+The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
+
+The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.
+
+The classes are completely mutually exclusive. There is no overlap between automobiles and trucks. “Automobile” includes sedans, SUVs, things of that sort. “Truck” includes only big trucks. Neither includes pickup trucks.
+
+Source: CIFAR 10 Dataset - cs.toronto.edu
+
+### Let’s explore the dataset:
+
+```
+!pip install torch torchvision matplotlib
+```
+
+```
+import torch
+import torchvision
+import torchvision.transforms as transforms
+import matplotlib.pyplot as plt
+import numpy as np
+
+# Define data transformations (you can customize these)
+transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
+
+# Load the training dataset
+trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
+trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, num_workers=2)
+
+# Define class labels
+classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
+
+# Function to display an image
+def imshow(img):
+img = img / 2 + 0.5  # Unnormalize the image
+npimg = img.numpy()
+plt.imshow(np.transpose(npimg, (1, 2, 0)))
+plt.show()
+
+# Function to display a grid of images from a specified class with labels
+def show_images_from_class_with_labels(class_id, num_images=10):
+# Find indices of images from the specified class
+class_indices = [i for i, label in enumerate(trainset.targets) if label == class_id]
+
+# Randomly select num_images indices from the class
+selected_indices = np.random.choice(class_indices, num_images, replace=False)
+
+# Create a grid for displaying images and labels
+images_grid = []
+labels_grid = []
+
+for idx in selected_indices:
+image, label = trainset[idx]
+# Convert image tensor to a NumPy array and reshape it from (C, H, W) to (H, W, C)
+image = np.transpose(image.numpy(), (1, 2, 0))
+
+# Normalize the image data to be in the [0, 1] range
+image = (image - image.min()) / (image.max() - image.min())
+
+images_grid.append(image)
+
+# Show the grid of images and one label for the class
+fig, axs = plt.subplots(1, 2, figsize=(12, 1))  # Create a 1x2 grid of subplots
+
+axs[0].axis('off')  # Turn off axis for labels
+axs[0].text(0.5, 0.5, f'Class: {classes[class_id]}', ha='center', va='center', fontsize=12)
+
+axs[1].axis('off')  # Turn off axis for images
+axs[1].imshow(np.concatenate(images_grid, axis=1))  # Concatenate images horizontally
+
+plt.show()
+
+# Display 10 random images from each class with labels
+for class_id, class_name in enumerate(classes):
+show_images_from_class_with_labels(class_id)
+```
+
+```
+Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz
+```
+
+```
+100%|██████████| 170498071/170498071 [43:03<00:00, 66006.41it/s]  
+```
+
+```
+Extracting ./data/cifar-10-python.tar.gz to ./data
+```
+
+Regular NN version:
+-------------------
+
+Let’s try again using the NN version:
+
+```
+import ssl #only for localhost
+ssl._create_default_https_context = ssl._create_unverified_context #only for localhost
+
+# MNIST solver
+
+import torch
+
+# Load MNIST data
+from torchvision import datasets, transforms
+from torch.utils.data import DataLoader
+
+# Define data transformations
+transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
+
+# Load MNIST data
+cifar10_train = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
+cifar10_test = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
+
+# Use Data Loader
+train_loader = DataLoader(cifar10_train, batch_size=100, shuffle=True)
+test_loader = DataLoader(cifar10_test, batch_size=100, shuffle=False)
+
+# Train
+import torch.nn as nn
+
+# Define model
+class FNNModel(nn.Module):
+def __init__(self):
+super(FNNModel, self).__init__()
+self.fc1 = nn.Linear(3 * 32 * 32, 128)  # Adjust input size for CIFAR-10 (3 color channels, 32x32 pixels)
+self.fc2 = nn.Linear(128, 10)  # 10 classes for CIFAR-10
+
+def forward(self, x):
+x = x.view(x.size(0), -1)  # Flatten the input
+x = torch.relu(self.fc1(x))
+x = self.fc2(x)
+return x
+
+# Instantiate the model
+model = FNNModel()
+
+# Define loss function
+loss_fn = nn.CrossEntropyLoss()
+
+# Define optimizer
+learning_rate = 1e-2
+optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
+
+# Define accuracy function
+def accuracy(outputs, labels):
+_, preds = torch.max(outputs, dim=1)
+return torch.tensor(torch.sum(preds == labels).item() / len(preds))
+
+# Train
+total_epochs = 10
+model.train()
+for epoch in range(5):
+for images, labels in train_loader:
+# Generate predictions
+outputs = model(images)
+loss = loss_fn(outputs, labels)
+# Perform gradient descent
+optimizer.zero_grad()
+loss.backward()
+optimizer.step()
+print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, total_epochs, loss.item()))
+
+# Evaluate
+model.eval()
+with torch.no_grad():
+accum_acc = 0
+for images, labels in test_loader:
+outputs = model(images)
+loss = loss_fn(outputs, labels)
+acc = accuracy(outputs, labels)
+accum_acc += acc
+
+print('Test loss: {:.4f}, Test accuracy: {:.4f}'.format(loss.item(), accum_acc/len(test_loader)))
+```
+
+```
+Files already downloaded and verified
+Files already downloaded and verified
+Epoch [1/10], Loss: 1.8312
+Epoch [2/10], Loss: 1.7067
+Epoch [3/10], Loss: 1.6943
+Epoch [4/10], Loss: 1.5868
+Epoch [5/10], Loss: 1.5829
+Test loss: 1.5123, Test accuracy: 0.4600
+```
+
+Only 46% accuracy.
+
+CNN Version
+-----------
+
+Let’s also try the CNN version:
+
+```
+import torch
+import torch.nn as nn
+import torch.optim as optim
+from torchvision import datasets, transforms
+from torch.utils.data import DataLoader
+
+# Define a CNN model for CIFAR-10
+class CNNModel(nn.Module):
+def __init__(self):
+super(CNNModel, self).__init__()
+self.conv1 = nn.Conv2d(3, 32, kernel_size=3)  # Adjust input channels for CIFAR-10 (3 color channels)
+self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
+self.fc1 = nn.Linear(64 * 6 * 6, 128)  # Adjust input size for CIFAR-10 (32x32 images)
+self.fc2 = nn.Linear(128, 10)  # 10 classes for CIFAR-10
+
+def forward(self, x):
+x = torch.relu(self.conv1(x))
+x = torch.max_pool2d(x, 2)
+x = torch.relu(self.conv2(x))
+x = torch.max_pool2d(x, 2)
+x = x.view(x.size(0), -1)
+x = torch.relu(self.fc1(x))
+x = self.fc2(x)
+return x
+
+# Define data transformations
+transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
+
+# Load CIFAR-10 data
+cifar10_train = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
+cifar10_test = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
+
+# Use Data Loader
+train_loader = DataLoader(cifar10_train, batch_size=100, shuffle=True)
+test_loader = DataLoader(cifar10_test, batch_size=100, shuffle=False)
+
+# Instantiate the CNN model
+cnn_model = CNNModel()
+
+loss_fn = nn.CrossEntropyLoss()
+learning_rate = 0.01
+optimizer = optim.SGD(cnn_model.parameters(), lr=learning_rate)
+
+# Define accuracy function
+def accuracy(outputs, labels):
+_, preds = torch.max(outputs, dim=1)
+return torch.tensor(torch.sum(preds == labels).item() / len(preds))
+
+total_epochs = 10
+# Training loop
+cnn_model.train()
+for epoch in range(total_epochs):
+for images, labels in train_loader:
+outputs = cnn_model(images)
+loss = loss_fn(outputs, labels)
+
+optimizer.zero_grad()
+loss.backward()
+optimizer.step()
+
+print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, total_epochs, loss.item()))
+
+# Evaluation
+cnn_model.eval()
+with torch.no_grad():
+accum_acc = 0
+for images, labels in test_loader:
+outputs = cnn_model(images)
+
+loss = loss_fn(outputs, labels)
+acc = accuracy(outputs, labels)
+accum_acc += acc
+
+print('Test loss: {:.4f}, Test accuracy: {:.4f}'.format(loss.item(), accum_acc/len(test_loader)))
+```
+
+```
+Files already downloaded and verified
+Files already downloaded and verified
+Epoch [1/10], Loss: 2.1188
+Epoch [2/10], Loss: 1.9551
+Epoch [3/10], Loss: 1.6204
+Epoch [4/10], Loss: 1.5244
+Epoch [5/10], Loss: 1.4974
+Epoch [6/10], Loss: 1.3608
+Epoch [7/10], Loss: 1.4381
+Epoch [8/10], Loss: 1.3200
+Epoch [9/10], Loss: 1.2991
+Epoch [10/10], Loss: 1.1985
+Test loss: 1.2555, Test accuracy: 0.5520
+```
+
+Wow, 55% accuracy!
+
+Result
+------
+
+We can see the gap widens now, 45% vs 55%, compared to 91% vs 97% on the MNIST dataset.
+
+You can increase the epochs to find out when one of model reaches 90%, which one do you think ?
+
+I hope I make my point across that CNN is far superior to regular NN when we are working with images.
+
+Exercise CNN
+------------
+
+```
+!pip install rggrader
+```
+
+```
+# @title #### Student Identity
+student_id = "student_id" # @param {type:"string"}
+name = "your_name" # @param {type:"string"}
+```
+
+```
+# @title #### 00. CIFAR10 Dataset Accuracy
+
+from rggrader import submit
+
+# TODO: Improve the accuracy of the CNN model using the CIFAR10 dataset above. Write your code here.
+
+# You may add any code here to derive your variables
+# Please change this
+accuracy = 5
+
+print(f"The accuracy is {accuracy}")
+
+# Submit Method
+assignment_id = "03_cnn"
+question_id = "00_cifar10_accuracy"
+submit(student_id, name, assignment_id, str(accuracy), question_id, "")
+```
+
+Back to top
\ No newline at end of file
diff --git a/image-processing/04.Convolution Layer.md b/image-processing/04.Convolution Layer.md
new file mode 100644
index 0000000..1067ffa
--- /dev/null
+++ b/image-processing/04.Convolution Layer.md	
@@ -0,0 +1,495 @@
+Title: Convolution Layer – Mastering AI Bootcamp 
+
+Let’s see how CNN actually does it’s magic!
+
+CNN is short for Convolutional Neural Network, and convolution process as well as mathematical formula is the primary factor. Now, before we go deeper into the math, let’s check out how the convolution process works.
+
+Convolution Process
+-------------------
+
+The convolution process refers to the action of mapping a filter called kernel across the image and performing a convolution mathematical operations to produce a feature map. It’s easier to show an animation of this:
+
+(Source: Miro)
+
+In the animation above, we can see an image of 5x5 being convoluted with a 3x3 kernel resulting in a 3x3 feature map.
+
+Convolution Operation
+---------------------
+
+The convolution operation of two arrays \\(a\\) and \\(b\\) is denoted by \\(a \* b\\) and defined as:
+
+\\\[ (a \* b)\_{n} = \\sum\_{i=1} a\_{i} b\_{n-i} \\\]
+
+Let’s see how this works in practice. Let’s say we have an \\(A\\) is \\(\[1, 2, 3, 4, 5\]\\) and \\(B\\) is \\(\[10, 9, 8\]\\). The convolution operation of \\(A\\) and \\(B\\) is:
+
+\\\[ \\begin{align} (a \* b)\_{2} &= \\sum\_{i=1} a\_{i} b\_{2-i} \\\\ &= a\_{1} b\_{1} \\end{align} \\\]
+
+\\\[ \\begin{align} (a \* b)\_{3} &= \\sum\_{i=1} a\_{i} b\_{3-i} \\\\ &= a\_{1} b\_{2} + a\_{2} b\_{1} \\end{align} \\\]
+
+\\\[ \\begin{align} (a \* b)\_{4} &= \\sum\_{i=1} a\_{i} b\_{4-i} \\\\ &= a\_{1} b\_{3} + a\_{2} b\_{2} + a\_{3} b\_{1} \\end{align} \\\]
+
+Confusing? Let’s watch the following video, it’s actually pretty simple.
+
+Convolution Operation in Python
+-------------------------------
+
+In Python, we can use `numpy.convolve` to perform the convolution operation.
+
+```
+import numpy as np
+
+a = [1, 2, 3]
+b = [4, 5, 6]
+
+print(np.convolve(a, b, 'full'))
+```
+
+```
+[ 4 13 28 27 18]
+```
+
+`same` parameter will make sure the output has the same length as the input.
+
+```
+import numpy as np
+
+a = [1, 2, 3]
+b = [4, 5, 6]
+
+print(np.convolve(a, b, 'same'))
+```
+
+```
+[13 28 27]
+```
+
+`valid` parameter will make sure the calculation is only performed where the input and the filter fully overlap.
+
+```
+import numpy as np
+
+a = [1, 2, 3]
+b = [4, 5, 6]
+
+print(np.convolve(a, b, 'valid'))
+```
+
+```
+[28]
+```
+
+In the above example, the output is only calculated for \\(1 \* 6 + 2 \* 5 + 3 \* 4 = 28\\).
+
+```
+import numpy as np
+
+a = [1, 2, 3]
+b = [4, 5, 6, 7]
+
+print(np.convolve(a, b, 'valid'))
+```
+
+```
+[28 34]
+```
+
+How about 2D?
+
+It’s possible to use numpy to perform 2D convolution operation, but it’s not as simple as 1D. We’ll use `scipy.signal.convolve2d` instead.
+
+```
+import numpy as np
+from scipy.signal import convolve2d
+
+# Given arrays
+a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
+b = np.array([[1, 0], [1, 0]])
+
+# 'valid' means that we only compute the convolution where the input arrays fully overlap
+c = convolve2d(a, b, mode='valid')
+
+print(c)
+```
+
+```
+[[ 7  9]
+[13 15]]
+```
+
+Feature Extraction
+------------------
+
+But that still doesn’t answer how does this process help detect edges ?
+
+> Note: To better illustrate, we prepare a spreadsheet for a quick simulation of Convolution: GSheet Link
+> 
+> So you can try it for yourself, it’s much easier using Google Sheet. Remember to make a copy for your own use.
+
+Now, let’s put this knowledge into action:
+
+Let’s say we have a 16x16 grid containing a letter H. And we have a 3x3 kernel with the identity, meaning the only activated is the center number.
+
+```
+0 0 0
+0 1 0
+0 0 0
+```
+
+```
+!pip install matplotlib numpy scipy
+```
+
+```
+import matplotlib.pyplot as plt
+import numpy as np
+from scipy.signal import convolve2d
+
+def simulate_convolution(input_grid, kernel):
+# Get the size of the input grid
+grid_size = input_grid.shape[0]
+
+# Perform convolution
+feature_map = convolve2d(input_grid, kernel, 'same')
+
+# Create a figure and subplots
+fig, axs = plt.subplots(1, 3, figsize=(12, 4))
+
+# Plot the input grid on the left
+axs[0].imshow(input_grid, cmap='gray')
+axs[0].set_title('Input Grid')
+
+# Plot the kernel in the middle
+axs[1].imshow(kernel, cmap='gray')
+axs[1].set_title('Kernel')
+
+# Plot the feature map on the right
+axs[2].imshow(feature_map, cmap='gray')
+axs[2].set_title('Feature Map')
+
+# Remove axis labels and ticks
+for ax in axs:
+ax.set_xticks([])
+ax.set_yticks([])
+# Show the grids
+plt.show()
+
+print("input_grid", input_grid, sep='\n')
+print("kernel", kernel, sep='\n')
+print("feature_map", feature_map, sep='\n')
+
+# Create a 16x16 input grid with the letter "H"
+grid_size = 16
+input_grid = np.zeros((grid_size, grid_size))
+
+# Draw the letter "H" on the input grid
+# Horizontal line
+input_grid[7, 3:12] = 1
+# Vertical lines
+input_grid[4:12, 3] = 1
+input_grid[4:12, 12] = 1
+
+# Create a 3x3 identity kernel
+conv_kernel = np.array([[0, 0, 0],
+[0, 1, 0],
+[0, 0, 0]])
+
+# Call the function to simulate convolution
+simulate_convolution(input_grid, conv_kernel)
+```
+
+The result on the right is the same letter, what happens if we change the kernel to:
+
+```
+0.00    0.20    0.00
+0.20    0.20    0.20
+0.00    0.20    0.00
+```
+
+```
+# Create a 3x3 blur kernel
+conv_kernel = np.array([[0, 0.2, 0],
+[0.2, 0.2, 0.2],
+[0, 0.2, 0]])
+
+# Call the function to simulate convolution
+simulate_convolution(input_grid, conv_kernel)
+```
+
+```
+input_grid
+[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
+[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
+[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
+[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
+[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
+[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
+[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
+[0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0.]
+[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
+[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
+[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
+[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
+[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
+[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
+[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
+[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
+kernel
+[[0.  0.2 0. ]
+[0.2 0.2 0.2]
+[0.  0.2 0. ]]
+feature_map
+[[0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ]
+[0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ]
+[0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ]
+[0.  0.  0.  0.2 0.  0.  0.  0.  0.  0.  0.  0.  0.2 0.  0.  0. ]
+[0.  0.  0.2 0.4 0.2 0.  0.  0.  0.  0.  0.  0.2 0.4 0.2 0.  0. ]
+[0.  0.  0.2 0.6 0.2 0.  0.  0.  0.  0.  0.  0.2 0.6 0.2 0.  0. ]
+[0.  0.  0.2 0.6 0.4 0.2 0.2 0.2 0.2 0.2 0.2 0.4 0.6 0.2 0.  0. ]
+[0.  0.  0.2 0.8 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.8 0.2 0.  0. ]
+[0.  0.  0.2 0.6 0.4 0.2 0.2 0.2 0.2 0.2 0.2 0.4 0.6 0.2 0.  0. ]
+[0.  0.  0.2 0.6 0.2 0.  0.  0.  0.  0.  0.  0.2 0.6 0.2 0.  0. ]
+[0.  0.  0.2 0.6 0.2 0.  0.  0.  0.  0.  0.  0.2 0.6 0.2 0.  0. ]
+[0.  0.  0.2 0.4 0.2 0.  0.  0.  0.  0.  0.  0.2 0.4 0.2 0.  0. ]
+[0.  0.  0.  0.2 0.  0.  0.  0.  0.  0.  0.  0.  0.2 0.  0.  0. ]
+[0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ]
+[0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ]
+[0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ]]
+```
+
+The image is blurred right ? Now, as we change the kernel, the feature map will change accordingly.
+
+Can we use the kernel to detect horizontal line and vertical line ?
+
+```
+# Create a 3x3 vertical line detection kernel
+conv_kernel = np.array([[0, 1, 0],
+[0, 1, 0],
+[0, 1, 0]])
+
+# Call the function to simulate convolution
+simulate_convolution(input_grid, conv_kernel)
+```
+
+```
+input_grid
+[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
+[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
+[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
+[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
+[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
+[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
+[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
+[0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0.]
+[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
+[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
+[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
+[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
+[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
+[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
+[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
+[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
+kernel
+[[0 1 0]
+[0 1 0]
+[0 1 0]]
+feature_map
+[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
+[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
+[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
+[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
+[0. 0. 0. 2. 0. 0. 0. 0. 0. 0. 0. 0. 2. 0. 0. 0.]
+[0. 0. 0. 3. 0. 0. 0. 0. 0. 0. 0. 0. 3. 0. 0. 0.]
+[0. 0. 0. 3. 1. 1. 1. 1. 1. 1. 1. 1. 3. 0. 0. 0.]
+[0. 0. 0. 3. 1. 1. 1. 1. 1. 1. 1. 1. 3. 0. 0. 0.]
+[0. 0. 0. 3. 1. 1. 1. 1. 1. 1. 1. 1. 3. 0. 0. 0.]
+[0. 0. 0. 3. 0. 0. 0. 0. 0. 0. 0. 0. 3. 0. 0. 0.]
+[0. 0. 0. 3. 0. 0. 0. 0. 0. 0. 0. 0. 3. 0. 0. 0.]
+[0. 0. 0. 2. 0. 0. 0. 0. 0. 0. 0. 0. 2. 0. 0. 0.]
+[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
+[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
+[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
+[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
+```
+
+```
+# Create a 3x3 vertical line detection kernel
+conv_kernel = np.array([[0, 0, 0],
+[1, 1, 1],
+[0, 0, 0]])
+
+# Call the function to simulate convolution
+simulate_convolution(input_grid, conv_kernel)
+```
+
+```
+input_grid
+[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
+[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
+[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
+[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
+[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
+[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
+[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
+[0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0.]
+[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
+[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
+[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
+[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
+[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
+[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
+[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
+[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
+kernel
+[[0 0 0]
+[1 1 1]
+[0 0 0]]
+feature_map
+[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
+[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
+[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
+[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
+[0. 0. 1. 1. 1. 0. 0. 0. 0. 0. 0. 1. 1. 1. 0. 0.]
+[0. 0. 1. 1. 1. 0. 0. 0. 0. 0. 0. 1. 1. 1. 0. 0.]
+[0. 0. 1. 1. 1. 0. 0. 0. 0. 0. 0. 1. 1. 1. 0. 0.]
+[0. 0. 1. 2. 3. 3. 3. 3. 3. 3. 3. 3. 2. 1. 0. 0.]
+[0. 0. 1. 1. 1. 0. 0. 0. 0. 0. 0. 1. 1. 1. 0. 0.]
+[0. 0. 1. 1. 1. 0. 0. 0. 0. 0. 0. 1. 1. 1. 0. 0.]
+[0. 0. 1. 1. 1. 0. 0. 0. 0. 0. 0. 1. 1. 1. 0. 0.]
+[0. 0. 1. 1. 1. 0. 0. 0. 0. 0. 0. 1. 1. 1. 0. 0.]
+[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
+[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
+[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
+[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
+```
+
+Yes we can!
+
+Convolution Demo
+----------------
+
+Again, we prepare a spreadsheet for a quick simulation of Convolution: GSheet Link
+
+So you can try it for yourself, it’s much easier using Google Sheet. Remember to make a copy for your own use.
+
+Convolution Layer
+-----------------
+
+The input data to a convolutional layer is usually in 3-dimensions: **height**, **width** and **depth**. Height and weight clearly refers to the dimension of the image. But what about depth ? Depth here simply refers to the image channels, in the case of RGB it has a depth of 3, for grayscale image it has a depth of 1.
+
+Kernel
+------
+
+The convolution layer then takes the input and apply the kernel to an area of the image and a dot product is calculated between the input pixels and the kernel. The kernel size is usually 3x3 but it can be adjusted. A larger kernel naturally covers a larger area and better detect large shapes or objects but less adapt to detecting the finer details such as edges, corners or textures which are better performed with a small kernel.
+
+Convolution Animation
+
+Source: Hochschule Der Medien
+
+Then how can we create a kernel matrix? Is it by hand or is there a way to create it automatically?
+
+Well, it turns out we can define the kernel size, but the kernel matrix itself is learned by the CNN Neural Network. Here’s how it works:
+
+*   Initially the values within the kernel matrix are randomly initialized. These random values do not represent any specific pattern.
+*   During the training process of the CNN, the network learns the optimal values for the kernel matrix by adjusting the values within the kernel to minimize the error.
+*   Once training is complete, the learned kernel matrix is then used for feature extraction during the convolution operation.
+
+Strides
+-------
+
+We know that the kernel moves across the image, but how it moves and steps from one position to another is determined by a parameter known as “strides.”
+
+Strides dictate the amount by which the kernel shifts its position as it scans the input data. Specifically, strides control both the horizontal and vertical movement of the kernel during the convolution operation.
+
+Larger stepsizes yield a correspondingly smaller output. In the picture below filtering with stepsize of $ s = 2 $ is shown below filtering the same input with a stepsize of $ s = 1 $.
+
+Convolution Step Size
+
+Padding
+-------
+
+Padding is usually applied on the input image by adding additional rows and columns around its border before convolution starts.
+
+The objective is to ensure that the convolution operation considers the pixels at the borders of the input image, preventing information loss and border effects.
+
+The most commonly used padding is zero-padding because of its performance, simplicity, and computational efficiency. The technique involves adding zeros symmetrically around the edges of an input.
+
+```
+# using numpy to create 5x5 matrix, with random values
+
+import numpy as np
+
+# init random seed
+np.random.seed(0)
+
+a = np.random.rand(5, 5)
+# print a nicely
+np.set_printoptions(precision=3, suppress=True)
+print("Original matrix")
+print(a)
+print()
+
+# add 0 to the left and right of the matrix
+b = np.pad(a, pad_width=1, mode='constant', constant_values=0)
+print("After padding, p = 1")
+print(b)
+
+# we can also pad more than one row or column
+c = np.pad(a, pad_width=2, mode='constant', constant_values=0)
+print("After padding, p = 2")
+print(c)
+```
+
+```
+Original matrix
+[[0.549 0.715 0.603 0.545 0.424]
+[0.646 0.438 0.892 0.964 0.383]
+[0.792 0.529 0.568 0.926 0.071]
+[0.087 0.02  0.833 0.778 0.87 ]
+[0.979 0.799 0.461 0.781 0.118]]
+
+After padding
+[[0.    0.    0.    0.    0.    0.    0.   ]
+[0.    0.549 0.715 0.603 0.545 0.424 0.   ]
+[0.    0.646 0.438 0.892 0.964 0.383 0.   ]
+[0.    0.792 0.529 0.568 0.926 0.071 0.   ]
+[0.    0.087 0.02  0.833 0.778 0.87  0.   ]
+[0.    0.979 0.799 0.461 0.781 0.118 0.   ]
+[0.    0.    0.    0.    0.    0.    0.   ]]
+After padding
+[[0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
+[0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
+[0.    0.    0.549 0.715 0.603 0.545 0.424 0.    0.   ]
+[0.    0.    0.646 0.438 0.892 0.964 0.383 0.    0.   ]
+[0.    0.    0.792 0.529 0.568 0.926 0.071 0.    0.   ]
+[0.    0.    0.087 0.02  0.833 0.778 0.87  0.    0.   ]
+[0.    0.    0.979 0.799 0.461 0.781 0.118 0.    0.   ]
+[0.    0.    0.    0.    0.    0.    0.    0.    0.   ]
+[0.    0.    0.    0.    0.    0.    0.    0.    0.   ]]
+```
+
+ACTIVATION FUNCTION
+-------------------
+
+We perform an Activation function after every convolutional layer in the network architecture.
+
+The ReLU activation function is specifically used as a non-linear activation function, as opposed to other non-linear functions such as Sigmoid because it has been empirically observed that CNNs using ReLU are faster to train than their counterparts
+
+How to create kernel matrix?
+----------------------------
+
+Then how can we create a kernel matrix? Is it by hand or is there a way to create it automatically?
+
+Well, it turns out we can define the kernel size, but the kernel matrix itself is learned by the CNN Neural Network. Here’s how it works:
+
+*   Initially the values within the kernel matrix are randomly initialized. These random values do not represent any specific pattern.
+*   During the training process of the CNN, the network learns the optimal values for the kernel matrix by adjusting the values within the kernel to minimize the error.
+*   Once training is complete, the learned kernel matrix is then used for feature extraction during the convolution operation.
+
+Size of the output
+------------------
+
+The size of the output feature map is controlled by **stride** and **padding**.
+
+\\\[ W\_{out} = \\frac{W\_{in} - F + 2P}{S} + 1 \\\]
+
+Back to top
\ No newline at end of file
diff --git a/image-processing/05.POOLING LAYER.md b/image-processing/05.POOLING LAYER.md
new file mode 100644
index 0000000..edd119f
--- /dev/null
+++ b/image-processing/05.POOLING LAYER.md	
@@ -0,0 +1,94 @@
+Title: POOLING LAYER – Mastering AI Bootcamp 
+
+The feature maps generated by the convolutional layer are subsequently forwarded to the pooling layer. Pooling layers play a crucial role in reducing the spatial dimensions (width and height) of these feature maps, a process commonly referred to as downsampling. This reduction in dimensionality is essential for controlling computational complexity, creating translation invariance, and emphasizing important local features within the feature maps.
+
+The image below shows a single sequence of convolution and pooling.
+
+max pooling
+
+Similar to the convolutional layer, the pooling operation sweeps a filter across the entire input, but the difference is that this filter does not have any weights. Instead, the kernel applies an aggregation function to the values within the receptive field, populating the output array. Also normally the kernel does not overlap in a pooling layer.
+
+Main type of Pooling
+--------------------
+
+There are two main types of pooling: - Max pooling: As the filter moves across the input, it selects the pixel with the maximum value to send to the output array. As an aside, this approach tends to be used more often compared to average pooling. - Average pooling: As the filter moves across the input, it calculates the average value within the receptive field to send to the output array.
+
+The most popular process is max pooling, which reports the maximum output from the neighborhood.
+
+Figure 4: Pooling Operation (Source: O’Reilly Media)
+
+```
+!pip install numpy matplotlib
+```
+
+```
+import numpy as np
+import matplotlib.pyplot as plt
+
+# Create input data (2D array)
+input_data = np.array([[1, 2, 4, 3],
+[6, 2, 9, 8],
+[7, 1, 5, 4],
+[3, 2, 1, 0]])
+
+# Define pooling function (e.g., max pooling)
+def max_pooling(input_array, pool_size):
+output_shape = (input_array.shape[0] // pool_size, input_array.shape[1] // pool_size)
+output_array = np.zeros(output_shape)
+
+for i in range(0, input_array.shape[0], pool_size):
+for j in range(0, input_array.shape[1], pool_size):
+pool_region = input_array[i:i+pool_size, j:j+pool_size]
+output_array[i//pool_size, j//pool_size] = np.max(pool_region)
+
+return output_array
+
+# Apply pooling
+pool_size = 2
+output_data = max_pooling(input_data, pool_size)
+
+# Create subplots for input and output data
+fig, axs = plt.subplots(1, 2, figsize=(10, 5))
+
+# Define custom colors for boxes
+colors = ['#FFFFFF', '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF', '#FFD700', '#FFA500', '#0088FF']
+
+# Plot input data with numbers and boxes using a custom colormap
+axs[0].imshow(np.zeros_like(input_data), cmap='gray', interpolation='nearest')
+for i in range(input_data.shape[0]):
+for j in range(input_data.shape[1]):
+num = int(input_data[i, j])  # Convert to integer
+color = colors[num] if num < len(colors) else 'white'
+rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, color=color)
+axs[0].add_patch(rect)
+axs[0].text(j, i, str(num), va='center', ha='center', color='black', fontsize=12)
+axs[0].set_title('Input Data')
+axs[0].axis('off')
+
+# Plot output data with numbers and boxes using a custom colormap
+axs[1].imshow(np.zeros_like(output_data), cmap='gray', interpolation='nearest')
+for i in range(output_data.shape[0]):
+for j in range(output_data.shape[1]):
+num = int(output_data[i, j])  # Convert to integer
+color = colors[num] if num < len(colors) else 'white'
+rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, color=color)
+axs[1].add_patch(rect)
+axs[1].text(j, i, str(num), va='center', ha='center', color='black', fontsize=12)
+axs[1].set_title('Pooled Data')
+axs[1].axis('off')
+
+plt.show()
+```
+
+Pooling Layer Output Volume
+---------------------------
+
+If we have an activation map of size W x W x D, a pooling kernel of spatial size F, and stride S, then the size of output volume can be determined by the following formula:
+
+\\\[ W\_{out} = \\frac{W - F}{S} + 1 \\\]
+
+This will yield an output volume of size \\\[W\_{out} \\times W\_{out} \\times D\\\]
+
+In all cases, pooling provides some translation invariance which means that an object would be recognizable regardless of where it appears on the frame.
+
+Back to top
\ No newline at end of file
diff --git a/image-processing/06.Fully Connected Layer.md b/image-processing/06.Fully Connected Layer.md
new file mode 100644
index 0000000..ee6f1e5
--- /dev/null
+++ b/image-processing/06.Fully Connected Layer.md	
@@ -0,0 +1,20 @@
+Title: Fully Connected Layer – Mastering AI Bootcamp 
+
+Fully Connected Layer
+---------------------
+
+The result coming out of the Convolution/Pooling layer combination is still in the form of features. To classify or come to a conclusion we need to take into account all the data or feature that we have collected so far and check all the possible combination. This is the job of the Fully Connected layer, which is basically our regular Neural Network that we learn prior to CNN, where all the nodes are connected to nodes from the previous layer.
+
+As shown in the picture below, the output of the last pooling layer is serialized before it is fed into a fully connected layer. In this example only one fully connected layer is applied. Since there are 8 neurons in the output of the fully connected layer, this example architecture can be applied for a classification into 8 classes. In this case the output is usually processed by a softmax-activation function, which is not depicted in the image below.
+
+Note that the fully connected layer only accept 1 Dimensional data. To convert our 3D data to 1D, we can use the function flatten in Python. This essentially arranges our 3D volume into a 1D vector.
+
+FC Layer
+
+### Softmax
+
+A softmax operation serves a key purpose: making sure the CNN outputs sum to 1. Because of this, softmax operations are useful to scale model outputs into probabilities with a range from 0 to 1.
+
+This topic has been covered earlier, so if you need to refresh, then please visit the appropriate course.
+
+Back to top
\ No newline at end of file
diff --git a/image-processing/07.Training.md b/image-processing/07.Training.md
new file mode 100644
index 0000000..3afebe1
--- /dev/null
+++ b/image-processing/07.Training.md
@@ -0,0 +1,234 @@
+Title: Training – Mastering AI Bootcamp 
+
+Let’s try our hand in training a CNN
+
+```
+!pip install torch torchvision scipy matplotlib numpy
+```
+
+```
+import torch
+import torch.nn as nn
+import torch.optim as optim
+from torchvision import datasets, transforms
+from torch.utils.data import DataLoader
+
+# Define a CNN model for MNIST
+class CNNModel(nn.Module):
+def __init__(self):
+super(CNNModel, self).__init__()
+self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
+self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
+self.fc1 = nn.Linear(64 * 5 * 5, 128)
+self.fc2 = nn.Linear(128, 10)
+
+def forward(self, x):
+x = torch.relu(self.conv1(x))
+x = torch.max_pool2d(x, 2)
+x = torch.relu(self.conv2(x))
+x = torch.max_pool2d(x, 2)
+x = x.view(x.size(0), -1)
+x = torch.relu(self.fc1(x))
+x = self.fc2(x)
+return x
+
+# Define data transformations
+transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
+
+# Load MNIST data
+mnist_train = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
+mnist_test = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
+
+# Use Data Loader
+train_loader = DataLoader(mnist_train, batch_size=100, shuffle=True)
+test_loader = DataLoader(mnist_test, batch_size=100, shuffle=False)
+
+# Instantiate the CNN model
+cnn_model = CNNModel()
+
+# Define loss function and optimizer
+loss_fn = nn.CrossEntropyLoss()
+learning_rate = 0.01
+optimizer = optim.SGD(cnn_model.parameters(), lr=learning_rate)
+
+# Define accuracy function
+def accuracy(outputs, labels):
+_, preds = torch.max(outputs, dim=1)
+return torch.tensor(torch.sum(preds == labels).item() / len(preds))
+
+# Training loop
+total_epochs = 5
+for epoch in range(total_epochs):
+for images, labels in train_loader:
+outputs = cnn_model(images)
+loss = loss_fn(outputs, labels)
+
+optimizer.zero_grad()
+loss.backward()
+optimizer.step()
+
+print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, total_epochs, loss.item()))
+
+# Evaluation
+#cnn_model.eval()
+with torch.no_grad():
+accum_acc = 0
+for images, labels in test_loader:
+outputs = cnn_model(images)
+
+loss = loss_fn(outputs, labels)
+acc = accuracy(outputs, labels)
+accum_acc += acc
+
+print('Test loss: {:.4f}, Test accuracy: {:.4f}'.format(loss.item(), accum_acc/len(test_loader)))
+```
+
+```
+Epoch [1/5], Loss: 0.7850
+Epoch [2/5], Loss: 0.4941
+Epoch [3/5], Loss: 0.4238
+Epoch [4/5], Loss: 0.4913
+Epoch [5/5], Loss: 0.4813
+Test loss: 0.4732, Test accuracy: 0.8098
+```
+
+Data Augmentation
+-----------------
+
+Can we train a CNN mdoel on a relatively small dataset ? What happens if the dataset is small ?
+
+It is possible to train on a small dataset, and quite accurate too. However there is one major problem, if the input image differs, for example, it’s upside down, the model will fail. This is known as overfitting. Overfitting occurs when a model learns to perform well on the training data but fails to generalize to unseen data.
+
+To overcome this issue, we can use data augmentation. What is Data augmentation ?
+
+Basically we artificially increase the size and diversity of the training dataset. We can do this by: - Rotation: Data augmentation can involve rotating the digit images by various angles. This helps the model learn to recognize digits even if they are slightly tilted or rotated when written by different people. - Scaling and Shearing: You can apply transformations that stretch or compress the digit images in both the x and y directions. This allows the model to handle variations in digit size and aspect ratio. - Translation: Shifting the digit images within the image frame helps the model learn to recognize digits in different positions on the input image. - Noise: Adding random noise to the images simulates variations in writing style and drawing imperfections.
+
+Let’s assume we want to make sure that make sure that our CNN model based on the MNIST dataset to recognize digits written by various individuals with different writing styles. Here’s what we could do:
+
+```
+import torch
+import torchvision
+import torchvision.transforms as transforms
+import matplotlib.pyplot as plt
+from PIL import Image
+import numpy as np
+
+transform = transforms.Compose([transforms.ToTensor()])
+train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
+
+# Example data augmentation transformations
+data_augmentation = transforms.Compose([
+transforms.RandomRotation(degrees=(-10, 10), fill=0),  # Fill with black for rotation
+transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
+transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
+transforms.RandomResizedCrop(size=(28, 28), scale=(0.8, 1.2)),
+transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
+transforms.RandomErasing(p=0.5),
+])
+
+# Create a custom dataset class to store augmented data
+class AugmentedDataset(torch.utils.data.Dataset):
+def __init__(self, original_dataset, data_augmentation):
+self.original_dataset = original_dataset
+self.data_augmentation = data_augmentation
+
+def __len__(self):
+return len(self.original_dataset)
+
+def __getitem__(self, idx):
+image, label = self.original_dataset[idx]
+augmented_image = self.data_augmentation(image)
+return augmented_image, label
+
+# Create an augmented dataset
+augmented_dataset = AugmentedDataset(train_dataset, data_augmentation)
+
+# Choose a digit class (e.g., digit 7)
+digit_class = 7
+
+# Filter the dataset to get images of the chosen class
+digit_images = [image for image, label in train_dataset if label == digit_class]
+
+# Apply data augmentation to the images and convert to PIL Images
+augmented_images_pil = [transforms.ToPILImage()(data_augmentation(image)) for image in digit_images]
+
+# Convert PIL Images to NumPy arrays before visualization
+original_images_np = [image.squeeze().numpy() for image in digit_images]
+augmented_images_np = [np.array(image) for image in augmented_images_pil]
+
+# Visualize original and augmented images
+plt.figure(figsize=(12, 6))
+
+for i in range(5):
+plt.subplot(2, 5, i + 1)
+plt.imshow(original_images_np[i], cmap='gray')
+plt.title("Original")
+
+for i in range(5):
+plt.subplot(2, 5, i + 6)
+plt.imshow(augmented_images_np[i], cmap='gray')
+plt.title("Augmented")
+
+plt.show()
+```
+
+How do we combine them ? We can use ConcatDataset
+
+```
+from torch.utils.data import ConcatDataset
+
+# Assuming you have a dataset object, e.g., mnist_train
+num_images = len(train_dataset)
+print("Number of images in the dataset (before):", num_images)
+
+# Combine the original and augmented datasets
+combined_dataset = ConcatDataset([train_dataset, augmented_dataset])
+
+# Create a DataLoader for the combined dataset
+combined_train_loader = DataLoader(combined_dataset, batch_size=100, shuffle=True)
+
+# Assuming you have a dataset object, e.g., mnist_train
+num_images = len(combined_dataset)
+print("Number of images in the dataset (after):", num_images)
+```
+
+```
+Number of images in the dataset (before): 60000
+Number of images in the dataset (after): 120000
+```
+
+Next we can train them as usual. Pretty neat, eh ?
+
+Exercise CNN Training
+---------------------
+
+```
+!pip install rggrader
+```
+
+```
+# @title #### Student Identity
+student_id = "student_id" # @param {type:"string"}
+name = "your_name" # @param {type:"string"}
+```
+
+```
+# @title #### 00. CNN Model using SHVN Dataset
+
+from rggrader import submit
+
+# TODO: Train a model on your own, using the SVHN (Street View House Numbers) dataset :: https://huggingface.co/datasets/svhn
+
+# You may add any code here to derive your variables
+# Please change this
+accuracy = 0
+
+print(f"The accuracy is {accuracy}")
+
+# Submit Method
+assignment_id = "03_cnn"
+question_id = "01_training_svhn"
+submit(student_id, name, assignment_id, str(accuracy), question_id, "")
+```
+
+Back to top
\ No newline at end of file
diff --git a/image-processing/08.Pretrained CNN Model.md b/image-processing/08.Pretrained CNN Model.md
new file mode 100644
index 0000000..e7b73a5
--- /dev/null
+++ b/image-processing/08.Pretrained CNN Model.md	
@@ -0,0 +1,365 @@
+Title: Pretrained CNN Model – Mastering AI Bootcamp 
+
+Pretrained Model
+----------------
+
+So far we have been creating our own CNN model. Are there pretrained CNN model ?
+
+Yes, there are, quite a lot, some of the most popular are: - **ResNet**: Residual Networks (ResNets) are known for their effectiveness in deep learning tasks. Models like ResNet-50, ResNet-101, and ResNet-152 are available, and you can fine-tune them on your specific high-res dataset.
+
+*   **InceptionV3**: The Inception architecture, particularly InceptionV3, is designed to capture intricate patterns in images and is suitable for high-res images.
+
+*   **VGG16** and **VGG19**: The VGG architecture, specifically VGG16 and VGG19, consists of multiple convolutional layers and is effective for image classification tasks.
+
+Let’s start with one of the easiest: AlexNet
+
+AlexNet
+-------
+
+```
+import torchvision.models as models
+
+# weights = None means that we don't want to load the weights of the model, only the architecture
+alexnet = models.alexnet(weights=None)
+print(alexnet)
+```
+
+```
+AlexNet(
+(features): Sequential(
+(0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
+(1): ReLU(inplace=True)
+(2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
+(3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
+(4): ReLU(inplace=True)
+(5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
+(6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
+(7): ReLU(inplace=True)
+(8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
+(9): ReLU(inplace=True)
+(10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
+(11): ReLU(inplace=True)
+(12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
+)
+(avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
+(classifier): Sequential(
+(0): Dropout(p=0.5, inplace=False)
+(1): Linear(in_features=9216, out_features=4096, bias=True)
+(2): ReLU(inplace=True)
+(3): Dropout(p=0.5, inplace=False)
+(4): Linear(in_features=4096, out_features=4096, bias=True)
+(5): ReLU(inplace=True)
+(6): Linear(in_features=4096, out_features=1000, bias=True)
+)
+)
+```
+
+Please spend some time to understand the output above
+
+It’s just a simple layers of Convolutional, MaxPooling, and Linear layers. The exact same Conv2D and MaxPool2D layers that we have been using so far. You also have learned kernal size, stride, padding, etc.
+
+Despite its simplicity, AlexNet is monumental! It’s the winner of the 2012 ImageNet Large Scale Visual Recognition Competition (ILSVRC) beating the second place with huge gap: The network achieved a top-5 error of 15.3%, more than **10.8%** points lower than that of the runner up.
+
+VGG16
+-----
+
+```
+import torchvision.models as models
+
+vgg16 = models.vgg16(weights=None)
+print(vgg16)
+```
+
+```
+VGG(
+(features): Sequential(
+(0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
+(1): ReLU(inplace=True)
+(2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
+(3): ReLU(inplace=True)
+(4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
+(5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
+(6): ReLU(inplace=True)
+(7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
+(8): ReLU(inplace=True)
+(9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
+(10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
+(11): ReLU(inplace=True)
+(12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
+(13): ReLU(inplace=True)
+(14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
+(15): ReLU(inplace=True)
+(16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
+(17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
+(18): ReLU(inplace=True)
+(19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
+(20): ReLU(inplace=True)
+(21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
+(22): ReLU(inplace=True)
+(23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
+(24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
+(25): ReLU(inplace=True)
+(26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
+(27): ReLU(inplace=True)
+(28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
+(29): ReLU(inplace=True)
+(30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
+)
+(avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
+(classifier): Sequential(
+(0): Linear(in_features=25088, out_features=4096, bias=True)
+(1): ReLU(inplace=True)
+(2): Dropout(p=0.5, inplace=False)
+(3): Linear(in_features=4096, out_features=4096, bias=True)
+(4): ReLU(inplace=True)
+(5): Dropout(p=0.5, inplace=False)
+(6): Linear(in_features=4096, out_features=1000, bias=True)
+)
+)
+```
+
+You should be able to understand the output above, only Conv2d, MaxPool2d, and Linear layers!
+
+ResNet
+------
+
+ResNet is a very deep CNN model, it has 152 layers! (compare it to AlexNet which only has 8 layers)
+
+It’s the winner of the 2015 ImageNet competition
+
+resnet
+
+### Residual Network
+
+The special thing about ResNet is the skip connection, which is the addition of the input to the output of the stacked layers.
+
+The output of layer \\(i\\)\-th is passed to layer \\(i+2\\)\-th, this is called **skip connection**.
+
+\\\[ a^{\[l+2\]} = g(z^{\[l+2\]} + a^{\[l\]}) \\\]
+
+ResBlock
+
+The problem with deep neural network is that it’s hard to train, the gradient vanishing problem, i.e. the gradient becomes smaller and smaller as it goes deeper and deeper resulting in the weights of the earlier layers don’t get updated much.
+
+The skip connection solves this problem by passing the output of the earlier layers to the deeper layers, so the gradient doesn’t vanish.
+
+We can implement the skip connection in PyTorch by adding the input to the output of the layer (before `ReLU`):
+
+```
+import torch
+from torch import nn
+
+class ResidualBlock(nn.Module):
+def __init__(self, in_channels, out_channels, stride=1):
+super(ResidualBlock, self).__init__()
+
+# Convolutional layers 
+self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
+stride=stride, padding=1, bias=False)
+self.bn1 = nn.BatchNorm2d(out_channels)
+self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
+stride=1, padding=1, bias=False)
+self.bn2 = nn.BatchNorm2d(out_channels)
+
+# Skip connection
+self.skip = nn.Sequential()
+if stride != 1 or in_channels != out_channels:
+self.skip = nn.Sequential(
+nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
+nn.BatchNorm2d(out_channels)
+)
+
+def forward(self, x):
+out = torch.relu(self.bn1(self.conv1(x)))
+out = self.bn2(self.conv2(out))
+out += self.skip(x)  # Skip connection
+out = torch.relu(out)
+return out
+```
+
+Above is the implementation of ResNet, you can see the skip connection in the `forward` method.
+
+Put attention to this part:
+
+```
+out += self.skip(x)
+```
+
+`x` is the original input. It’s **not** the output from the previous layer:
+
+```
+out = self.skip(out)
+```
+
+Fine tuning
+-----------
+
+Before we go into Applied CNN next, let’s do a quick fine tuning/transfer learning on CNN. We’ll use Resnet18 and the CIFAR10 dataset. ResNet-18 is a convolutional neural network that is 18 layers deep. And it’s pretrained on more than a million images from the ImageNet database.
+
+For the last step we will save our trained model and use it without training it every time, let’s see how that’s done:
+
+```
+import torchvision.datasets as datasets
+import torchvision.transforms as transforms
+
+# Define the transforms
+transform = transforms.Compose([
+transforms.ToTensor(),
+transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
+])
+
+# Download the dataset
+train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
+test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
+
+import torch
+import torch.nn as nn
+import torch.optim as optim
+from torchvision.models import resnet18
+
+# Define the model
+model = resnet18(weights='DEFAULT')
+num_ftrs = model.fc.in_features
+model.fc = nn.Linear(num_ftrs, 1000)
+
+# Define the loss function and optimizer
+criterion = nn.CrossEntropyLoss()
+optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
+
+from torch.utils.data import DataLoader
+
+# Define the batch size and number of epochs
+batch_size = 32
+num_epochs = 1
+
+# Create the data loaders
+train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
+test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
+
+# Train the model
+for epoch in range(num_epochs):
+running_loss = 0.0
+for i, (images, labels) in enumerate(train_loader):
+# Zero the gradients
+optimizer.zero_grad()
+
+# Forward pass
+outputs = model(images)
+
+# Calculate the loss
+loss = criterion(outputs, labels)
+
+# Backward pass
+loss.backward()
+
+# Update the weights
+optimizer.step()
+
+# Print statistics
+running_loss += loss.item()
+if (i + 1) % 100 == 0:  # Print every 100 mini-batches
+print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
+running_loss = 0.0
+
+# Test the model
+correct = 0
+total = 0
+with torch.no_grad():
+for images, labels in test_loader:
+outputs = model(images)
+_, predicted = torch.max(outputs.data, 1)
+total += labels.size(0)
+correct += (predicted == labels).sum().item()
+
+# Print the accuracy
+print('Epoch [{}/{}], Test Accuracy: {:.2f}%'.format(epoch+1, num_epochs, 100*correct/total))
+
+# Save the trained model to a file
+torch.save(model.state_dict(), 'resnet18_cifar10_classifier.pth')
+print('Model saved')
+```
+
+```
+Files already downloaded and verified
+Files already downloaded and verified
+Epoch [1/1], Batch [100/1563], Loss: 2.4852
+Epoch [1/1], Batch [200/1563], Loss: 1.3526
+Epoch [1/1], Batch [300/1563], Loss: 1.2018
+Epoch [1/1], Batch [400/1563], Loss: 1.0929
+Epoch [1/1], Batch [500/1563], Loss: 1.0548
+Epoch [1/1], Batch [600/1563], Loss: 1.0232
+Epoch [1/1], Batch [700/1563], Loss: 0.9527
+Epoch [1/1], Batch [800/1563], Loss: 0.9520
+Epoch [1/1], Batch [900/1563], Loss: 0.9351
+Epoch [1/1], Batch [1000/1563], Loss: 0.8726
+Epoch [1/1], Batch [1100/1563], Loss: 0.8719
+Epoch [1/1], Batch [1200/1563], Loss: 0.8006
+Epoch [1/1], Batch [1300/1563], Loss: 0.8739
+Epoch [1/1], Batch [1400/1563], Loss: 0.8623
+Epoch [1/1], Batch [1500/1563], Loss: 0.7862
+Epoch [1/1], Test Accuracy: 72.49%
+Model saved
+```
+
+Let’s try out our model with an image from the internet:
+
+```
+import torch
+import torchvision.models as models
+import requests
+from PIL import Image
+from torchvision import transforms
+import io
+
+# Load the model architecture (e.g., ResNet18)
+model = models.resnet18(weights=None)
+
+# Load the saved model weights
+model_weights_path = 'resnet18_cifar10_classifier.pth'  # Path to the saved model weights on your local disk
+model.load_state_dict(torch.load(model_weights_path))
+
+# Put the model in evaluation mode
+model.eval()
+
+# URL of the image you want to classify
+#image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/200125-N-LH674-1073_USS_Theodore_Roosevelt_%28CVN-71%29.jpg/1200px-200125-N-LH674-1073_USS_Theodore_Roosevelt_%28CVN-71%29.jpg'  # aircraft carrier
+image_url = 'https://akm-img-a-in.tosshub.com/businesstoday/images/assets/202307/16-17-1-sixteen_nine.jpg?size=948:533' #airplane
+
+# Download the image from the URL
+response = requests.get(image_url)
+
+if response.status_code == 200:
+# Open the downloaded image with PIL
+img = Image.open(io.BytesIO(response.content))
+else:
+print('Failed to download the image.')
+
+# Apply the same transformations used during training (resize, normalize, etc.)
+transform = transforms.Compose([
+transforms.Resize((32, 32)),  # Resize to match model's input size, CIFAR10 was 32x32, try commenting this and see if the prediction is still correct
+transforms.ToTensor(),
+transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
+])
+
+input_data = transform(img).unsqueeze(0)  # Add a batch dimension
+
+#We use CIFAR10 labels
+class_to_label = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
+
+with torch.no_grad():
+output = model(input_data)
+
+# Get the predicted class label
+_, predicted_class = torch.max(output, 1)
+
+# Print the predicted class
+print('Predicted Class:', predicted_class.item())
+print('Predicted Label:', class_to_label[predicted_class.item()])
+```
+
+```
+Predicted Class: 0
+Predicted Label: airplane
+```
+
+Back to top
\ No newline at end of file
diff --git a/image-processing/09.Applied CNN: Object Detection and YOLO in Action.md b/image-processing/09.Applied CNN: Object Detection and YOLO in Action.md
new file mode 100644
index 0000000..a15f0a9
--- /dev/null
+++ b/image-processing/09.Applied CNN: Object Detection and YOLO in Action.md	
@@ -0,0 +1,270 @@
+Title: Applied CNN: Object Detection and YOLO in Action – Mastering AI Bootcamp 
+
+Object Detection
+----------------
+
+In the real world, an image can have multiple objects, our previous CNN model can only detect one object. In this section, we will learn how to detect multiple objects in an image.
+
+Image Classification vs Object Detection Example
+
+Image source
+
+Object detection involves identifying and locating objects of interest within an image or a video. Above, in the left image, truck and car are identified, which is image classification. The image on the right illustrates object detection by surrounding the members of each class — cars and trucks — with a bounding box and showing the correct labels.
+
+### Object localization
+
+The bounding box is called object localization. We already learn about the image classification task where an algorithm looks a picture and gives an output saying this is a car. The problem of classification with localization is not only do you have to label this as, say, a car but the algorithm also is responsible for put a bounding box or drawing a red rectangle around the position of the car in the image, where the term localization refers to figuring out where in the picture is the car you detected.
+
+Let’s try the following image:
+
+cars
+
+To add bounding box, we can use the following CLI command (don’t worry, we will learn how to do it in Python later):
+
+```
+!yolo predict model=yolov8s.pt source="./assets/cars.jpeg"
+```
+
+Voila! We have successfully added bounding boxes to the cars in the image.
+
+cars\_with\_bounding\_box
+
+We can say here that the bounding box is basically a set of coordinates that forms around the object that identify closely with a label.
+
+We also know that if we we define the model to be able to detect 10 classes, we’ll have an output containing the 10 classes with percentage of each class.
+
+If we add them up together, then, for object localization, the output will not only contain class probabilities for the detected objects but also bounding box coordinates that specify the location of each detected object within the image. This allows the model to not only classify objects into predefined classes but also localize and outline their positions using bounding boxes.
+
+The question is how we do that ?
+
+### Building Bounding Box
+
+In the previous section, we have learned how to build a CNN model for image classification.
+
+In the case of image classification, the model output is
+
+\\\[ \\hat{y} = \\begin{bmatrix} p\_1 \\\\ p\_2 \\\\ p\_3 \\end{bmatrix} \\\]
+
+where \\(\\hat{y}\\) is the predicted class probabilities for the three classes. Then we can use the argmax function to get the class with the highest probability.
+
+However, in the context of object detection, we need to add bounding box coordinates to the output. We need to add four more elements to the output vector to represent the bounding box coordinates.
+
+\\\[ \\hat{y} = \\begin{bmatrix} x \\\\ y \\\\ w \\\\ h \\\\ p\_1 \\\\ p\_2 \\\\ p\_3 \\end{bmatrix} \\\]
+
+Where \\(x\\) and \\(y\\) are the coordinates of the center of the bounding box, \\(w\\) and \\(h\\) are the width and height of the bounding box
+
+### Empty Bounding Box
+
+But what if there is no object in the image?
+
+To represent empty box, we can add a new output element to indicate the probability of the image containing an object.
+
+\\\[ \\hat{y} = \\begin{bmatrix} confidence \\\\ x \\\\ y \\\\ w \\\\ h \\\\ p\_1 \\\\ p\_2 \\\\ p\_3 \\end{bmatrix} \\\]
+
+Where \\(confidence\\) is the probability of the image containing an object.
+
+Note that it differs from \\(p\_1\\), \\(p\_2\\), and \\(p\_3\\), which are the probabilities of the image containing a car, a truck, and a motorcycle, respectively.
+
+Example:
+
+\\\[ \\hat{y} = \\begin{bmatrix} 0.9 \\\\ 200 \\\\ 300 \\\\ 20 \\\\ 30 \\\\ 0.1 \\\\ 0.1 \\\\ 0.8 \\end{bmatrix} \\\]
+
+It means that the model predict 0.9 probability that the image contains an object, and the object is a car with 0.1 probability, a truck with 0.1 probability, and a motorcycle with 0.8 probability. The bounding box is located at (200, 300) with width 20 and height 30.
+
+Another example:
+
+\\\[ \\hat{y} = \\begin{bmatrix} 0 \\\\ 0 \\\\ 0 \\\\ 0 \\\\ 0 \\\\ 0.1 \\\\ 0.1 \\\\ 0.8 \\end{bmatrix} \\\]
+
+The model predicts 0 probability that the image contains an object. We can forget the remaining elements because they are not relevant.
+
+### Sliding Window
+
+Now that we have found a way to define a bounding box in our model, next let’s think about how we’ll implement object detection.
+
+From our own logic, once we define a bounding box, the easiest and fastest way to detect the object is to simply test the box on the image, much like how a convolution or pooling works in CNN, except we don’t do any calculation, but we take the image inside the box and check if the object exists inside the box. Then move on to the next pixel. Let’s call this a sliding window.
+
+After 1 pass through the image and we did not find anything, we then change the size of the bounding box and try again. Rinse and repeat.
+
+Sliding Window
+
+Image source
+
+The above algorithm surely is not efficient. But there is a way to make it more efficient.
+
+Remember the convolution operation in CNN? It is basically a sliding window with a kernel. The kernel is a matrix of weights that we multiply with the image pixels. The kernel is then **moved** to the next pixel and the process is repeated.
+
+It turns out that we can implement the sliding window algorithm using convolution. However, the detail is beyond the scope of this course.
+
+YOLO Algorithm
+--------------
+
+Yolo animation
+
+Image source
+
+YOLO, short for “You Only Look Once,” is a popular object detection algorithm known for its efficiency and real-time performance. It excels at swiftly detecting and localizing objects within images or video frames, making it valuable for various applications, including autonomous driving, surveillance, and computer vision tasks. YOLO’s fame stems from its unique approach of dividing images into grids and simultaneously predicting bounding boxes and class probabilities within each grid cell. This one-pass design makes it exceptionally fast and suitable for real-time tasks. While YOLO is versatile and widely used, it may not be ideal for scenarios involving small objects, extreme size variations, or very high localization precision requirements, and it typically demands substantial training data for optimal performance.
+
+### How does Yolo works ?
+
+Souce: MLForNerds
+
+This is based on YOLOv1, the original YOLO paper.
+
+#### Smaller Grid
+
+Yolo Step 1
+
+First the original image is divided into $ S x S $ grid cells of equal size. Each cell in the grid will predict the class of the object inside the cell, if there is one.
+
+#### Bounding Box Definition
+
+Yolo Step 2
+
+Inside the cell, we will predict the bounding box coordinates. One cell predicts 2 bounding boxes. The bounding box is defined by the center of the box, width, and height.
+
+So in the above picture, the red dot is the center of the red box, the blue dot is the center of the blue box
+
+#### Bounding Box Encoding
+
+The bounding box is encoded as follow:
+
+\\(x\\) and \\(y\\) are calculated relative to the cell top-left corner (anchor) and normalized by the cell width and height.
+
+\\\[ x = \\frac{x\_{center} - x\_{anchor}}{cell\\\_width} \\\\ y = \\frac{y\_{center} - y\_{anchor}}{cell\\\_height} \\\]
+
+and \\(w\\) and \\(h\\) are calculated relative to the whole image width and height.
+
+\\\[ w = \\frac{w\_{box}}{image\\\_width} \\\\ h = \\frac{h\_{box}}{image\\\_height} \\\]
+
+#### Data representation
+
+In previous section, we model the data as follow:
+
+\\\[ \\hat{y} = \\begin{bmatrix} confidence \\\\ x \\\\ y \\\\ w \\\\ h \\\\ p\_1 \\\\ p\_2 \\\\ p\_3 \\end{bmatrix} \\\]
+
+In YOLO - since we have multiple grids - we model the data as follow:
+
+\\\[ \\hat{y} = \\begin{bmatrix} confidence \\\\ x \\\\ y \\\\ w \\\\ h \\\\ confidence \\\\ x \\\\ y \\\\ w \\\\ h \\\\ ... \\\\ p\_1 \\\\ p\_2 \\\\ p\_3 \\end{bmatrix} \\\]
+
+So we repeat the confidence, x, y, w, and h for each grid.
+
+Yolo Step 3
+
+#### Combining the output
+
+Yolo Step 4
+
+#### Intersection over Union (IoU)
+
+Yolo howto 4
+
+Naturally the bounding box will overlap with more than one grid cells. The goal of IoU is to keep only those that are relevant to the image.
+
+Here is how we do it logically:
+
+*   The user defines its IOU selection threshold, which can be, for instance, 0.5.
+*   Then YOLO computes the IOU of each grid cell which is the Intersection area divided by the Union Area.
+*   Finally, it ignores the prediction of the grid cells having an IOU ≤ threshold and considers those with an IOU > threshold.
+
+### YOLO in action
+
+```
+!pip install ultralytics
+```
+
+```
+import torch
+from pathlib import Path
+from PIL import Image
+
+# Load YOLOv5 model (you may need to adjust the model path)
+model_path = 'yolov5s.pt'
+model = torch.hub.load('ultralytics/yolov5:master', 'yolov5s')
+
+# Load an input image
+image_path = './assets/yolo-input.jpg'
+img = Image.open(image_path)
+
+# Perform object detection
+results = model(img)
+
+# Print detected classes and confidence scores
+#print(results.names)
+#print(results.pred[0][:, -1].tolist())  # Confidence scores
+
+# Show the annotated image with bounding boxes
+results.show()
+```
+
+```
+Using cache found in /Users/ruangguru/.cache/torch/hub/ultralytics_yolov5_master
+requirements: Ultralytics requirement ['setuptools>=65.5.1'] not found, attempting AutoUpdate...
+Requirement already satisfied: setuptools>=65.5.1 in /Users/ruangguru/Projects/ai-bootcamp/env/lib/python3.11/site-packages (68.2.2)
+
+requirements: AutoUpdate success ✅ 2.6s, installed 1 package: ['setuptools>=65.5.1']
+requirements: ⚠️ Restart runtime or rerun command for updates to take effect
+
+YOLOv5 🚀 2023-9-14 Python-3.11.3 torch-2.0.1 CPU
+
+Fusing layers... 
+YOLOv5s summary: 213 layers, 7225885 parameters, 0 gradients
+Adding AutoShape... 
+```
+
+Realtime Object Detection
+-------------------------
+
+Let’s use YOLO to detect objects in a video.
+
+```
+!pip install opencv-python
+```
+
+```
+# use YOLO v5 to detect objects in a video via webcam
+
+import cv2
+import torch
+from pathlib import Path
+from PIL import Image
+
+model_path = 'yolov5s.pt'
+model = torch.hub.load('ultralytics/yolov5:master', 'yolov5s')
+
+# use webcam as the video source
+cap = cv2.VideoCapture(0)
+
+while True:
+# Capture frame-by-frame
+ret, frame = cap.read()
+
+# Convert the frame to a PIL Image
+pil_img = Image.fromarray(frame)
+
+# Perform object detection
+results = model(pil_img)
+
+# Print the detected classes and confidence scores
+#print(results.names)
+#print(results.pred[0][:, -1].tolist())  # Confidence scores
+
+# Show the annotated image with bounding boxes
+results.show()
+
+# Display the resulting frame
+#cv2.imshow('frame', frame)
+
+# Press Q on keyboard to  exit
+if cv2.waitKey(1) & 0xFF == ord('q'):
+break
+
+print("continue? [Y/n]")
+if input() == 'n':
+break
+
+# When everything done, release the capture
+cap.release()
+cv2.destroyAllWindows()
+```
+
+Back to top
\ No newline at end of file
