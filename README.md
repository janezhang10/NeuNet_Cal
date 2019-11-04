Hard-coding neural networks is a hefty task for ML Engineers, especially when tasked with training a myriad networks in search for the best hyper parameters. We have greatly simplified this task by allowing the user to rapidly prototype and train networks using plain English.

NeuNet is a Python utility that allows the user to easily construct Neural Networks using their voice or just plain text, which is essential for rapid prototyping and hyper parameter optimization. Simply describe the parameters of your model in plain English and NeuNet will build and train the network for you.

Example: “Make me a UNet with input size of 192 pixels and 4 output classes. Use ReLU nonlinearities and train the model on 25 epochs of the buildings dataset.”

NeuNet implements the Google Cloud Speech-to-text API to transform user voice input into string form. The semantic meaning of the string request is then extracted using language processing techniques. NeuNet then uses these parameters to dynamically build and train a corresponding neural network in PyTorch, and displays the model and training metrics in the terminal window.
