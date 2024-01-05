#   Handwritten Digit Recognition

##  Objective
Code to classify handwritten digits with a training accuracy of 99% or above without any fixed number of epochs, using Convolutional Neural Network (CNN).

##  Tech Stack

### Python:
Python is high level, interpreted, interactive and object-oriented scripting language. Python is highly readable using English keywords frequently unlike other languages using punctuations and more syntactical constructions. Python is easy to learn, read and maintain.

It has a broad standard library which is very portable and cross platform compatible on UNIX, Windows, and Macintosh. Python supports interactive testing and debugging of snippets of code. This language is portable and extendable and thus provides interfaces to all major commercial databases. It also supports GUI applications that can be created and ported to many system calls, libraries, and windows systems, such as Windows MFC, Macintosh, and the X Window system of UNIX. It also provides a better structure and support for large programs than shell scripting.

### TensorFlow:
TensorFlow is an end-to-end open-source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML powered applications.

TensorFlow helps in building and training ML models easily using intuitive high-level APIs like Keras with eager execution, which makes for immediate model iteration and easy debugging. It easily trains and deploy models in the cloud, on-prem, in the browser, or on-device no matter what language one use. TensorFlow is a simple and flexible architecture to take new ideas from concept to code, to state-of-the-art models, and to publication faster.

### Matplotlib:
Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy. It provides an object-oriented API for embedding plots into applications using general-purpose GUI toolkits like Tkinter, wxPython, Qt, or GTK+. There is also a procedural "pylab" interface based on a state machine (like OpenGL), designed to closely resemble that of MATLAB, though its use is discouraged. SciPy makes use of Matplotlib. Pyplot is a Matplotlib module which provides a MATLAB-like interface. Matplotlib is designed to be as usable as MATLAB, with the ability to use Python, and the advantage of being free and open-source.

### Google Colaboratory:
Colaboratory, or "Colab" for short, allows you to write and execute Python in your browser, with

-   Zero configuration required
-   Free access to GPUs
-   Easy sharing

Colab can make any student’s, any data scientists or any AI researchers work easier.

##  Theory
There is a dataset called MNIST which has items of handwriting -- the digits 0 through 9.

So, I wrote an MNIST classifier that trains to 99% accuracy or above, and it is trained without a fixed number of epochs -- i.e., it should stop training once it reaches the level of accuracy to 99%.

Some notes:
1.  It succeeds in less than 10 epochs, so it is okay to change epochs to 10, but nothing larger
1.  When it reaches 99% or greater it prints out the string "Reached 99% accuracy so cancelling training!"

##  Algorithm
1.  Start
1.  Import required and/or supporting libraries
1.  Load Dataset
1.  Define data and labels
1.  Reshaping testing and training dataset for convolution and pooling activity by ANN
1.  Define model layers (No. of Layers, layer type, No. of neurons, neuron type, activation type, etc.)
1.  Initialize model hyperparameter values (optimizer, loss type, metrics, etc.)
1.  Training the model (letting the model learn/fit the relationship between data and resp. label, by setting iteration/epoch value)
1.  Check the accuracy value
    1.  If accuracy reaches 99%, then stop training and move to step 9
    1.  Else (If accuracy does not reach 99%, then), move to step 7
1. Testing/evaluating the model
1. Classifying (or predicting) unseen image
1. Finish

##  Conclusion
In this work, we tested variants of a convolutionary neural network to avoid complex pre-processing, costly feature extraction and a complex ensemble classifier combination approach of a conventional recognition method with the goal of implementing the Handwritten Digit Recognition. The current work indicates the function of different hyper-parameters by thorough assessment using an MNIST dataset. We also confirmed that to enhance the efficiency of CNN architecture, fine tuning of hyper-parameters is vital. With the Adam optimizer for the MNIST database, we achieved a recognition rate of 99.89 percent. The impact on the output of handwritten digital recognition of increasing the number of convolutionary layers in CNN architecture is clearly presented through the experiments. The uniqueness of the present work is that all the parameters of the CNN architecture that provide the highest accuracy of recognition for an MNIST dataset are thoroughly examined.

Various CNN architectures, including hybrid CNN, viz., versions of CNN-RNN and CNN-HMM, and domain-specific recognition systems, may be explored in the future. To optimize CNN learning parameters, evolutionary algorithms can be explored, namely the number of layers, learning rate and convolutional philtre kernel sizes.
