In this assignment you will practice writing backpropagation code, and training Neural Networks and Convolutional Neural Networks. The goals of this assignment are as follows:

- understand **Neural Networks** and how they are arranged in layered architectures
- understand and be able to implement (vectorized **backpropagation**
- implement various **update rules** used to optimize Neural Networks
- implement **batch normalization** for training deep networks
- implement **dropout** to regularize networks
- understand the architecture of **Convolutional Neural Networks** and train gain experience with training these models on data

## Setup
You can work on the assignment locally on your own machine. 

As for the dependencies:

**[Option 1] Use Anaconda for Python 2.7:**
The preferred approach for installing all the assignment dependencies is to use [Anaconda](https://www.continuum.io/downloads), which is a Python distribution that includes many of the most popular Python packages for science, math, engineering and data analysis. Once you install it you can skip all mentions of requirements and you are ready to go directly to working on the assignment. The code for this assignment will only work with Python 2.7, so make sure to get the corresponding version of Anaconda.

**[Option 2] Manual install, virtual environment:**
If you do not want to use Anaconda and want to go with a more manual and risky installation route you will likely want to create a [virtual environment](http://docs.python-guide.org/en/latest/dev/virtualenvs/) for the project. If you choose not to use a virtual environment, it is up to you
to make sure that all dependencies for the code are installed globally on your machine. To set up a virtual environment, run the following:

```bash
cd hw1
sudo pip install virtualenv      # This may already be installed
virtualenv .env                  # Create a virtual environment
source .env/bin/activate         # Activate the virtual environment
pip install -r requirements.txt  # Install dependencies
# Work on the assignment for a while ...
deactivate                       # Exit the virtual environment
```

**Download data:**
Once you have the starter code, you will need to download the CIFAR-10 dataset.
Run the following from the `hw1` directory:

```bash
cd code/datasets
./get_datasets.sh
```

[see readme_windows.txt for how to download the data in Windows]

**Compile the Cython extension:** Convolutional Neural Networks require a very efficient implementation. We have implemented the functionality using [Cython](http://cython.org/) [Cython is included in Anaconda]; you will need to compile the Cython extension before you can run the code. From the `code` directory, run the following command:

```bash
python setup.py build_ext --inplace
```
[this command in windows may generate an "Unable to find vcvarsall.bat" error. See http://stackoverflow.com/questions/2817869/error-unable-to-find-vcvarsall-bat/ . What I found to work is installing the Microsoft compiler ( https://www.microsoft.com/en-us/download/details.aspx?id=44266 ) and then using "pip install -e ." instead of the "python setup.py" command]

**Start IPython:**
After you have the CIFAR-10 data, you should start the IPython notebook server from the `hw1` directory [use the command "ipython notebook". You may get a warning that this command is deprecated and jupyter notebook will start instead. That's fine. This command works in Windows]. If you are unfamiliar with IPython, you can read a IPython tutorial at http://cs231n.github.io/ipython-tutorial/. 


**NOTE:** If you are working in a virtual environment on OSX, you may encounter errors with matplotlib due to the [issues described here](http://matplotlib.org/faq/virtualenv_faq.html). You can work around this issue by starting the IPython server using the `start_ipython_osx.sh` script from the `hw1` directory; the script
assumes that your virtual environment is named `.env`.


### Submitting your work:
Once you are done working run the `collectSubmission.sh` script; this will produce a file called `hw1.zip`. Upload this file to T-Square.

We will grade is the four ipython notebooks listed below and the output of each stage. 

### Q1: Fully-connected Neural Network (30 points)
The IPython notebook `FullyConnectedNets.ipynb` will introduce you to our modular layer design, and then use those layers to implement fully-connected networks of arbitrary depth. To optimize these models you will implement several popular update rules.

### Q2: Batch Normalization (30 points)
In the IPython notebook `BatchNormalization.ipynb` you will implement batch normalization, and use it to train deep fully-connected networks.

### Q3: Dropout (10 points)
The IPython notebook `Dropout.ipynb` will help you implement Dropout and explore its effects on model generalization.

### Q4: ConvNet on CIFAR-10 (30 points)
In the IPython Notebook `ConvolutionalNetworks.ipynb` you will implement several new layers that are commonly used in convolutional networks. You will train a (shallow) convolutional network on CIFAR-10, and it will then be up to you to train the best network that you can.

Additional resources that can help you:
http://cs231n.github.io/
http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial

### Honor Code reminder ( http://osi.gatech.edu/content/honor-code )

The code for this homework must be entirely your own. You can discuss the mathematical basis for neural networks, backprogatation, etc. as a group. You can work on a whiteboard together on Math. But you shouldn't share code and you shouldn't ever see someone else's code.

## Acknowledgement
This homework is modified from Stanford's CS231n course: http://cs231n.stanford.edu/. We thank the instructors for open-sourcing the course materials.

