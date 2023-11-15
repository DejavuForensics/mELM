# melm
Morphological Extreme Learning Machine

Neural network is a computational intelligence model employed to solve pattern recognition problems. Neural networks can generalize and recognize new data that wasn't taught during training. In backpropagation strategy, adjusting many parameters is necessary to improve neural network performance. 
The neural network usually gets stuck in local minima. To eliminate these areas, control strategies are added to these networks. The network needs a lot of training time to classify the samples. The latest neural networks are accurate, but training can take many days to complete.

We have developed ELM (Extreme Learning Machine) neural networks. ELMs train and predict data faster than neural networks that use backpropagation and Deep Learning.
Kernels form the basis for learning ELMs networks. The kernels are mathematical functions used as learning method of ELMs neural networks. Kernel-based learning offers the possibility of creating a non-linear mapping of data. You don't need to add more adjustable settings, like the learning rate in neural networks. 

## Pattern Recognition

Kernels are mathematical functions used as a learning method for neural networks. Kernel-based learning allows for a non-linear mapping of data without increasing adjustable parameters. But kernels can have limitations. A linear kernel can't solve a non-linearly separable problem, like a sine distribution. Whereas a sinusoidal kernel may be able to solve a problem as long as it is separable by a sine function. Finding an optimized kernel is a big challenge in artificial neural networks. The kernel helps determine the decision boundary for different classes in an application. 

We introduce mELMs. They are ELMs with hidden layer kernels inspired by image processing operators. We call these operators morphological Erosion and Dilation. We claim that morphological kernels can adapt to any boundary decision. Mathematical morphology studies the shapes of objects in images using mathematical theory. It looks at how sets intersect and join together. Morphological operations detect the shapes of objects in images.  The decision frontier of a neural network can be seen as an n-dimensional image. In this case, n represents the number of extracted features.
The decision frontier of a neural network can be seen as an _n_-dimensional image. In this case, _n_ represents the number of extracted features. mELMs can naturally identify and represent the n-dimensional areas associated with various classes.

Mathematical Morphology is a theory used in digital image processing to process nonlinearly. Various applications like object detection, segmentation, and feature extraction use Mathematical Morphology. Morphology is based on shape transformations that preserve the inclusion relationships of objects. There are two fundamental morphological operations: Erosion and Dilation. Mathematical Morphology is a constructive theory. It builds operations on Erosions and Dilations.




