# mELM
Morphological Extreme Learning Machine

# Usage in distinct antiviruses

## Antivirus for IoT _malware_ from SPARC architectures

python [melm.py](https://github.com/DejavuForensics/mELM/blob/main/mELM/melm.py) -tall [/Antivirus_Dataset_IoT_SPARC_mELM_format.csv](https://github.com/DejavuForensics/IoT_SPARC/blob/main/Antivirus_Dataset_IoT_SPARC_mELM_format.csv) 
-ty 1 -nh 500 -af dilation -v

```
Pinheiro Henriques de Araújo, I., Mariano Santiago de Abreu, L., Henrique Mendes Tavares Silva, S. et al.
Antimalware applied to IoT malware detection based on softcore processor endowed with authorial sandbox.
 Journal of Computer Virology and Hacking Techniques (2024).
https://doi.org/10.1007/s11416-024-00526-0
```

## Antivirus for Citadel _malware_

python [melm.py](https://github.com/DejavuForensics/mELM/blob/main/mELM/melm.py) -tall [Antivirus_Dataset_PE32_Citadel_mELM_format.csv](https://github.com/DejavuForensics/Citadel/blob/main/Antivirus_Dataset_PE32_Citadel_mELM_format.csv) 
-ty 1 -nh 500 -af dilation -v

```
Carlos Henrique Macedo dos Santos, Sidney Marlon Lopes de Lima,
XAI-driven antivirus in pattern identification of citadel malware,
Journal of Computational Science,
82,102389 (2024).
https://doi.org/10.1016/j.jocs.2024.102389.
```

## Antivirus for Java apps

python [melm.py](https://github.com/DejavuForensics/mELM/blob/main/mELM/melm.py) -tall [Antivirus_Dataset_Jar_mELM_format.csv](https://github.com/DejavuForensics/REJAFADA/blob/master/Antivirus_Dataset_Jar_mELM_format.zip) 
-ty 1 -nh 500 -af dilation -v

```
Pinheiro, R.P., Lima, S.M.L., Souza, D.M. et al. 
Antivirus applied to JAR malware detection based on runtime behaviors. 
Scientific Reports - Nature Research 12, 1945 (2022). 
https://doi.org/10.1038/s41598-022-05921-5
```

## Antivirus for PHP apps

python [melm.py](https://github.com/DejavuForensics/mELM/blob/main/mELM/melm.py) -tall [Antivirus_Dataset_PHP_batch_1_mELM_format](https://github.com/DejavuForensics/PAEMAL/blob/master/Antivirus_Dataset_PHP_batch_1_mELM_format.csv) 
-ty 1 -nh 500 -af dilation -v

```
Lima, S.M.L., Silva, S.H.M.T., Pinheiro, R.P. et al.
Next-generation antivirus endowed with web-server Sandbox applied to audit fileless attack.
Soft Comput 27, 1471–1491 (2023).
https://doi.org/10.1007/s00500-022-07447-4
```

## Antivirus for JavaScript apps

python [melm.py](https://github.com/DejavuForensics/mELM/blob/main/mELM/melm.py) -tall [Antivirus_Dataset_JavaScript_mELM_format.csv](
https://github.com/DejavuForensics/MARADJS/blob/master/Antivirus_Dataset_JavaScript_mELM_format.zip) 
-ty 1 -nh 500 -af dilation -v

```
de Lima, S.M.L., Souza, D.M., Pinheiro, R.P. et al. 
Next-generation antivirus for JavaScript malware detection based on dynamic features. 
Knowl Inf Syst (2023). 
https://doi.org/10.1007/s10115-023-01978-4
```

## Introduction

Neural network is a computational intelligence model employed to solve pattern recognition problems. Neural networks can generalize and recognize new data that wasn't taught during training. In backpropagation strategy, adjusting many parameters is necessary to improve neural network performance. 
The neural network usually gets stuck in local minima. To eliminate these areas, control strategies are added to these networks. The network needs a lot of training time to classify the samples. The latest neural networks are accurate, but training can take many days to complete.

We have developed ELM (Extreme Learning Machine) neural networks. ELMs train and predict data faster than neural networks that use backpropagation and Deep Learning.
Kernels form the basis for learning ELMs networks. The kernels are mathematical functions used as learning method of ELMs neural networks. Kernel-based learning offers the possibility of creating a non-linear mapping of data. You don't need to add more adjustable settings, like the learning rate in neural networks. 

## Pattern Recognition

Kernels are mathematical functions used as a learning method for neural networks. Kernel-based learning allows for a non-linear mapping of data without increasing adjustable parameters. But kernels can have limitations. A linear kernel can't solve a non-linearly separable problem, like a sine distribution. Whereas a sinusoidal kernel may be able to solve a problem as long as it is separable by a sine function. Finding an optimized kernel is a big challenge in artificial neural networks. The kernel helps determine the decision boundary for different classes in an application. 

We introduce mELMs. They are ELMs with hidden layer kernels inspired by image processing operators. We call these operators morphological Erosion and Dilation. We claim that morphological kernels can adapt to any boundary decision. Mathematical morphology studies the shapes of objects in images using mathematical theory. It looks at how sets intersect and join together. Morphological operations detect the shapes of objects in images.  The decision frontier of a neural network can be seen as an n-dimensional image. In this case, n represents the number of extracted features.
The decision frontier of a neural network can be seen as an _n_-dimensional image. In this case, _n_ represents the number of extracted features. mELMs can naturally identify and represent the n-dimensional areas associated with various classes.

Mathematical Morphology is a theory used in digital image processing to process nonlinearly. Various applications like object detection, segmentation, and feature extraction use Mathematical Morphology. Morphology is based on shape transformations that preserve the inclusion relationships of objects. There are two fundamental morphological operations: Erosion and Dilation. Mathematical Morphology is a constructive theory. It builds operations on Erosions and Dilations.

Fig. 1 (b) and Fig. (c) show the result of the Erosion and Dilation operation on the same original image: Fig. 1 (a). In the eroded image, the target object is "withered". In the dilated image, the target object is dilated, as the name suggests. The Dilation kernel expands the region connected to the target class. An analogy is drawn between image processing operations and authorial mELM networks. The Dilation kernel expands the region attached to the target class (e.g. malware). In turn, the Erosion kernel expands the region belonging to the counter-class (e.g. benign).

One major difficulty in artificial neural networks is finding the best kernel for a specific task. ELM neural networks can solve a linearly separable problem using a Linear kernel, as seen in (Fig.). 2 (a). The Sigmoid, RBF, and Sinusoid kernels can solve problems that can be separated by Sigmoid, Radial, and Sinusoidal functions, as shown in Fig. 2 (b), Fig. 2 (c) and Fig. 2 (d), respectively.

A neural network's ability to generalize well can depend on the right choice of kernel. The best kernel may be dependent on the problem to be solved. 
To improve the results, consider studying various kernels for the neural network.
Studying different kernel types is computationally expensive. It requires cross-validation and multiple random starting points. 

To see a counter-example, look at how the Linear kernel is used with the Sigmoid and Sine distributions in Fig. 2 (a) and Fig. 2 (b), respectively. The classification accuracies shown in Fig. 2 (a) and Fig. 2 (b) are 78.71% and 73.00%, respectively. The Linear kernel doesn't clearly show the decision boundaries of Sigmoid and Sinusoid distributions.

Fig. 3 (a), Fig. 3 (b), Fig. 3 (c) and Fig. 3 (d) show the performance of the mELM Erosion kernel on the Linear, Sigmoid, Radial and Sinusoid distributions. The respective accuracies are 100%, 93.07%, 98.18% and 99.50%.  Fig. 4(a), Fig. 4 (b), Fig. 4 (c) and Fig. 4 (d) show the performance of the mELM Dilation kernel on the Linear, Sigmoidal, Radial and Sinusoidal distributions. The respective accuracies of 100%, 95.05%, 98.18% and 99.50%. We can see that the mELMs accurately map various distributions for different problems. It should be noted that the two attributes are normalized to the same lower and upper limit.

The kernels mELMs are successful because they can model any decision boundary. They do not follow conventional geometric surfaces like ellipses and hyperbolas. The mELMs kernels map the decision boundary in _n_-dimensional. The coordinates are based on training samples. _n_ represents the number of extracted features. mELMs can detect and model different classes by using Mathematical Morphology. It detects the shapes of bodies in images.

## Follow the instructions:

### didactic repository for pattern recognition:

-	It is not within the scope of this package to create the database. A third party has already created the learning repository. This structure follows the methodology of the ELM inventors.
-	In the path **dataset/classification/diabetes_train**, you can see the structure of the repository as shown in Fig. 5. 
    - **First column**: 1; the sample (row) belongs to the class. 0; the sample (row) belongs to the counter-class.
    - **Other columns**: input attributes (neurons) referring to features of the target application. In this diabetes_train, there are 8 input neurons. For example, in the first sample (row), the first neuron has a value of 0.11764700.
- At the end of learning (training), the ELM neural network will be capable of generalization. The ELM will classify the unseen sample as either class (1.0) or counter-class (0.0). An unseen sample refers to a sample not presented during training.

### Authorial repository for pattern recognition:

-	In the path **dataset/classification/Antivirus_Dataset_PE32_Citadel_mELM_format.csv**, you can see the structure of the repository as shown in Fig. 6.
    - **First column**: The application's name. The applications and their respective raw analyses are in the [Citadel repository](https://github.com/DejavuForensics/Citadel).
 	- **Second column**: 1; the sample (row) belongs to the class malware (Citadel). 0; the sample (row) belongs to the counter-class (serious app).
    - **Other columns**: input attributes (neurons) referring to features of the target application. In this diabetes_train, there are 430 input neurons. For example, in the first sample (row), the first neuron has a value of 0.
- At the end of learning (training), the ELM neural network will be capable of generalization. The ELM will classify the unseen sample as either class (1.0) or counter-class (0.0). An unseen sample refers to a sample not presented during training.

### Parameters of the extreme learning machine:

-    -tr: learning repository reversed to the training phase.
-    -ts: learning repository reversed to the testing phase.
-    -tall: learning repository, this includes the training and testing phase.
-    -ty:
        -    1: classification (pattern recognition). 
        -    0: regression (prediction: prediction with scientific-methodological rigor).
-    -nh: number of neurons in the hidden layer.
-    -af: activation function.
        - Kernel Linear (default): linear
        - Kernel mELM Dilation: dilation
        - Kernel mELM Erosion: erosion
        - Kernel Fuzzy-Erosion: fuzzy-erosion or fuzzy_erosion
        - Kernel Fuzzy-Dilation: fuzzy-dilation ou fuzzy\_dilation
        - Kernel Bitwise-Erosion: bitwise-erosion ou bitwise\_erosion
        - Kernel Bitwise-Dilation: bitwise-dilation ou bitwise\_dilation
        - Kernel Sigmoid: sig or sigmoid
        - Kernel Sine: sin or sine
        - Kernel Hard Limit: hardlim
        - Kernel Triangular Basis Transfer Function: tribas
        - Kernel Radial Basis Function: radbas
-    -sd: random number generator seed.
-    -v: verbose output.
-    In the console, use the extreme neural network in didactic repository. Here's an example:
```
python melm.py -tr dataset/classification/diabetes_train -ts dataset/classification/diabetes_test -ty 1 -nh 100 -af dilation -v
```
-    In the console, use the extreme neural network in a real repository. Here's an example:
```
python melm.py -tall dataset/classification/Antivirus_Dataset_PE32_Citadel_mELM_format.csv -ty 1 -nh 500 -af dilation -v
```

## Prediction

When it comes to prediction, the neural network estimates a fractional value. By prediction, we mean a forecast with scientific-methodological rigor. To train the neural network, we show it a series of historical data over time. Example, the neural network examines the daily price of oil for the past few years. The output neuron receives the daily oil price. The input neurons receive the events during that time. Events refer to government decisions, elections, coup attempts, and natural phenomena like droughts.

During training, neurons adjust their connections to prioritize periodic events. 
Seemingly irrelevant daily events can affect the daily oil price.
A person would not usually connect a small event to changes in oil prices. The neural network helps people link past events to current oil prices.
Neural networks are helpful in predicting and preventing natural disasters. Neural networks can predict when a flood will happen, giving advanced warning. In synthesis, neural networks play a fundamental role in building smart city metamodels. Smart cities use electronic sensors to gather data and reduce the effects of natural disasters. 

The evaluation of pattern recognition is different from prediction. In the context of pattern recognition, the main metric is accuracy. Neural networks perform better when closer to 100 percent in specific application. In the prediction scenario, the crucial metric is the root mean square error. The closer this is to 0, the better the performance of the neural network. The model aims to predict well by minimizing the gap between expected and actual outcomes.

## Follow the instructions:

### Didactic repository for prediction:

-	It is not within the scope of this package to create the database. A third party has already created the learning repository. This structure follows the methodology of the ELM inventors.
-	In the path **dataset/dataset/regression/sinc_train**, you can see the structure of the repository as shown in Fig. 7. 
    - **First column**: floating value to be estimated.
    - **Second column**: input attribute (neuron). This is a didactic and hypothetical database. A real application could have hundreds of input neurons.
- At the end of learning (training), the ELM neural network will be capable of prediction. The ELM will estimate a floating value from a given input neuron. 
-    The "mELM" package has a didactic dataset for prediction. It follows the methodology of the creators of ELM. However the authorial structure allows people to experiment with databases in real-world situations. These follow the structure shown in Figure \ref{fig:8_71}. In prediction problems, the second column must be a floating value. We only use classes in pattern recognition.
-    You can use databases from [Kaggle](https://www.kaggle.com/) or [UCI](https://archive.ics.uci.edu/datasets). Search for repositories on their websites that specialize in predicting and analyzing time series.
-    In the console, use the extreme neural network in didactic repository. Here's an example:
```
python melm.py -tr dataset/dataset/regression/sinc_train -ts dataset/dataset/regression/sinc_test -ty 1 -nh 500 -af dilation -v
```



