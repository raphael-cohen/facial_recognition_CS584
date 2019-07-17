# facial_recognition_CS584
Testing different algorithms for facial recognition. 

This project aims at making us better understand different classification methods as well as
their parameters by fine tuning them to get the best performances possible. The dataset used
as testbed for our experimentation consists of face images distributed among three major
classes. Namely, “Bush”, “Serena Williams” and “Other”. As we are only interested in binary
classification and this is why we will consider two classification problems, classifying Bush
versus the other ones (including Serena Williams) and classifying Serena Williams versus the
other ones (including Bush). One of the greatest challenges we will try to overcome for this
project is the fact that Bush and Williams classes are highly unbalanced. In order to have
meaningful results for the classification task we want to accomplish, we will use the F1-score
metric throughout the entirety of this project. For the first two phases of this project, we will
consider the results from two classifiers that are KNN and SVC, with and without prior PCA
transformation. The last two phases focus on convolutional neural network models (CNN),
with and without pre-training them first on a larger dataset.

PHASE 1
During this first phase, instructions were to focus on two specific classifiers. K-nearest
neighbors (KNN) and Support vector machine (SVM). Both of them have their own set of
parameters that we are going to explore to improve our classification performances. For Phase
1 and Phase 2 all our results will be obtained by using a 3-Fold Cross Validation, with the
StratifiedKFold parameter to respect the distribution of each class in our samples.

PHASE 2
The reason for this phase is to test whether or not we should use PCA transformation on our
data for either, improving our F1-score or simply reduce the computation time by reducing
our problem’s dimensions.


PHASE 3
For this phase, the objective was to build a deep neural network model with keras, that
would try to classify Bush and Williams images. Convolutional Neural Networks (CNN) are
often used when it comes to image recognition and this is why I decided to use this type of
architecture for my model.


PHASE 4
For this phase we had to select an external dataset to pre-train the network. Why ? Because
as we do not have many observations from Bush and Williams, we can’t use a complicated
model nor train the simple one correctly. By pre-training the network on some bigger dataset,
we can actually make it learn properly how to recognize useful traits to classify faces. Once
this new pre-trained model is able to detect interesting features from a face picture, we can
re-train it but this time on the Bush/Williams dataset so we can specialize it for our purpose. It
is supposed to help the network generalize its learning and then apply it on something more
specific. It should limit the overfit and help overall performances.


See report for more details
Feel free to ask the data

