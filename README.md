AI Based Covid-19 Detection from Lung X-Rays



***Abstract* — COVID-19 continues to have catastrophic effects on the lives of human beings throughout the world and also exposes the vulnerability of healthcare services worldwide, especially in underdeveloped countries. Hence, it is very important to design an automated and early diagnosis system which can provide fast decisions and greatly reduce the diagnosis error. The lung X-ray images along with emerging Artificial Intelligence methodologies, in particular Deep Learning algorithms, have recently become a worthy choice for early COVID-19 screening. This paper proposes a Deep Learning assisted automated method using  X-ray images for early diagnosis of COVID-19.**

***Index Terms*—Artificial Intelligence, Deep Learning, X-ray images, Convolutional Neural Networks, COVID-19, Binary Classifier** 

1. Introduction 

The outbreak of coronavirus disease 2019 (COVID-19) proceeds to have an emerging impact on public health and global well-being. The virus was first recognized in Wuhan, Hubei, China, in December 2018, and on March 11, 2020, the world health organisation (WHO) perceived it as a pandemic . Over 12.7 million people have been affected with COVID-19 to date (July 12, 2020) globally, with more than 565,219 losses of life . Studies have discovered that the transmission rate (TR) of the virus is extremely frightening, with a generative rate between 2.24 to 3.58, which is enormously higher than any other type of virus flu . The remedy for this viral infection is symptomatic and supportive since there are no acknowledged vaccines or drugs .

Recently, the reverse transcriptase-polymerase chain reaction (RT–PCR) diagnostic method was found to be effective in detecting the virus. However, the method has some drawbacks, including longer detection time and lower detection rate of the virus. Strict requirements in the laboratory and different features for the testing could be accounted for the drawbacks. Researchers are working on overcoming the limitations of RT – PCR testing to better diagnose and detect COVID-19. According to the recommendations by WHO provided in October 2020, chest imaging examination is an effective method for the detection of clinical symptoms of people who have been affected and recovered from the virus. In addition to that, other diagnostics tests are also suggested, such as ultrasound, X-rays and MRI of the chest and computed tomography and needle biopsy of the lung. At present, chest X-ray is enormously used for the detection of the COVID-19 cases compared to the CT image as it takes longer for imaging, and CT scanners are not available in many underdeveloped countries. In addition, CT imaging is expensive, and pregnant women and children may face health risks due to its high radiation. In order to avoid that, X-ray imaging has played a major role in many medical and epidemiological cases due to its wider availability. Chest X-ray is promising for emergency cases and treatment due to its operational speed, cost and simplicity for the radiologists. However, in earlier research, some inconsistencies were observed for the chest X-ray images taken from people affected by the COVID-19.

Previously, artificial intelligence techniques were engaged to successfully diagnose Pneumonia either from chest X-ray images or CT. The classification methods employed differ from Bayesian function to convolutional neural network (CNN). Lately, CNN has been found to be useful and effective in identifying COVID-19 through image classification. CNN consists of multilayer neural networks, which are highly capable of recognizing the image patterns without conducting diverse preprocessing of the images. Although several CNN models, including AlexNet, Resnet50, VGG16, VGG19, are available, VGG19 demonstrates better performance for the COVID-19 classification.

2. Literature Survey 

In recent months, researchers have investigated and analysed chest X-ray images using deep learning algorithms to detect COVID-19. First, the images are preprocessed using the CNN technique for extracting better features, which are fed in deep learning algorithms for image classification. Ahammed et al. [1] proposed a deep neural network based system where CNN provided high accuracy (94.03%). The authors trained the system with normal or  pneumonia and COVID-19 patient’s chest X-ray images. The limitation of the work was that a dataset with only 285 images was used for developing the system, and this small number of data was not perfect for training a deep learning-based system for the COVID-19 prediction.

Nur-A-Alam et al. [5] designed and developed an intelligent system for the COVID-19 identification with high accuracy and minimum complexity by combining the features extracted by histogram-oriented gradient (HOG) features and convolutional neural network (CNN). Chest X-ray images were entered into the system in order to produce the output of the marked lung significant region, which was used to identify COVID-19. The proposed feature fusion system showed a higher classification accuracy (99.49%) than the accuracies obtained by using features obtained by individual feature extraction techniques, such as HOG and CNN. CNN produced the best classification accuracy compared to the other classification techniques, such as ANN, KNN and SVM. Furthermore, the proposed fusion technique was validated with higher accuracies using generalisation and k-fold validation techniques.

El-Rashidy et al. [2] introduced a framework consisting of three layers: patient layer, cloud layer and hospital layer. A set of data was collected from the patient layer using some wearable sensors and a mobile app. A neural network-based deep learning model was used to detect COVID-19 using the patient X-ray images. The proposed model achieved 97.9% accuracy and 98.85% specificity. 

Khalifa et al. [3] developed a classification approach for the treatment purposes of coronavirus on a single human cell-based on treatment type and treatment concentration level using deep learning and machine learning (ML) methods. Numerical features of the data sets were converted to images for building the DCNN model. The testing accuracy of treatment classification obtained by the model was as high as 98.05% compared to the other traditional ML methods, including support vector machine (SVM) and decision tree (DT). However, the proposed DCNN model showed less testing accuracy (98.2%) compared to the DT (98.5%) for the prediction of treatment concentration level. Deep transfer models (i.e., Alexnet) have not been employed in their study.

Sekeroglu et al. [4] developed a model using deep learning and machine learning classifiers where a total of 38 experiments was conducted by CNN for the detection of the COVID-19 using the chest X-ray images with high accuracy. Among them, 10 experiments were performed using 5 different machine-learning algorithms, and 14 experiments were carried out by the state-of-the-art pre-trained network for transfer learning. The system demonstrated 98.50% accuracy, 99.18% specificity and 93.84% sensitivity. They concluded that the system developed by CNN was capable of achieving COVID-19 detection from a limited number of images without any preprocessing and with minimised layers.

Boran SekerogluIn et al. [8]  In their study, several experiments were performed for the high-accuracy detection of COVID-19 in chest X-ray images using ConvNets. Various groups — COVID-19 / Normal, COVID-19 / Pneumonia, and COVID-19 / Pneumonia / Normal — were considered for the classification. Different image dimensions, different network architectures, state-of-the-art pre-trained networks, and machine learning models were implemented and evaluated using images and statistical data. The results showed that the convolutional neural network with minimised convolutional and fully connected layers is capable of detecting COVID-19 images within the two-class, COVID-19/Normal and COVID-19/Pneumonia classifications, with mean ROC AUC scores of 96.51 and 96.33%, respectively. In addition, the second proposed architecture, which had the second-lightest architecture, is capable of detecting COVID-19 in three-class, COVID-19 / Pneumonia / Normal images, with a macro-averaged F1 score of 94.10%. 

Sahlol et al. [9] proposed an improved hybrid classification approach using CNNs and marine predators algorithm for classifying COVID-19 images, which were obtained from international cardiothoracic radiologists. Inception architecture of CNNs was employed to extract features, and a swarm-based marine predators algorithm was used to select the most relevant features from the images. Drawbacks were  the research work did not consider any fusion approach to improve the classification and feature extraction of the COVID-19 images.

Wang et al. [12] have evolved a transfer learning approach (Xception version) using deep learning models for diagnosing COVID-19. The proposed approach confirmed 96.75% diagnostics accuracy. Furthermore, Deep capabilities and system learning classification (Xception + SVM) had been additionally hired to expand an efficient diagnostic approach for enhancing the accuracy of the Xception version through 2.58%. From the result, the authors claimed that their proposed approach attained better classification accuracy and efficient diagnostic overall performance of the COVID-19. However, the authors have now no longer compared their consequences with the existing comparable works.

This article [13] described the effectiveness of test protocols that are common in most articles dealing with automatic diagnostics for COVID 19. They have shown that they can bias these logs and learn how to predict characteristics that depend more on the source dataset than compared to relevant medical information. Process relevant medical information. However the authors also suggested some solutions to find  new test protocols and evaluation methods. Finally, the article is concluded by showing that creating a fair testing protocol is a challenging task, and by providing a method to measure how fair a specific testing protocol is.

The proposed method[14] has used four CNN models and two cascaded network models to divide X-ray samples into two categories: COVID-19 and healthy people. In addition to that the model architectures were applied for feature extraction and classified categories through an FC layer. Experimental results showed that, under the same conditions, cascade network model 2 was best for classifying COVID-19 and healthy people. It could significantly improve classification performance, with accuracy of 96%, F1-score of 95.5%, 96.10% precision, 96.42% recall, and 98.7% AUC. The proposed method has major limitations. Such as, the experiment only applies to X-ray images, and not CT images, because X-ray images are RGB and CT images are grayscale. It can only be used to classify COVID-19 patients and healthy people, and it cannot classify COVID-19 and general pneumonia.

3. Methodology 

For the study in classification of a given lung x-ray image, we use an Artificial Intelligence model, and the underlying algorithm is based on the Convolutional Neural Network. The Workflow employed for this research.

- Obtaining dataset
  - Dataset has been obtained from open source resources.
  - We use the metadata.csv available for each dataset to split data into two classes (binary) normal and covid. 
  - Multiple classes of x-ray images were available, pneumonia, sars-cov-1, etc. 
  - But for this research we employed only a binary classifier.
  - We use the libraries - pandas, numpy, os - to study the metadata, clear null values, transfer images of covid and normal into respective directories and store the data in an array. 
- Input data set
- Split into train and test data
  - This is one within the model with the help of the ImageDataGenerator package from keras.preprocessing.image.
- X-Ray image preprocessing		
- Designing Proposed CNN model
- Training Proposed CNN model
  - Feature Extraction
- Performance Evaluation
  - Accuracy
  - Loss
  - Confusion Matrix
- Covid-19 or Normal Classification 

Modules used are:

- Tensorflow

TensorFlow is a free and open-source software library for machine learning and artificial intelligence. It can be used across a range of tasks but has a particular focus on training and inference of deep neural networks. [Source: Wikipedia]

- Keras

Keras is an open-source software library that provides a Python interface for artificial neural networks. Keras acts as an interface for the TensorFlow library. Up until version 2.3, Keras supported multiple backends, including TensorFlow, Microsoft Cognitive Toolkit, Theano, and PlaidML. [Source: Wikipedia]

Keras is the main module used in this research to implement the Sequential model.

- Numpy

NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays. [Source: Wikipedia]

- Pandas

Pandas is a Python library used for working with data sets. It has functions for analysing, cleaning, exploring, and manipulating data. Pandas allows us to analyse big data and make conclusions based on statistical theories. Pandas can clean messy data sets, and make them readable and relevant. Relevant data is very important in data science. [Source: w3schools]

- OS

The OS module in Python provides functions for interacting with the operating system. OS comes under Python's standard utility modules. This module provides a portable way of using operating system-dependent functionality. [Source: GeeksforGeeks]

- Matplotlib

Matplotlib is a comprehensive library for creating static, animated, and interactive visualisations in Python. Create plots. Make interactive figures that can zoom, pan, update. Customise visual style and layout.  [Source: [Matplotlib — Visualization with Python](https://matplotlib.org/)]

- Google Colaboratory 

Google Colab is an open source tool provided by google to work on python notebooks (.ipynb files). While maintaining all the data on Google drive and doing computation on an online tool like colab has provided an edge on both speed and time. Thus all the data and computations were run on the cloud instead of investing into any personal hardware resources.

A little about CNN

- Within Deep Learning, a Convolutional Neural Network or CNN is a type of artificial neural network, which is widely used for image/object recognition and classification. Deep Learning thus recognizes objects in an image by using a CNN.
- CNNs have fundamentally changed our approach towards image recognition as they can detect patterns and make sense of them. They are considered the most effective architecture for image classification, retrieval and detection tasks as the accuracy of their results is very high.
4. Experimentations and Results

Section 1 - Obtaining dataset [7]

`	`We have obtained a dataset included with a code for a data generator to get 10,000 images. But the dataset was used using the python package, instead of generating 10,000 images of different dimensions. Two classes of images were recorded into an array ready for model generator. And in turn this generated data is sent to the model for training and validation.

The data available from dataset [7], contained 28 normal human X-rays and 70 COVID-19 infected human X-rays which is specific from the code snippet in Fig.1. In these initial stages of the research we had to tackle the dataset as this classification model was binary and the percentages of the image files were not equal, which is evident from the mentioned figures. This uneven dataset thus created a bias.

![](Aspose.Words.d4f095c1-66a9-4fe4-ae0d-c10bccb6b3b2.001.png)

Fig. 1. Number images from the dataset used

Section 2 - Adjusting and augmenting data set 

From Fig.2. it is evident that the images have different sizes. To accommodate this to the model we have set the parameters of the images, i.e. the height to 150, the width to 150 and the colour channels to 3. Using ImageDataGenerator we augment the pictures using horizontal\_flip.

![](Aspose.Words.d4f095c1-66a9-4fe4-ae0d-c10bccb6b3b2.002.png)

Section 3 - Preparing model from reference; Training and Validation [6]

This model includes 7 convolutional layers along with the “relu” and “sigmoid” activation method. Parameters of the referred model are mentioned in Fig. 3. 70% of the dataset was used for training and 30% for Validation/Testing. This model obtained a training accuracy of 95% and validation accuracy of 100%. Taking assumptions from the size of the data set we can notice there might be underfitting of the data. This assumption was evident from Fig. 6. when a Normal X-ray image was classified as Covid-19 infected. And also there is an underlying bias in the training of the model as previously mentioned in Section 1. ![](Aspose.Words.d4f095c1-66a9-4fe4-ae0d-c10bccb6b3b2.003.png)



![](Aspose.Words.d4f095c1-66a9-4fe4-ae0d-c10bccb6b3b2.004.png)

![](Aspose.Words.d4f095c1-66a9-4fe4-ae0d-c10bccb6b3b2.005.png)

![](Aspose.Words.d4f095c1-66a9-4fe4-ae0d-c10bccb6b3b2.006.png)

Section 4 - Increasing the size of the dataset 

As a result of underfitting, a larger dataset is used [10][11]. As COVID-19 lung X-rays are rare to find, as part of this paper we have accumulated just above 400 images belonging to each class COVID-19 and just above 250 belonging to class Normal. This caused a bias in the model estimator. And thus we move forward to the next section implementing a model estimator with lesser bias.   ![](Aspose.Words.d4f095c1-66a9-4fe4-ae0d-c10bccb6b3b2.007.png)

![](Aspose.Words.d4f095c1-66a9-4fe4-ae0d-c10bccb6b3b2.008.png)

Section 5 - Increasing the size of the dataset for the class Normal

In this section, the number of X-ray images of both the classes (Normal, COVID-19) are just above 400, thus eliminating the biases used in the above sections. New performance evaluation of the model can be seen in Fig. 9 and Fig. 10. For this Section, the training accuracy was 95% and validation accuracy was 88%.

![](Aspose.Words.d4f095c1-66a9-4fe4-ae0d-c10bccb6b3b2.009.png)

![](Aspose.Words.d4f095c1-66a9-4fe4-ae0d-c10bccb6b3b2.010.png)


Section 6 - Modifying the current model

The model layers have been reduced to 6 from 7, parameters for the layers have been modified (Fig. 11). The data is modified as 75% data for training and 25% data for validation. As part of this experimentation, some observations where there were spikes in loss of training the model - Fig.12.B. This model acquired a training accuracy of 90.8% and validation accuracy of 81%.

![](Aspose.Words.d4f095c1-66a9-4fe4-ae0d-c10bccb6b3b2.011.png) 

![](Aspose.Words.d4f095c1-66a9-4fe4-ae0d-c10bccb6b3b2.012.png)

![](Aspose.Words.d4f095c1-66a9-4fe4-ae0d-c10bccb6b3b2.013.png)

Section 7 - Creating a model from Scratch

This model is built on 4 convolutional layers. This is 3 layers less than the reference model[6]. The model has been modified with trial and errors/experimentation (Fig. 14). Dataset used is just below 850 images of two classes, mentioned in Section 5. The number of epochs used are 18. The batch size is 50. And 75% of the dataset has been set for training and 25% for validation. We achieved a training accuracy of 92% with a maxima of the training\_accuracy curve at 95% and a validation accuracy of 76% with a maxima of the validation\_accuracy at 88% (Fig. ). As the dataset remains unchanged from the previous section, the confusion matrix remains the same (Fig. 13).![](Aspose.Words.d4f095c1-66a9-4fe4-ae0d-c10bccb6b3b2.014.png)![](Aspose.Words.d4f095c1-66a9-4fe4-ae0d-c10bccb6b3b2.015.png)




V. Challenges Faced

We took the approach of learning on an application basis. Instead of looking into the basics of Keras library, we took up the project from the reference available from Kaggle and started working on it and as we went further, we understood the concepts. The underlying modules in python were extremely helpful to complete this project.

Another potential challenge in classification is differentiating between SARS Cov-2 also called COVID-19 and Pneumonia. Adding to this, classification of other low end lung diseases and COVID-19 using X-ray images is also a unique challenge. 

Though the accuracy of the models in this classification is high, the accuracy of a radiologist is always preferable given experience and also the dependence of a human being’s life cannot be put at risk due to small errors that might be caused due to the model.

VI. Conclusions 

In this paper, Convolutional Neural Network has been implemented considering it’s advantages over Artificial Neural Network. The classification model estimator can be given a better treatment and then can be deployed in hospitals to give a quick check on whether a patient has COVID-19 or not. At the times of crisis, saving time is very important. But the classification is completely dependent on past data and when presented with a new variation might put the life of the patient at risk and thus is a drawback of this implementation. A Sequential model from the TensorFlow library in Python has been used in binary classification (two classes) and a training accuracy of 92% and validation accuracy of 76% were obtained. 

References

1. Ahammed, K.; Satu, M.S.; Abedin, M.Z.; Rahaman, M.A.; Islam, S.M.S. Early Detection of Coronavirus Cases Using Chest X-ray Images Employing Machine Learning and Deep Learning Approaches. medRxiv 2020. medRxiv 2020.06.07.20124594.
1. El-Rashidy, N.; El-Sappagh, S.; Islam, S.M.R.; El-Bakry, H.M.; Abdelrazek, S. End-To-End Deep Learning Framework for Coronavirus (COVID-19) Detection and Monitoring. Electronics 2020, 9, 1439. [[End-To-End Deep Learning Framework for Coronavirus (COVID-19) Detection and Monitoring](https://www.mdpi.com/2079-9292/9/9/1439)]
1. Khalifa, N.E.M.; Taha, M.H.N.; Manogaran, G.; Loey, M. A deep learning model and machine learning methods for the classification of potential coronavirus treatments on a single human cell. J. Nanoparticle Res. 2020, 22, 1–13. [[RETRACTED ARTICLE: A deep learning model and machine learning methods for the classification of potential coronavirus treatments on a single human cell - Journal of Nanoparticle Research](https://link.springer.com/article/10.1007%2Fs11051-020-05041-z)]
1. Sekeroglu, B.; Ozsahin, I. Detection of COVID-19 from Chest X-Ray Images Using Convolutional Neural Networks. SLAS Technol. Transl. Life Sci. Innov. 2020, 25, 553–565. [[Detection of COVID-19 from Chest X-Ray Images Using Convolutional Neural Networks](https://pubmed.ncbi.nlm.nih.gov/32948098/)]
1. Nur-A-Alam  , Mominul Ahsan  , Md. Abdul Based  , Julfikar Haider and Marcin Kowalski ; COVID-19 Detection from Chest X-ray Images Using Feature Fusion and Deep Learning [[COVID-19 Detection from Chest X-ray Images Using Feature Fusion and Deep Learning](https://www.mdpi.com/1424-8220/21/4/1480/pdf)]
1. Eswar Chand; 2019; Kaggle; accessed in July 2021; [Covid-19 Detection from Lung X-rays](https://www.kaggle.com/eswarchandt/covid-19-detection-from-lung-x-rays)
1. Nabeel Sajid; 2019; Kaggle; accessed in July 2021; [COVID-19 Patients Lungs X Ray Images 10000](https://www.kaggle.com/nabeelsajid917/covid-19-x-ray-10000-images)
1. [Boran Sekeroglu](https://www.ncbi.nlm.nih.gov/pubmed/?term=Sekeroglu%20B%5BAuthor%5D&cauthor=true&cauthor_uid=32948098) , Ilker Ozsahin; Detection of COVID-19 from Chest X-Ray Images Using Convolutional Neural Networks ; [SLAS Technol.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7502682/#) 2020 Dec; 25(6): 553–565. [<https://link.springer.com/article/10.1007/s11051-020-05041-z>]
1. Sahlol, A.T.; Yousri, D.; Ewees, A.A.; Al-Qaness, M.A.; Damasevicius, R.; Abd Elaziz, M. COVID-19 image classification using deep features and fractional-order marine predators algorithm. Sci. Rep. 2020, 10, 1–15.[<https://www.nature.com/articles/s41598-020-71294-2>]
1. Paul Mooney; 2018; Kaggle; accessed in August 2021; [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) 
1. <https://github.com/ieee8023/covid-chestxray-dataset> 
1. Wang, D.; Mo, J.; Zhou, G.; Xu, L.; Liu, Y. An efficient mixture of deep and machine learning models for COVID-19 diagnosis in chest X-ray images. PLoS ONE 2020 Nov, 15, e0242535. <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0242535>
1. Gianluca Maguolo\*, Loris Nanni; A Critic Evaluation of Methods for COVID-19 Automatic Detection from X-Ray Images, 2020 sep. <https://arxiv.org/ftp/arxiv/papers/2004/2004.12823.pdf>
1. Dongsheng Ji, Zhujun Zhang, Yanzhong Zhao, Qianchuan Zhao, "Research on Classification of COVID-19 Chest X-Ray Image Modal Feature Fusion Based on Deep Learning", Journal of Healthcare Engineering, vol. 2021, Article ID 6799202, 12 pages, 2021.
1. [Research on Classification of COVID-19 Chest X-Ray Image Modal Feature Fusion Based on Deep Learning](https://www.hindawi.com/journals/jhe/2021/6799202/)

