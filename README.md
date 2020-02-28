# Network Traffic Data Analysis Project
The basis of this project is a small exploration of data analysis techniques and its application to cybersecurity. These notes are meant to outline a rough draft for a set of tools that can be used to move towards applied machine learning in the network traffic analysis and IDS (Intrustion Detection System) space. 

Overall, I am trying to dig into the answer for this question:

**What is required to perform applied machine learning in a cybersecurity setting?**

![alt text](TheStackML.png)


# Data Sources and Collection

In cybersecurity, datasources are often comprised of:
1) PCAP files
2) Event Logs
3) Splunk Searches
4) Academic/Research data sources

Splunk provides a quality depiction of how one might use their organization's SIEM as a data source. (https://www.splunk.com/en_us/blog/conf-splunklive/what-s-new-in-the-splunk-machine-learning-toolkit-4-0-part-2.html)
![alt text](splunk.png)

There are a wealth of datasets available online. A popular curated list is linked below.

**Curated Cybersecurity Dataset List**
[https://github.com/jivoi/awesome-ml-for-cybersecurity](https://github.com/jivoi/awesome-ml-for-cybersecurity)

## UNSW-NB15 Dataset

The UNSW-NB15 Dataset can be found here: [https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/)

It is what we will use in order to build our machine learning model. The following table defines the fields is provided on the unsw website.

![alt text](unsw.png)

The network traffic was synthetically generated in a Cyber Range by the IXIA PerfectStorm tool by the Australian Center for Cybersecurity. The traffic was captured through `tcpdump` (100GB). The packets were then passed into two SIEMs, Argus and BroIDS, in order to generate the features (columns) of this dataset. Some of the 49 total features were calculated with custom algoithms.

A network topology can be found on the unsw-nb15 dataset website and is shown here. It is worth noting the collection point of the tcpdump is depicted on Router 1.
![alt text](unsw-nb15-testbed.jpg)

## Research on UNSW-NB15

Now that we have a robust and labeled dataset, it is wise to do some research on it to see how academia has used the data. The analysis done in this project is far more simple, but academic papers often give a good idea on ways to proceed with Data Preprocessing and Analysis. 

Some of the notable papers that informed this project are described here:

1. **Research on Network Intrusion Detection Based on Incremental Extreme Learning Machine and Adaptive Principal Component Analysis**
Jianlei Gao, Senchun Chai *, Baihai Zhang and Yuanqing Xia. School of Automation, Beijing Institute of Technology, Beijing 100081, China; jianleixinye@163.com (J.G.); smczhang@bit.edu.cn (B.Z.); xia_yuanqing@163.net (Y.X.)

> Gist: This paper provides descriptions of network traffic IDS integrated with Training/Testing Data sets for a I-ELM and A-PCA models. They are able to get better results in many categories over SVM, BP, and CNN.

2. **Improving the Classification Effectiveness of Intrusion Detection by Using Improved Conditional Variational AutoEncoder and Deep Neural Network**
Yanqing Yang, Kangfeng Zheng, Chunhua Wu and Yixian Yang

> Gist: This paper uses an Autocoder to generate an improved dataset before piping it into a CNN. This helps balance the dataset and improves the effectiveness of the neural network.

# Data Preprocessing
Now that we have chosed a dataset, the next step is to process the data in various ways in order to:
1) Get a better understanding of the data
2) Prepare the data for input into machine learning algorithms and neural networks

## Importing CSV into a Pandas Dataframe

Initially the data is in a CSV. Although it is possible to observer the data in excel, it may become somewhat unweildy as the data gets larger in size. Furthermore, we need to turn the data into something our python programs can understand. The `dataframe` class from the Pandas library serves this purpose. It allows us to manipulate and observe the data in an efficient and pipeline-friendly way.

The data is provided as a mix of categorical and numeric data. Some of the columns such as `proto` and `service` are example of categorical data. Example of numeric columns include `smeansz` and `dur`. 

## Categorical Feature Encoding

Since neural networks operate on numeric data, we need to encode the categorical columns. We will use One-Hot encoding and feature hashing depending on the model we are fitting to the data. 

### One Hot

One Hot encoding allows us to convert each category of a categorical feature into its own feature. This new feature will consist of only 0's or 1's depending on whether or not the given row was the associated category or not.

We use One-Hot encoding for our Neural Network in the notebook: `UNSW-NB15_PREPROC_NN.ipynb`

A disadvantage on one-hot encoding is that is increases the dimensionality of our dataset by the number of categories in each categorical column. This can make it unsuitable for certain machine learning algorithms. 

### Hash Trick

The hash trick is a method of encoding categorical columns into numeric columns without increasing the dimensionality of the dataset. However, data may be lost during the hashing process. There are many ways to go about hashing the data, we will use the `sklearn.feature_extraction.FeatureHasher` built in function.

We use the hash trick in the random forest notebook: `UNSW-NB15_RandomForest.ipynb`



## Normalization (Numeric Columns)
Neural Networks are sensitive to data with features that have large differences in their numeric range. For example, features in the UNSW-NB15 dataset such as `dur` have values in the hundedths and `dbytes` with values in the thousands. 

Normalization is performed in order to ensure that all values in every numeric column are between 0 and 1. This is important in ensuring that no features are overshadowed by others during the NN learning process. 

![alt text](numeric_scaling.png)

# Data Analysis
## Observing UNSW-NB15 Data with UMAP

Using UMAP perform dimension reduction and view the spacial relationships of clusters in the data:

https://github.com/jackg-ch/unswnb15-model-1/blob/master/UMAP_UNSW-NB15_2.ipynb


## SKLearn Random Forest on UNSW-NB15

![alt text](decision_tree.jpeg)

Source: https://towardsdatascience.com/understanding-random-forest-58381e0602d2

Running SKLearn's RandomForest classifier on UNSW-NB15 Dataset:

https://github.com/jackg-ch/unswnb15-model-1/blob/master/UNSW-NB15_RandomForest.ipynb

## Tensorflow 2 ANN on UNSW-NB15

Implementing a tensorflow 2 neural network for binary classification on UNSW-NB15

https://github.com/jackg-ch/unswnb15-model-1/blob/master/Overall_Demo.ipynb

# Data Products
Current recommendation: Python's Flask Backend with a ReactJS frontend to deliver lightweight and modern AI-driven dashboards.

# Sources/Citations
Moustafa, Nour. Designing an online and reliable statistical anomaly detection framework for dealing with large high-speed network traffic. Diss. University of New South Wales, Canberra, Australia, 2017.

Moustafa, Nour, and Jill Slay. "UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)." Military Communications and Information Systems Conference (MilCIS), 2015. IEEE, 2015.