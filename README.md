# Udacity_AWS_MLE_NanoDegree_Capstone
## This is the repository for the Capstone Project for the Udacity AWS Machine Learning Engineer Nano Degree

### `proposal.pdf` is the Capstone Project Proposal 


# Inventory Monitoring at Distribution Centres

## Introduction to the Problem Domain and Statement

A lot of Corporations, which handle physical cargo and deal with supply chain of any kind of goods have tried to bring in automation to make their processes more efficient and accurate. A great example of this is Amazon, who is one of the biggest hubs of delivery of all kinds of goods. These goods are often stored in big warehouses. Since the quantity of these items are in a huge amount, to physically do inventory monitoring would require both large and intelligent human resources, which are both expensive and prone to errors. 

This is where robots come in to help in Inventory Monitoring. They can be trained with Machine Learning Models, to perform tasks like Object Detection,  Outlier & Anomaly Detection and much more. Once trained, these models are scalable, and can be deployed at a low cost for usage in actual warehouses and distribution centres on industry level robots. 

The robots carry objects which are present in bins, and for our problem, each bin can contain 1-5 objects.

The problem we aim to tackle in this project is to count the number of items present in the bin. This is an **Multi-Class Image Classification task**, of classifying number of items in 1 – 5 in input image. This is a worthwhile problem to solve, for it has immense real world applications. If we can develop a model, which can take in a picture of a bin, and accurately return the number of objects present in that, we could solve & thus fully automate one crucial step in the Inventory Management process! 


## Project Set Up and Installation
**OPTIONAL:** If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to make your `README` detailed and self-explanatory. For instance, here you could explain how to set up your project in AWS and provide helpful screenshots of the process.

## Dataset

### Overview

The dataset used in this problem is the open source Amazon Bin Image Dataset. This dataset has 500,000 images of bins containing one or more objects present in it. Corresponding to each image, is a metadata file, which contains information about the image, like the number of objects it has, the dimensions and type of objects. For our problem statement, we only need the total count of objects in the image. 

**TODO**: Explain about the data you are using and where you got it from.

### Access
**TODO**: Explain how you are accessing the data in AWS and how you uploaded it

## Model Training
**TODO**: What kind of model did you choose for this experiment and why? Give an overview of the types of hyperparameters that you specified and why you chose them. Also remember to evaluate the performance of your model.

## Machine Learning Pipeline
**TODO:** Explain your project pipeline.

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
