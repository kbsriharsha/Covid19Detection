# Covid19 Detection using Xray Imaging ![](https://img.shields.io/github/license/sourcerer-io/hall-of-fame.svg?colorB=ff0000) ![](https://img.shields.io/badge/Harsha-Karpurapu-brightgreen.svg?colorB=ff0000)

This model helps in predicting the Covid19 from the patients X-ray. This project is created for research purposes and for people who are interested in social good using AI. 

### Code Requirements
Just execute the below command from your terminal for installing all the required dependencies. 
##### pip install requirement.txt

### Data Collection
The important step in this project is the data collection process. JHU has a github repo with the xrays of the patients who tested positive (positive samples). Currently JHU is still collecting the xray images of patients and the github is having only 100 images at the time when I pulling was the data. Negative samples have been collected from a kaggle competetion dataset (pneumonia prediction)

### Model Description
As the dataset is pretty small (200 samples; 100 positive and 100 negative), we used transfer learning for this purpose. WE have selected mobilenet as pretrained model and retrained the last 23 layers to made it useful for this purpose. 

### Model Dynamics


### References
- John Hopkins University Data Github https://github.com/ieee8023/covid-chestxray-dataset
- Kaggle Competetion dataset https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
- Keras: https://keras.io/
- Adrian Research Article: https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in-x-ray-images-with-keras-tensorflow-and-deep-learning/


