# UHI LENS

UHI LENS is an innovative solution designed to mitigate Urban Heat Islands caused due to the incremental difference in temperatures in Urban areas compared to the rural areas surrounding them. This designed system aims to mitigate Urban Heat Islands by considering a diverse set of factors such as Land Use Land Cover, Socio-Economic features and Sentinel features of the region of interest(ROI) allowing users to perform a detailed analysis and hence identifying the root cause for UHI and recommending mitigation strategies to curb it.

## Design of the System

The system is designed using HTML, CSS and Javascript for frontend and Flask as the backend framework. 

Main Frameworks used for Frontend:
* Tailwind CSS
* ArcGIS Maps SDK for JavaScript
* Plotly 

Backend Framework:
* Flask

## Flow of the System
The system takes a location name as input to assess the impact of urbanization. Through a series of computations, it generates the Urban Heat Island Index (UHII) for the area, categorizes the heat source as man-made (controllable) or natural (uncontrollable), analyzes monthly UHII variations, and provides LLM-based mitigation recommendations.

### Fetching the required data for analysis
The following APIs are used to gather data required to the analysis:

* Geocoding API: This API fetches the latitude and longitude of the desired ROI. 
* Earth Engine API: This API enables communication between Google Earth Engine and our Flask server. The fetched latitude and longitude are given as inputs to capture the most recent satellite image of the ROI post authentication.

Using the above APIs, we get:
* The Satellite Image of the ROI
* Sentinel Features of the ROI
* Socio Urban Features of the ROI

### Preprocessing and Feature Extraction
Preprocessing techniques such as cloud masking, color correction, atmospheric correction, etc are applied to improve the quality of the data. Semantic Segmentation is applied to the satellite image to avail the LULC Mask and features related to it. Various statistics such as mean, standard deviation, min, max for sentinel and socio urban features are also extracted.

The final set of features utilized for model development are:
* 15 Sentinel Features
* 9 Socio Urban Features
* LULC features

Each of these features can be viewed and analysed separately. 

### Model Development
This system utilizes outputs of 5 machine learning models, each used for a specific purpose. All the models are briefed out:

* UHI Index Prediction: A DL model to predict UHI Index directly from Satellite Imagery

* Semantic Segmentation Model: A DL model to predict the LULC of the ROI

* UHI Classification: A Random Forest Classifier to predict the cause of UHI as natural/manmade.

* 12 month UHI Regression Model: An ensemble model combining base models RFR, SVR and LGBM to forecast month wise UHI Index. 

* LLM: Utilizing Llama3 to provide region specific mitigation strategies. 

### Deployment

All the above discussed models are deployed and this github repository gives access to the code.

MVP Link [Demo Video]​
https://youtu.be/PXvqN1cuZ3M?si=hxvLe0Qr48IIPIOe 