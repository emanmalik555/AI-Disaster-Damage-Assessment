# AI-Disaster-Damage-Assessment
An offline AI-powered computer vision system for assessing building damage after natural disasters.
# 🏚️ AI Disaster Damage Assessment

**Building AI course project**

## Summary

AI Disaster Damage Assessment is an offline computer vision system that analyzes images of buildings affected by natural disasters such as earthquakes and floods. It classifies the level of structural damage to support faster emergency response and disaster recovery without relying on paid APIs or internet connectivity.

## Background

Natural disasters damage thousands of buildings every year, making manual damage assessment slow and difficult. This project aims to automate the initial assessment process using artificial intelligence.

Problems it solves:
* Speeds up damage assessment
* Reduces manual inspection effort
* Helps prioritize rescue operations
* Supports disaster management agencies

## How is it used?

Users upload an image of a building, and the AI model predicts the level of damage (No Damage, Minor Damage, Major Damage, or Destroyed). The system also provides a confidence score and a basic recommendation for emergency response.

Target users:
* Emergency responders
* Disaster management authorities
* NGOs
* Researchers

## Data Sources and AI Methods

**Dataset:**
* xBD Building Damage Assessment Dataset
* Public disaster image datasets

**AI Methods:**
* Convolutional Neural Networks (CNN)
* Image Classification
* Computer Vision using OpenCV

## Challenges

This project does not replace professional structural engineers and may be less accurate on poor-quality images or disaster types not included in the training dataset.

## Future Improvements

* Support satellite and drone imagery
* Damage localization using Grad-CAM
* Mobile application
* Multi-disaster classification
* PDF damage assessment reports

## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* Streamlit
* NumPy
* Matplotlib

## Author

**Malik Eman Waheed**

Computer Science Student

Solo AI Hackathon Project
