# Phishing Mail Detector

**A machine learning-based system to detect phishing emails using text and image analysis.**

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Dataset](#dataset)
* [Installation](#installation)
* [Usage](#usage)
* [Model Details](#model-details)
* [Evaluation](#evaluation)
* [Future Improvements](#future-improvements)

## Overview

This project is a **Phishing Mail Detector** that classifies emails as **spam/phishing** or **safe** using a machine learning approach. It leverages **textual features** (like URLs, domains, content patterns) for accurate detection.

It is designed to help users and organizations prevent phishing attacks and secure their email communications.

## Features

* Detects phishing emails based on text.
* CLI-based interface for easy testing of new emails.
* Provides **probability score** of an email being phishing.
* Detailed evaluation metrics including accuracy, confusion matrix, and classification report.
* Easily extendable for new features and datasets.

## Dataset

The model was trained on a combination of publicly available **phishing and legitimate email datasets**. Features extracted include:

* Number of URLs and unique domains.
* Presence of suspicious keywords.
* HTML content structure.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/phishing-mail-detector.git
cd phishing-mail-detector
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the script to test a new email:

```bash
python detect_phishing.py --email "path_to_email_file.eml"
```

2. The system outputs whether the email is **safe** or **phishing**, along with the confidence score.

## Model Details

* **Algorithm:** Logistic Regression (text features) + optional image classifier for emails with images.
* **Text Features:** TF-IDF vectorization, URL/domain analysis, content features.
* **Image Features:** Processed using OpenCV/PIL for suspicious patterns.
* **Training:** 80/20 train-test split with cross-validation for hyperparameter tuning.

## Evaluation

The model was evaluated using standard metrics:

* **Accuracy:** ~[Your Accuracy]%
* **Precision, Recall, F1-Score:** Provided in the classification report.
* **Confusion Matrix:** Helps visualize false positives and false negatives.

## Future Improvements

* Integrate with **real-time email clients** for live phishing detection.
* Add **advanced NLP features** like semantic analysis and transformers.
* Improve image analysis using deep learning models.
* Reduce false positives for safe emails.


