# NLP Transformers
> The entire repository encourages the execution of transformer models for NLP related tasks. It covers state-of-art transformers models such as BERT, TF5, etc. To understand the context of this repository, you will require to have an idea on how to implement tokenizers, perform NER, extract POS tags, etc.  

The prime objective of this repository is to build the custom training pipelines which are very modular in nature and if the input is provided in a specific format, the modules available must be able to generate models accordingly. The current scope of framework covers Pytorch for now but could be extended to support other frameworks like tensorflow, etc.

## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Features](#features)
* [Screenshots](#screenshots)
* [Setup](#setup)
* [Dataset Utilized](#dataset-utilized)
* [Usage](#usage)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [References](#references)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)
<!-- * [License](#license) -->

## General Information
- The prime objective of this repository is to build the custom training pipelines which are very modular in nature and if the input is provided in a specific format, the modules available must be able to generate models accordingly. The current scope of framework covers Pytorch for now but could be extended to support other frameworks like tensorflow, etc.  
- Make the individual NLP task based modules integratable to NLP Platter.  

## Technologies Used
- Pytorch  
- Python  

## Features
List the ready features here:
- BERT (Bidirectinal Encoder Representation of Transformers):  
  1. NER - **Done**
  2. Text Classification - **In Progress** 

## Screenshots

## Setup:
- git clone https://github.com/ManashJKonwar/NLP-Transformers.git (Clone the repository)
- python3 -m venv transformersVenv (Create virtual environment from existing python3)
- activate the "transformersVenv" (Activating the virtual environment)
- pip install -r requirements.txt (Install all required python modules)

## Dataset Utilized:
### BERT Based
- [Named Entity Recognition](https://www.kaggle.com/datasets/abhinavwalia95/entity-annotated-corpus)  
Annotated Corpus for Named Entity Recognition using GMB(Groningen Meaning Bank) corpus for entity classification with enhanced and popular features by Natural Language Processing applied to the data set.
- [Text Classification](https://www.kaggle.com/datasets/sainijagjit/bbc-dataset)  
Labeled data which classifies news covered in BBC into multiple categories. It consists of 2126 different texts and each one is labelled under 5 categories: entertainment, sport, tech, business, or politics

## Usage
### BERT Based  
1. Named Entity Recognition (NER):  
    - Training custom NER Model
      > python bert\ner\train.py
  
    - Inferencing custom NER Model  
      > python bert\ner\predict.py

2. Text Classification:  
    - Training custom Text Classification Model
      > python bert\text_classification\train.py

    - Inferencing custom Text Classification Model
      > python bert\text_classification\predict.py

## Project Status
Project is: __in progress_ 

## Room for Improvement
Room for improvement:
- Provide support for other deep learning frameworks like tensorflow

## References
[1] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding - Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova - [Paper Link](https://arxiv.org/pdf/1810.04805.pdf)

## Acknowledgements
- Official code base for BERT - [Link](https://github.com/google-research/bert).

## Contact
Created by [@ManashJKonwar](https://github.com/ManashJKonwar) - feel free to contact me!

<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->
