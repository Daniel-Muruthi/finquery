# FinQuery: Intelligent Banking Intent Classification for Customer Support

## Overview

FinQuery is a Django-based web application that demonstrates how to build and deploy text classification pipelines for customer support in digital banking. By leveraging both a lightweight TF‑IDF + Linear SVC baseline and a more powerful BERT transformer model, FinQuery classifies incoming user queries into one of 77 fine‑grained banking intents (using the BANKING77 dataset).

This repository contains:

- A Jupyter notebook (`index.ipynb`) detailing the full data science workflow: data loading, preprocessing, feature engineering, model training, hyperparameter tuning, and evaluation for both Linear SVC and BERT pipelines.
- A Django project (`finquery_site`) with an app called `categorizer` that serves two endpoints (`bert_home` and `linear_svc_home`) for real‑time inference.

## Project Structure

```
finquery_site/
├── categorizer/                     # Django app for intent classification
│   ├── bert_intent_classifier.pth   # Trained BERT model (place here)
│   ├── tuned_linear_svc_model.pkl   # Trained LinearSVC model (place here)
│   ├── views.py                     # View functions for both classifiers
│   ├── urls.py                      # URL routes for the app
│   ├── utils.py                     # Preprocessing utilities for BERT pipeline
│   ├── linear_svc_utils.py          # Preprocessing utilities for Linear SVC pipeline
│   └── templates/
│       └── categorizer/
│           ├── bert_home.html       # Template for the BERT classifier interface
│           └── linear_svc_home.html # Template for the Linear SVC interface
├── finquery_site/                   # Django project configuration
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── index.ipynb                      # Notebook documenting the full analysis and model development
├── requirements.txt                 # Python dependencies
├── train.csv                        # Training data for BANKING77 intents
└── test.csv                         # Test data for BANKING77 intents
```

## Installation

Follow these steps to set up the project locally:

1. **Clone the repository**

   ```bash
   git clone https://github.com/Daniel-Muruthi/finquery.git
   cd finquery
   ```

2. **Create a virtual environment (named virtual):**

   ```bash
   python3 -m venv virtual
   ```

3. **Activate the environment:**

   macOS/Linux

   ```bash
   source virtual/bin/activate
   ```

   Windows (PowerShell)

   ```powershell
   .\virtual\Scripts\Activate.ps1
   ```

4. **Install libraries via the provided requirements.txt:**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Add your trained model files**

   Copy the following model artifacts into the categorizer/ directory (next to views.py):

   - `bert_intent_classifier.pth`
   - `tuned_linear_svc_model.pkl`

   Your categorizer/ folder should contain these files alongside the code:

   ```
   categorizer/
   ├── bert_intent_classifier.pth
   ├── tuned_linear_svc_model.pkl
   ├── views.py
   └── ...
   ```

6. **Apply database migrations:**

   ```bash
   python manage.py migrate
   ```

7. **Run the development server:**

   ```bash
   python manage.py runserver
   ```

8. **Access the app in your browser:**
   - BERT classifier: http://localhost:8000/categorizer/bert/
   - Linear SVC classifier: http://localhost:8000/categorizer/linear-svc/

## Usage

Enter a banking-related customer query into either interface to see the predicted intent label. The underlying pipelines include:

- **Linear SVC**: TF‑IDF vectorization + Linear SVC
- **BERT**: Fine‑tuned BERT model for sequence classification

## Technologies Used

- Python 3.x
- Django
- scikit-learn (TF‑IDF, LinearSVC, LogisticRegression, MultinomialNB)
- PyTorch & Transformers (Hugging Face BERT)
- pandas & NumPy (data handling)
- NLTK (stopwords, lemmatization)
- Matplotlib & WordCloud (EDA and visualization)

## Contact

For questions or feedback, please reach out via email:

adinomuruthi1@gmail.com
