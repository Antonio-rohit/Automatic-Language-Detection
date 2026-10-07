<<<<<<< HEAD
# Automatic-Language-Detection
=======
# 🌍 AI Language Detection System

An AI-powered language detection system that automatically identifies the language of input text using **Character-Level TF-IDF** and a **Linear Support Vector Machine (Linear SVM)**.

The main goal of this project is to provide automatic source-language detection for a future multilingual translation system, eliminating the need for users to manually select the input language.

---
## 📚 Dataset

This project uses the **Language Detection** dataset available on Kaggle.

**Dataset:** Language Detection  
**Source:** Kaggle  
**Author:** Basil B2S

🔗 **Dataset:**  
https://www.kaggle.com/datasets/basilb2s/language-detection

The dataset contains text samples labeled according to their respective languages.

The dataset was used for:

- Training the language classification model
- Testing model performance
- Comparing different character n-gram configurations
- Evaluating language classification performance

---

## 🚀 Features

- 🌍 Automatic language detection
- ⚡ Live detection while typing
- 🔤 Character-level TF-IDF feature extraction
- 🤖 Linear SVM classification
- 📊 Top language predictions
- 🧹 Text preprocessing
- 📈 Model evaluation using Accuracy and Macro F1
- 🖥️ Interactive Streamlit web interface

---

## 🧠 How It Works

The system follows this pipeline:

```text
User Input
    ↓
Text Cleaning
    ↓
Character-Level TF-IDF
    ↓
Linear SVM
    ↓
Language Prediction
    ↓
Detected Language
````

### 1. Text Preprocessing

Input text is cleaned before being passed to the model.

The preprocessing includes:

* Removing numbers
* Removing unnecessary special characters
* Removing extra spaces
* Converting text to lowercase

### 2. TF-IDF Feature Extraction

The system uses **character-level TF-IDF** instead of word-level TF-IDF.

The selected character n-gram range is:

```text
2–5 characters
```

For example:

```text
Hello
```

can produce character patterns such as:

```text
He
Hel
Hell
el
ell
ello
...
```

These character patterns help the model identify language-specific writing patterns.

### 3. Linear SVM

A **Linear Support Vector Machine (LinearSVC)** is used as the classifier.

The model learns language-specific patterns from the TF-IDF representation and predicts the most likely language for new text.

---

## 🌐 Supported Languages

The current dataset contains the following languages:

* Arabic
* Danish
* Dutch
* English
* French
* German
* Greek
* Hindi
* Italian
* Kannada
* Malayalam
* Portuguese
* Russian
* Spanish
* Swedish
* Tamil
* Turkish

---

## 📊 Model Performance

Three different character n-gram configurations were evaluated.

| Character N-Gram |   Accuracy |   Macro F1 | Features |
| ---------------- | ---------: | ---------: | -------: |
| **2–5**          | **99.17%** | **99.34%** |  485,839 |
| 2–6              |     99.08% |     99.16% |  918,967 |
| 3–5              |     98.93% |     99.05% |  476,060 |

The **2–5 character n-gram configuration** was selected because it achieved the best performance while using significantly fewer features than the 2–6 configuration.

### Final Model

```text
Feature Extraction:
Character TF-IDF

N-Gram Range:
2–5

Classifier:
Linear SVM

Accuracy:
99.17%

Macro F1:
99.34%
```

---

## 🖥️ Web Application

The application is built using **Streamlit**.

Users can enter text into the interface and the system automatically detects the language.

Example:

```text
Input:
Bonjour, comment allez-vous?

Output:
🇫🇷 Detected Language: French
```

Another example:

```text
Input:
नमस्ते, आप कैसे हैं?

Output:
🇮🇳 Detected Language: Hindi
```

---

## 📁 Project Structure

```text
language-detection/
│
├── app.py
├── language_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

### Files

| File                   | Description               |
| ---------------------- | ------------------------- |
| `app.py`               | Streamlit web application |
| `language_model.pkl`   | Trained Linear SVM model  |
| `tfidf_vectorizer.pkl` | Trained TF-IDF vectorizer |
| `requirements.txt`     | Python dependencies       |
| `README.md`            | Project documentation     |
| `.gitignore`           | Git ignored files         |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Antonio-rohit/Automatic-Language-Detection.git
```

### 2. Navigate to the project

```bash
cd language-detection
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 📦 Requirements

The main dependencies are:

```text
streamlit
streamlit-keyup
scikit-learn
joblib
numpy
```

---

## 🔬 Model Development

The model was developed using the following workflow:

```text
Dataset
   ↓
Data Cleaning
   ↓
Duplicate Removal
   ↓
Train/Test Split
   ↓
Character TF-IDF
   ↓
Linear SVM
   ↓
Model Evaluation
   ↓
Hyperparameter Comparison
   ↓
Final Model
```

The dataset was divided into training and testing sets using an 80/20 split with stratification.

---

## 📈 Evaluation Metrics

The model was evaluated using:

### Accuracy

Measures the percentage of correctly classified samples.

```text
Accuracy =
Correct Predictions / Total Predictions
```

### Precision

Measures how many predictions for a particular language were actually correct.

### Recall

Measures how many samples belonging to a language were correctly identified.

### F1-Score

The harmonic mean of precision and recall.

```text
F1 = 2 × (Precision × Recall)
     ---------------------------
       Precision + Recall
```

### Macro F1

Macro F1 calculates the F1-score independently for each language and then takes the average.

This is useful because the dataset contains different numbers of samples for different languages.

---

## 🛠️ Technologies Used

* Python
* Scikit-learn
* NumPy
* Joblib
* Streamlit
* Streamlit-Keyup
* TF-IDF
* Linear SVM
* Git
* GitHub

---

## 🔮 Future Improvements

This language detector is the first component of a larger multilingual translation system.

Future versions may include:

```text
Automatic Language Detection
          ↓
   Target Language Selection
          ↓
    Translation Model
          ↓
    Translated Text
```

Planned improvements include:

* 🌐 Integration with a multilingual translation model
* 🔄 Automatic source-language detection
* 🗣️ Support for additional languages
* 📱 Improved responsive UI
* 🎤 Speech-to-text input
* 🔊 Text-to-speech output
* 📄 Translation history
* ⚡ Model optimization
* 🧠 Transformer-based language detection comparison

---

## 🎯 Project Goal

The ultimate goal is to build a multilingual translation system where users **do not need to manually select the source language**.

Instead:

```text
User enters text
       ↓
Language automatically detected
       ↓
User selects target language
       ↓
Text translated
```

This makes the translation process simpler and more user-friendly.

---

## 👨‍💻 Author

**Rohit Bharat Bharwade**

Artificial Intelligence & Machine Learning
D. Y. Patil College of Engineering and Technology, Kolhapur

---

## 📜 License

This project is intended for educational and research purposes.



