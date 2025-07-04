
# 🤖 Cyberbully Detection Using Machine Learning

A Python-based system that identifies cyberbullying content in tweets using classic ML techniques. This repository offers end-to-end functionality from data preprocessing to model evaluation and visual insights.

---

## 📌 Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Project Structure](#project-structure)  
4. [Installation](#installation)  
5. [Usage](#usage)  
6. [Model & Data](#model--data)  
7. [Results & Visualization](#results--visualization)  
8. [Contributing](#contributing)  
9. [License](#license)

---

## Overview

Cyberbullying on social platforms has serious emotional and psychological impacts. This repository uses machine learning to detect abusive tweets, helping to flag harmful content automatically.

---

## Features

- ✅ Data preprocessing: tokenization, stop-word removal, and vectorization  
- ✅ Feature extraction through TF-IDF  
- ✅ Model training and performance analysis (Logistic Regression, SVM, etc.)  
- ✅ Intuitive visualizations of metrics and confusion matrices

---

## Project Structure

```text
├── cyberbullying_tweets.csv    # Dataset: labeled tweets
├── preprocess.py               # Cleaning and preprocessing pipeline
├── feature_extraction.py       # Vectorization (TF-IDF)
├── model.py                    # Model definitions & training routines
├── main.py                     # Orchestrates data flow + pipeline execution
├── visualization.py           # Generates performance plots
├── results/                    # Stores model outputs and artifacts
└── models/                     # Serialized model files
```

---

## Installation

1. **Clone the repo:**
   ```bash
   git clone https://github.com/Tanishq123467658/Cyberbully-Detection-Using-Machine-Learning-Techniques.git
   cd Cyberbully-Detection-Using-Machine-Learning-Techniques
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. Place `cyberbullying_tweets.csv` in the root directory (already included).
2. Run the full pipeline:
   ```bash
   python main.py
   ```
3. View generated plots in `results/` and models in `models/`.

**Optional:** Customize preprocessing options or model hyperparameters by editing relevant scripts.

---

## Model & Data

- **Dataset:** Labeled tweets (bullying vs. non-bullying).
- **Preprocessing:** Lowercasing, punctuation stripping, stop-word removal.
- **Features:** TF-IDF vectors representing text frequency.
- **Models:** Logistic Regression, SVM, etc. capable of binary classification.
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1–Score, Confusion Matrix.

---

## Results & Visualization

The pipeline produces:

- **`results/metrics.json`** – Key metrics per classifier.
- **`results/confusion_matrix.png`** – Confusion matrices.
- **`results/feature_importance.png`** – Insights into influential terms.
- **`models/`** – Saved `.pkl` model files ready for reuse.

Use **`visualization.py`** separately to regenerate visual outputs anytime.

---

## Contributing

Enhancements are welcome! Consider improving the project by:

- Adding advanced NLP (e.g., word embeddings, BERT)
- Incorporating deep learning models (CNN, LSTM)
- Extending dataset and fine-tuning preprocessing
- Adding a frontend or API deployment layer

To contribute, fork the repo, make changes, and submit a pull request!

---

## License

This project is licensed under the [MIT License](./LICENSE).  
Feel free to use, modify, and distribute as you wish! 💡

---

## 🧠 Contact

For questions or suggestions, reach out to **Tanishq** at **battultanishq@gmail.com** or open an issue here on GitHub.
