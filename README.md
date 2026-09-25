# Sentiment Analysis using Machine Learning

A binary sentiment classification project using Natural Language Processing (NLP), TF-IDF feature extraction, and classical machine learning models.

## Project Overview

This project classifies movie-review sentences/snippets as **positive** or **negative** sentiment. The workflow includes exploratory data analysis, text preprocessing, TF-IDF vectorization, model training, evaluation, and prediction on new text.

## Dataset

The project uses the **Cornell Sentence Polarity Dataset v1.0**, containing 5,331 positive and 5,331 negative processed sentences/snippets (10,662 samples in total). The original dataset was introduced by Bo Pang and Lillian Lee in ACL 2005.

Source: Cornell Movie-Review Data  
https://www.cs.cornell.edu/people/pabo/movie-review-data/

The repository stores the two files used in the project as:

- `data/positive.txt`
- `data/negative.txt`

## Methodology

1. Load positive and negative text samples.
2. Assign sentiment labels.
3. Explore the class distribution.
4. Preprocess text:
   - Lowercasing
   - Special-character removal
   - Stopword removal
   - Lemmatization
5. Split data into training and test sets.
6. Convert text into numerical features using TF-IDF.
7. Train:
   - Logistic Regression
   - Multinomial Naive Bayes
8. Evaluate using accuracy, precision, recall, F1-score, and confusion matrix.
9. Test the trained model on unseen sentences.

## Models

### Logistic Regression
A linear classifier used for binary sentiment classification.

### Multinomial Naive Bayes
A probabilistic classifier commonly used for text classification tasks.

## Results

Using the reproducible pipeline included in the notebook:

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 0.776 |
| Multinomial Naive Bayes | 0.772 |

The exact values may vary if the preprocessing or train-test split is changed.

## Repository Structure

```text
sentiment-analysis-ml/
├── README.md
├── sentiment_analysis.ipynb
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── positive.txt
│   └── negative.txt
│
└── results/
    ├── sentiment_distribution.png
    └── confusion_matrix.png
```

## Technologies

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Matplotlib
- Jupyter Notebook
- NLP
- TF-IDF
- Logistic Regression
- Multinomial Naive Bayes

## How to Run

```bash
git clone <your-repository-url>
cd sentiment-analysis-ml
pip install -r requirements.txt
jupyter notebook sentiment_analysis.ipynb
```

Run the notebook cells from top to bottom.

## Dataset Attribution

The dataset is the Cornell Sentence Polarity Dataset v1.0 by Bo Pang and Lillian Lee. Please follow the original dataset's citation requirements when using it in academic or published work.

