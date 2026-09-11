# 🤖 BERT Sentiment Analysis on Twitter Data

An end-to-end **Sentiment Analysis** project using a fine-tuned **BERT (Bidirectional Encoder Representations from Transformers)** model to classify tweets as **Positive** or **Negative**.

The project covers data preprocessing, BERT fine-tuning, evaluation, error analysis, and deployment using **Streamlit**.

---

## 📌 Project Overview

Social media platforms contain a huge amount of text expressing opinions, emotions, and reactions.

This project uses the **Sentiment140 dataset** to build a binary sentiment classification model.

The fine-tuned BERT model predicts:

* 😊 **Positive**
* 😞 **Negative**

The trained model is deployed as an interactive **Streamlit web application**, where users can enter a sentence or tweet and receive a sentiment prediction with confidence.

---

## 📊 Dataset

### Sentiment140

The project uses the **Sentiment140** Twitter sentiment dataset.

The dataset contains approximately **1.6 million tweets**.

For this project, the sentiment labels were converted into:

| Label | Sentiment |
| ----- | --------- |
| `0`   | Negative  |
| `1`   | Positive  |

After cleaning, the dataset contained approximately:

* **631,500 Positive**
* **630,866 Negative**

---

## 🧠 Model

The project uses:

**BERT Base Uncased**

Architecture:

* 12 Transformer layers
* 12 attention heads
* Hidden size: 768
* Intermediate size: 3072
* Binary classification output

The BERT model was fine-tuned specifically for Twitter sentiment classification.

---

## ⚙️ Training Configuration

| Parameter               |               Value |
| ----------------------- | ------------------: |
| Pretrained Model        | `bert-base-uncased` |
| Training Samples        |             200,000 |
| Validation Samples      |              20,000 |
| Test Samples            |              20,000 |
| Maximum Sequence Length |                  64 |
| Batch Size              |                  16 |
| Learning Rate           |              `2e-5` |
| Weight Decay            |              `0.01` |
| Epochs                  |                   2 |
| Warmup                  |                 10% |

---

## 📈 Model Performance

### Training

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy |
| ----- | ---------: | -------------: | --------------: | ------------------: |
| 1     |     0.3849 |         82.82% |          0.3409 |              85.24% |
| 2     |     0.2701 |         89.18% |          0.3591 |              85.83% |

### Test Results

**Test Accuracy: 85.91%**

| Sentiment | Precision | Recall | F1-Score |
| --------- | --------: | -----: | -------: |
| Negative  |    84.91% | 87.27% |   86.07% |
| Positive  |    86.97% | 84.55% |   85.74% |

### Confusion Matrix

```text
                 Predicted
              Negative  Positive
Actual
Negative         8708      1270
Positive         1548      8474
```

---

## 🧹 Text Preprocessing

Unlike traditional NLP pipelines, the BERT model does not use aggressive preprocessing such as removing stopwords or punctuation.

The preprocessing mainly removes:

* URLs
* Twitter usernames
* Extra whitespace

Example:

```python
def clean_text_for_bert(text):
    text = str(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

The cleaned text is then passed directly to the **BERT tokenizer**.

---

## 🔤 Tokenization

The project uses the BERT tokenizer associated with `bert-base-uncased`.

The tokenizer:

* Converts text into BERT tokens
* Adds `[CLS]` and `[SEP]`
* Pads sequences to length 64
* Truncates longer sequences
* Generates an attention mask

Example:

```text
Input:
I absolutely love this product!

↓ BERT Tokenizer

[CLS] I absolutely love this product ! [SEP]
```

---

## 🏋️ Training Process

The overall workflow is:

```text
Sentiment140 Dataset
        ↓
Data Cleaning
        ↓
Label Conversion
        ↓
Train / Validation / Test Split
        ↓
BERT Tokenization
        ↓
BERT Fine-Tuning
        ↓
Validation
        ↓
Best Model Selection
        ↓
Test Evaluation
        ↓
Streamlit Deployment
```

---

## 💻 Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* BERT
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Matplotlib
* Seaborn

---

## 📁 Project Structure

```text
sentiment_analysis_using_bert/
│
├── app.py
├── best_bert_model.pt
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note:** The trained `best_bert_model.pt` file is approximately 387 MB and should not be committed directly to a normal GitHub repository. Use Git LFS or a model-hosting service if you want to host the model online.

---

## 🚀 Run the Project Locally

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### 2. Navigate to the project

```bash
cd sentiment_analysis_using_bert
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run Streamlit

```bash
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

---

## 🖥️ Streamlit Application

The application provides a simple interface where the user can enter a tweet or sentence.

Example:

```text
I absolutely love this product!
```

The application returns:

```text
😊 Positive

Confidence: 96.XX%
```

For negative text:

```text
This product is terrible and I hate it.
```

The application returns:

```text
😞 Negative
```

---

## 🧪 Model Challenge Testing

To evaluate the model beyond normal test data, it can be challenged using difficult real-world examples such as:

### Negation

```text
I don't like this.
```

### Sarcasm

```text
Great! Another three hours waiting for customer support.
```

### Mixed Sentiment

```text
The product is amazing, but the customer service is terrible.
```

### Slang

```text
This movie was lit!
```

### Spelling Variations

```text
Thisss product is amazinngg!!!
```

### Emoji

```text
I got the job! 😍🎉
```

These examples help identify weaknesses that may not be visible from the standard test accuracy alone.

---

## ⚠️ Limitations

Although the model achieves good performance, sentiment classification remains challenging.

Potential sources of errors include:

* Sarcasm
* Mixed sentiment
* Ambiguous statements
* Slang
* Misspellings
* Context-dependent meaning
* Noisy Twitter language
* Incorrect or ambiguous dataset labels

For example:

```text
Thanks for ruining my day 🙂
```

The literal words and emoji can make the sentiment difficult to classify correctly.

---

## 🔮 Future Improvements

Possible improvements include:

* Train BERT on a larger portion of the dataset
* Increase the number of training epochs with appropriate regularization
* Perform hyperparameter tuning
* Use more advanced transformer models such as RoBERTa or DeBERTa
* Build a larger manually labeled challenge dataset
* Perform detailed error analysis
* Add batch prediction for CSV files
* Deploy the application online
* Use Git LFS or a model-hosting platform for the trained model
* Add confidence visualization
* Extend classification to neutral sentiment

---

## 👨‍💻 Author

**Ranjith Kumar**

### Connect with me

* LinkedIn: https://www.linkedin.com/in/ranjithkumar975/
* GitHub: github.com/Ranjith-7

---

## ⭐ Acknowledgements

* Sentiment140 dataset
* Hugging Face Transformers
* PyTorch
* Streamlit

---

## 📜 License

This project is intended for educational and research purposes.
