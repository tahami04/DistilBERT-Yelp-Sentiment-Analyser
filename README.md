**DistilBERT-Yelp-Sentiment-Analyser**
Fine-tuned DistilBERT transformer model for multi-class sentiment classification on the Yelp Review Full dataset. This project builds an end-to-end pipeline from raw review text to a trained sentiment classification model, visualizations, and evaluation.

**Project Overview**
This project uses the yelp_review_full dataset from Hugging Face and fine-tunes DistilBERT, a lightweight version of BERT, for multi-class sentiment classification (1 to 5 stars). The model classifies customer reviews into five sentiment categories and provides a comprehensive training pipeline with data preprocessing, visualization, and evaluation.


**Dataset**
* Source: Hugging Face Datasets – yelp_review_full
* Task: Sentiment classification (5 classes: 1–5 stars)
* Size: 650K training samples, 50K test samples

**Technologies Used**
* Python
* Hugging Face Transformers
* PyTorch
* Scikit-learn
* NLTK
* Google Colab
* TensorBoard (optional)

**How to Run**
**1. Clone the Repository**
git clone https://github.com/your-username/distilbert-yelp-sentiment.git
cd distilbert-yelp-sentiment

**2. Set Up Environment**
pip install -r requirements.txt

**3. Run the Notebook**
Launch the Jupyter or upload to Google Colab:

jupyter notebook yelp_sentiment_analysis.ipynb

**Model Performance**
DistilBERT Classification Report (5 Epochs):
Class	Precision	Recall	F1-Score
1 Star	0.77	0.77	0.77
2 Stars	0.60	0.58	0.59
3 Stars	0.59	0.60	0.59
4 Stars	0.58	0.54	0.56
5 Stars	0.71	0.78	0.74
Accuracy	-	-	0.65

**Load Trained Model**

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained("path_to_folder")
tokenizer = DistilBertTokenizerFast.from_pretrained("path_to_folder")
**
Visualizations Included**
* Word clouds for each sentiment class
* Keyword frequency bar charts
* Bigram analysis per sentiment
* Classification report & accuracy

**Future Improvements**
* Use a larger BERT model (e.g., bert-base-uncased)
* Add hyperparameter tuning
* Implement inference API with FastAPI or Streamlit
* Handle class imbalance
* Explore cross-domain transfer learning

**License**
This project is open-source under the MIT License.

**Acknowledgments**
* Hugging Face Datasets
* Transformers Library
* Yelp Open Dataset
