# BERT & GPT-2 Studio: Privacy-Sensitive Text Detection + Text Generation

A Streamlit web app with two Transformer models:

- **BERT (encoder-only):** detects whether a text is **Sensitive** (contains or asks for private information) or **Non-Sensitive**, in **Urdu, English and Roman Urdu**.
- **GPT-2 (decoder-only):** continues the beginning of an English sentence.

## Live Demo
https://YOUR-APP-NAME.streamlit.app

## Models on Hugging Face
| Model | Architecture | Task | Link |
|---|---|---|---|
| BERT (fine-tuned `bert-base-multilingual-cased`) | Encoder-only | Privacy detection (0 = Non-Sensitive, 1 = Sensitive) | https://huggingface.co/Maryam033/bert-privacy-detector |
| GPT-2 (fine-tuned on AG News) | Decoder-only | English news-style text generation | https://huggingface.co/Maryam033/gpt2-text-generation |

## Project Structure
```
bert-gpt2-privacy-app/
├── app.py                       # Streamlit app (BERT + GPT-2)
├── requirements.txt             # for the Streamlit app
├── requirements-training.txt    # for the Colab notebooks
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml              # app theme
├── notebooks/
│   ├── Part_A_BERT_Privacy_Detection.ipynb
│   └── Part_B_GPT2_Improved_NewsGeneration.ipynb
├── data/
│   ├── README.md                # sources, cleaning, citation
│   ├── ag_news_generation_prompts.csv   # 6,865 test articles (prompts)
│   └── ag_news_finetune_train.csv       # 12,000 training articles
├── dataset/                     # optional: privacy_dataset.csv (synthetic data from the BERT notebook)
└── huggingface/                 # model cards and requirements for the two HF repos
    ├── bert-privacy-detector/
    └── gpt2-text-generation/
```

## How the parts connect
```
Google Colab (train / prepare models)
      |  push_to_hub
      v
Hugging Face Hub (stores the models)  <----- app.py loads them by name
      |
GitHub (stores code: app.py, notebooks, README)
      |  Streamlit Cloud pulls the repo
      v
Streamlit app (public web app)
```

## BERT: data and method
Three sources were combined: a privacy-masking dataset (parquet), `urdu_ner.txt`, and template-based synthetic sentences in Urdu, English and Roman Urdu covering 10 kinds of personal data (phone, CNIC, email, address, bank account, password, OTP, card, passport, date of birth).

- Train: 150,207 | Validation: 16,660 | Test: 31,065 (real data) + 181 (synthetic)
- 4 epochs, learning rate 3e-5, batch size 16, max length 96, T4 GPU

## Results (BERT)
| Test set | Class | Precision | Recall | F1 |
|---|---|---|---|---|
| Real data (31,065) | Non-Sensitive | 0.948 | 0.852 | 0.898 |
| Real data (31,065) | Sensitive | 0.997 | 0.999 | 0.998 |
| Synthetic (181) | Overall accuracy | | | 0.995 |

Accuracy on the real test set is 99.6%, but about 98% of those samples are *Sensitive*, so the Non-Sensitive scores are the fairer measure.

## GPT-2: data and method
- **Dataset:** [AG News](https://huggingface.co/datasets/ag_news) (clean English news in 4 categories), cleaned and saved in `data/`. See `data/README.md`.
- **Prompts:** the first 10 words of 6,865 test articles.
- **Fine-tuning (optional):** `gpt2` trained for 2 epochs on 12,000 training articles. Test articles are never used for training.
- **Better decoding:** repetition penalty 1.2, no repeated 3-grams, top-k 50, top-p 0.92, temperature 0.8.
- **Measured:** perplexity on 300 held-out texts before and after fine-tuning, and a repetition score (distinct-3). Fill in your numbers below.

| Model | Perplexity (lower is better) |
|---|---|
| Pre-trained GPT-2 | ... |
| Fine-tuned GPT-2 | ... |

## Run Locally
```bash
git clone https://github.com/marybaloch31-coder/bert-gpt2-privacy-app.git
cd bert-gpt2-privacy-app
pip install -r requirements.txt
streamlit run app.py
```

## Encoder vs Decoder
| | BERT (encoder) | GPT-2 (decoder) |
|---|---|---|
| Purpose | Understand text | Generate text |
| Attention | Bidirectional | Causal (previous tokens only) |
| Output here | Sensitive / Non-Sensitive | Continued text |

## Limitations
- The BERT training data is heavily imbalanced (mostly Sensitive), so the model leans towards "Sensitive".
- Sentences that announce personal data without giving it can be misclassified.
- Labeling rules are simple (for example person or location names from NER data count as Sensitive).
- GPT-2 output can be factually wrong and changes on every run. It works best with news-style prompts.
- Use fake sample data in the app. This is a class project, not a privacy tool.

## Author
MARYAM ZUBAIR AHMED
