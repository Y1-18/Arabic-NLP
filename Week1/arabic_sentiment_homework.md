# 📚 Homework Assignment: Arabic Sentiment Analysis from Scratch

**Course:** Natural Language Processing  
**Due:** Two Weeks from Today  
**Points:** 100  
**Language:** Python 3.10+

---

> **Learning Objectives:** By the end of this assignment, you will understand how language models work at a fundamental level, how to preprocess Arabic text, how to build a Naïve Bayes classifier from scratch, and how to evaluate it against a modern library baseline.

---

## 📦 Dataset

You will work with the **Arabic Sentiment Twitter Corpus** — a real-world dataset of Arabic tweets labeled as positive or negative.

```python
from datasets import load_dataset

ds = load_dataset("arbml/Arabic_Sentiment_Twitter_Corpus")

# DatasetDict({
#     train: Dataset({
#         features: ['tweet', 'label'],
#         num_rows: 47000
#     })
#     test: Dataset({
#         features: ['tweet', 'label'],
#         num_rows: 11751
#     })
# })

train_data = ds["train"]
test_data  = ds["test"]
```

---

## 🗂️ Project Structure

Organize your submission as follows:

```
arabic_sentiment/
│
├── arabic_sentiment/          # Your main package
│   ├── __init__.py
│   ├── preprocessing.py       # Part 1
│   ├── language_model.py      # Part 2
│   ├── naive_bayes.py         # Part 3
│   └── evaluation.py          # Part 4
│
├── notebooks/
│   └── exploration.ipynb      # Optional: scratch work
│
├── main.py                    # Ties everything together
├── requirements.txt
└── README.md                  # Explain your choices!
```

> **Code Quality Rule:** All functions must have a **docstring**, **type hints**, and handle edge cases gracefully. You will lose points for spaghetti code.

---

## Part 1 — Arabic Text Preprocessing `(20 pts)`

Arabic text is noisy — especially on Twitter. Before building any model, you need to clean and normalize the text. This step is critical: **you will run the full pipeline both with and without preprocessing**, and compare the results.

### 1.1 Build the Preprocessor

Inside `preprocessing.py`, implement the following class skeleton:

```python
import re
from typing import List

class ArabicPreprocessor:
    """
    A preprocessing pipeline for Arabic Twitter text.
    
    Each method is a standalone transformation step.
    The `preprocess` method chains them together.
    """

    def remove_diacritics(self, text: str) -> str:
        """
        Remove Arabic diacritics (tashkeel/harakat).
        
        Arabic diacritics are Unicode characters in the range U+064B to U+065F.
        They represent short vowels and are usually absent in informal text,
        so we normalize by removing them.
        
        Hint: use re.sub with a Unicode range pattern.
        """
        # TODO: implement
        raise NotImplementedError

    def normalize_alef(self, text: str) -> str:
        """
        Normalize all Alef variants (أ إ آ ٱ) to plain Alef (ا).
        
        This reduces vocabulary sparsity because the same word
        may be written with different Alef forms on social media.
        """
        # TODO: implement
        raise NotImplementedError

    def normalize_teh_marbuta(self, text: str) -> str:
        """
        Normalize Teh Marbuta (ة) to Heh (ه).
        
        Some writers use these interchangeably, especially in informal text.
        """
        # TODO: implement
        raise NotImplementedError

    def remove_urls(self, text: str) -> str:
        """Remove HTTP/HTTPS URLs from text."""
        # TODO: implement
        raise NotImplementedError

    def remove_mentions(self, text: str) -> str:
        """Remove Twitter @mentions."""
        # TODO: implement
        raise NotImplementedError

    def remove_hashtags(self, text: str) -> str:
        """
        Remove the '#' symbol but KEEP the word.
        
        Hashtags often carry sentiment signal, so we keep the word
        while removing the special character.
        """
        # TODO: implement
        raise NotImplementedError

    def remove_punctuation_and_emojis(self, text: str) -> str:
        """
        Remove punctuation and emoji characters.
        
        Hint: Unicode ranges for emojis include U+1F300–U+1F9FF and others.
        """
        # TODO: implement
        raise NotImplementedError

    def remove_repeated_characters(self, text: str) -> str:
        """
        Normalize elongated words, e.g., 'جميييييل' → 'جميل'.
        
        Arabic writers often repeat characters for emphasis.
        Collapse any character repeated more than twice to a single character.
        """
        # TODO: implement
        raise NotImplementedError

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text by splitting on whitespace.
        
        After cleaning, a simple whitespace split is sufficient.
        Filter out any empty strings.
        
        Returns:
            A list of word tokens.
        """
        # TODO: implement
        raise NotImplementedError

    def preprocess(self, text: str, tokenize: bool = True):
        """
        Run the full preprocessing pipeline.
        
        Apply all steps in a sensible order, then optionally tokenize.
        
        Args:
            text:      Raw input string.
            tokenize:  If True, return List[str]; otherwise return cleaned str.
        
        Returns:
            List of tokens or a cleaned string.
        """
        # TODO: chain all steps
        raise NotImplementedError
```

### 1.2 Demonstrate the Effect

In `main.py`, print a before/after table for at least **5 example tweets** from the dataset showing how each preprocessing step changes the text.

```python
# Expected output (roughly):
# ┌─────────────────────────────────────┬───────────────────────────────┐
# │ Before                              │ After                         │
# ├─────────────────────────────────────┼───────────────────────────────┤
# │ @user أنا سعيييد جداً! 😊 #خير    │ انا سعيد جدا خير              │
# └─────────────────────────────────────┴───────────────────────────────┘
```

---

## Part 2 — N-gram Language Model `(25 pts)`

You will build a **bigram and trigram language model** from scratch using only Python's standard library (no NLTK, no spaCy for this part).

A language model assigns a probability to a sequence of words. A bigram model estimates:

$$P(w_n \mid w_{n-1})$$

and a trigram model estimates:

$$P(w_n \mid w_{n-2}, w_{n-1})$$

### 2.1 Implement the Language Model

Inside `language_model.py`:

```python
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Literal
import math

NGramOrder = Literal[2, 3]

class NgramLanguageModel:
    """
    A bigram or trigram language model with Laplace (add-1) smoothing.

    Attributes:
        n:       The order of the model (2 for bigram, 3 for trigram).
        vocab:   The set of all known tokens.
        counts:  Raw n-gram counts.
        context_counts: Counts of (n-1)-gram contexts.
    """

    def __init__(self, n: NGramOrder = 2):
        """
        Args:
            n: Order of the n-gram model. Must be 2 or 3.
        """
        # TODO: initialize data structures
        raise NotImplementedError

    def _extract_ngrams(
        self, tokens: List[str]
    ) -> List[Tuple[str, ...]]:
        """
        Extract all n-grams from a token list.
        
        Remember to add special <s> start and </s> end tokens.
        For a trigram model, prepend TWO <s> tokens.
        
        Example (bigram): ['a', 'b', 'c'] → [('<s>','a'), ('a','b'), ('b','c'), ('c','</s>')]
        """
        # TODO: implement
        raise NotImplementedError

    def train(self, corpus: List[List[str]]) -> None:
        """
        Train on a list of tokenized sentences.
        
        Args:
            corpus: A list of token lists (one per tweet/sentence).
        """
        # TODO: count n-grams and context counts, build vocab
        raise NotImplementedError

    def log_probability(self, ngram: Tuple[str, ...]) -> float:
        """
        Return the log probability of an n-gram using Laplace smoothing.
        
        Formula (Laplace):
            P(w | context) = (count(context, w) + 1) / (count(context) + |V|)
        
        Returns log base 2 of the probability.
        """
        # TODO: implement with smoothing
        raise NotImplementedError

    def sentence_log_probability(self, tokens: List[str]) -> float:
        """
        Return the total log probability of a tokenized sentence.
        
        This is the sum of log probabilities of each n-gram in the sentence.
        """
        # TODO: implement
        raise NotImplementedError

    def perplexity(self, corpus: List[List[str]]) -> float:
        """
        Compute perplexity on a held-out corpus.
        
        Perplexity = 2^(-average log probability per token)
        
        Lower perplexity = better model.
        """
        # TODO: implement
        raise NotImplementedError

    def generate(self, seed: List[str] = None, max_tokens: int = 20) -> str:
        """
        Generate a random sequence of tokens using the language model.
        
        Sample the next token proportionally to its probability given
        the current context, until </s> is generated or max_tokens is reached.
        
        Args:
            seed: Optional starting context (list of tokens).
        
        Returns:
            A generated string.
        """
        # TODO: implement (use random.choices with weights)
        raise NotImplementedError
```

### 2.2 Train and Evaluate

In `main.py`:

1. Train **two models**: one on raw text, one on preprocessed text.
2. Compute **perplexity** on the test set for all four combinations (bigram/trigram × raw/preprocessed).
3. Print a comparison table.
4. Generate **3 sample tweets** from each model and print them.

```python
# Expected comparison table (your numbers will differ):
# ┌─────────────────┬────────────┬─────────────┐
# │ Model           │ Raw PPL    │ Preprocessed│
# ├─────────────────┼────────────┼─────────────┤
# │ Bigram          │  ???       │  ???        │
# │ Trigram         │  ???       │  ???        │
# └─────────────────┴────────────┴─────────────┘
```

> **Reflection Question (written, 5 pts):** In your README, explain: why does preprocessing lower (or raise) perplexity? What does a lower perplexity actually mean for your model?

---

## Part 3 — Naïve Bayes Classifier `(30 pts)`

Now you will build a **Multinomial Naïve Bayes** classifier from scratch — no `sklearn` allowed for this part.

Naïve Bayes classifies a document $d$ as:

$$\hat{c} = \arg\max_{c} \log P(c) + \sum_{w \in d} \log P(w \mid c)$$

### 3.1 Implement the Classifier

Inside `naive_bayes.py`:

```python
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import math

Label = str   # "positive" or "negative"

class NaiveBayesClassifier:
    """
    Multinomial Naïve Bayes classifier for text.

    Supports Laplace (add-k) smoothing. Operates on pre-tokenized input.
    
    Attributes:
        k:               Smoothing parameter (default 1.0).
        class_log_priors: log P(c) for each class.
        word_log_likelihoods: log P(w | c) for each class and word.
        vocab:           All words seen during training.
    """

    def __init__(self, k: float = 1.0):
        # TODO: initialize
        raise NotImplementedError

    def train(
        self,
        documents: List[List[str]],
        labels: List[Label]
    ) -> None:
        """
        Estimate log priors and log likelihoods from training data.
        
        Steps:
            1. Count documents per class  → compute log priors.
            2. Concatenate all tokens per class → build per-class word counts.
            3. Apply Laplace smoothing → compute log likelihoods.
        
        Store smoothed values as log probabilities to avoid underflow.
        
        Args:
            documents: List of tokenized documents.
            labels:    Corresponding class label for each document.
        """
        # TODO: implement
        raise NotImplementedError

    def predict_one(self, tokens: List[str]) -> Label:
        """
        Predict the class of a single tokenized document.
        
        For unknown words (not in vocab), skip them — do not crash.
        
        Returns:
            The predicted label.
        """
        # TODO: implement
        raise NotImplementedError

    def predict(self, documents: List[List[str]]) -> List[Label]:
        """
        Predict classes for a list of tokenized documents.
        
        Returns:
            A list of predicted labels, one per document.
        """
        # TODO: implement using predict_one
        raise NotImplementedError

    def top_features(self, n: int = 20) -> Dict[Label, List[Tuple[str, float]]]:
        """
        Return the top-n most discriminative words per class.
        
        Discriminative score for word w and class c:
            score(w, c) = log P(w | c) - log P(w | other_c)
        
        Returns:
            Dict mapping each label to a sorted list of (word, score) tuples.
        """
        # TODO: implement
        raise NotImplementedError
```

### 3.2 Evaluate on 100 Samples

In `evaluation.py`, implement:

```python
from typing import List, Tuple

def accuracy(predictions: List[str], gold: List[str]) -> float:
    """Compute accuracy as the fraction of correct predictions."""
    # TODO: implement
    raise NotImplementedError

def precision_recall_f1(
    predictions: List[str],
    gold: List[str],
    positive_label: str = "positive"
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 for the positive class.
    
    Returns:
        (precision, recall, f1) as a tuple of floats.
    """
    # TODO: implement without sklearn
    raise NotImplementedError

def confusion_matrix_str(
    predictions: List[str],
    gold: List[str],
    labels: List[str]
) -> str:
    """
    Return a pretty-printed confusion matrix string.
    
    Example output:
              Pred Pos  Pred Neg
    True Pos     42        8
    True Neg      5       45
    """
    # TODO: implement
    raise NotImplementedError
```

Then in `main.py`, run this evaluation:

```python
# 1. Sample exactly 100 examples from the test set (use random.seed(42))
# 2. Preprocess them
# 3. Predict with your classifier
# 4. Print accuracy, precision, recall, F1
# 5. Print the confusion matrix
# 6. Print 5 correct and 5 incorrect predictions with the original tweet text
```

### 3.3 Compare: With vs. Without Preprocessing

Run your full pipeline **twice** — once on raw tokenized text, once on preprocessed text — and report all metrics for both. Discuss the difference in your README.

---

## Part 4 — Bonus: Library Baseline `(+15 pts)`

Now that you have built everything from scratch, compare your implementation to a modern library solution.

```python
# You MAY use sklearn for this part only.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# TODO: Build a pipeline using TfidfVectorizer + MultinomialNB
# Train it on the full training set (preprocessed)
# Evaluate on the same 100 test samples you used in Part 3
# Print a sklearn classification_report
# Compare to your from-scratch model — which is better and why?
```

> **Reflection Question (written):** What does TF-IDF capture that a simple bag-of-words Naïve Bayes does not? Why might one outperform the other on Twitter data?

---

## Part 5 — Code Quality `(25 pts)`

Your code will be graded on the following rubric:

| Criterion | Points | What we look for |
|---|---|---|
| Type hints on all functions | 5 | `def fn(x: str) -> List[str]` |
| Docstrings on all classes & functions | 5 | Purpose, Args, Returns |
| No global mutable state | 5 | Use classes and parameters |
| Modular, reusable structure | 5 | Each file has one responsibility |
| Informative variable names | 5 | No `x`, `tmp`, `data2` |

### Code Quality Examples

**❌ Bad:**
```python
def nb(d, l):
    c = {}
    for i in range(len(d)):
        if l[i] not in c: c[l[i]] = []
        c[l[i]].append(d[i])
    return c
```

**✅ Good:**
```python
def group_documents_by_label(
    documents: List[List[str]],
    labels: List[str]
) -> Dict[str, List[List[str]]]:
    """
    Group tokenized documents by their class label.

    Args:
        documents: List of tokenized documents.
        labels:    Corresponding label for each document.

    Returns:
        A dict mapping each label to the list of documents with that label.
    """
    grouped: Dict[str, List[List[str]]] = defaultdict(list)
    for document, label in zip(documents, labels):
        grouped[label].append(document)
    return dict(grouped)
```

---

## 🏁 Putting It All Together — `main.py`

Your `main.py` should run the entire pipeline end-to-end and print a clean summary. Here is the skeleton:

```python
import random
from datasets import load_dataset

from arabic_sentiment.preprocessing import ArabicPreprocessor
from arabic_sentiment.language_model import NgramLanguageModel
from arabic_sentiment.naive_bayes import NaiveBayesClassifier
from arabic_sentiment.evaluation import accuracy, precision_recall_f1, confusion_matrix_str


def load_data():
    """Load the dataset and return train/test splits."""
    # TODO
    ...


def run_preprocessing_demo(preprocessor: ArabicPreprocessor, raw_tweets: list) -> None:
    """Print a before/after table for 5 sample tweets."""
    # TODO
    ...


def run_language_model(train_tokens: list, test_tokens: list) -> None:
    """Train bigram and trigram LMs and print perplexity comparison."""
    # TODO
    ...


def run_naive_bayes(
    train_docs: list,
    train_labels: list,
    test_sample_docs: list,
    test_sample_labels: list,
    raw_test_tweets: list
) -> None:
    """Train NB classifier and print evaluation results."""
    # TODO
    ...


def run_library_baseline(
    train_docs: list,
    train_labels: list,
    test_sample_docs: list,
    test_sample_labels: list
) -> None:
    """Train sklearn pipeline and compare to scratch implementation."""
    # TODO
    ...


def main() -> None:
    random.seed(42)
    
    print("=" * 60)
    print("Arabic Sentiment Analysis — Full Pipeline")
    print("=" * 60)

    # Step 1: Load data
    # Step 2: Preprocessing demo
    # Step 3: Language model comparison
    # Step 4: Naïve Bayes (raw vs preprocessed)
    # Step 5: Library baseline
    ...


if __name__ == "__main__":
    main()
```

---

## 📝 README Requirements

Your `README.md` must include:

1. **How to run:** a single command to reproduce all results.
2. **Results table:** all perplexity and classification metrics.
3. **Reflections** (required for full credit):
   - Does preprocessing help? For which metric and why?
   - Which n-gram order worked better and why?
   - How does your scratch implementation compare to sklearn's?
   - What surprised you about Arabic Twitter text?

---

## 📋 Submission Checklist

- [ ] `preprocessing.py` — all methods implemented and tested
- [ ] `language_model.py` — bigram & trigram, perplexity, generation
- [ ] `naive_bayes.py` — train, predict, top_features
- [ ] `evaluation.py` — accuracy, precision, recall, F1, confusion matrix
- [ ] `main.py` — full pipeline runs cleanly with `python main.py`
- [ ] Results reported both with and without preprocessing
- [ ] 100-sample evaluation printed with correct/incorrect examples
- [ ] Code quality rubric satisfied (type hints, docstrings, naming)
- [ ] `README.md` with reflections
- [ ] `requirements.txt` with pinned versions

---

## 🔗 Useful References

- [Arabic NLP Wikipedia](https://en.wikipedia.org/wiki/Arabic_natural_language_processing) — overview of Arabic NLP challenges  
- [Speech and Language Processing, Ch. 3 (N-grams)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — Jurafsky & Martin  
- [Speech and Language Processing, Ch. 4 (Naïve Bayes)](https://web.stanford.edu/~jurafsky/slp3/4.pdf) — Jurafsky & Martin  
- [Unicode Arabic Block](https://unicode.org/charts/PDF/U0600.pdf) — for preprocessing character ranges  
- [Hugging Face Datasets Docs](https://huggingface.co/docs/datasets/)

---

*Good luck! Building something from scratch is always harder — and always more rewarding — than calling a library.*
