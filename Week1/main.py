"""
Arabic Sentiment Analysis — Full Pipeline

Runs all four assignment parts end-to-end:
    1. Preprocessing demo
    2. Language model comparison (bigram/trigram × raw/preprocessed)
    3. Naïve Bayes evaluation (raw vs preprocessed)
    4. Library baseline (sklearn TF-IDF + MultinomialNB)
"""

import random
from typing import List, Tuple

from datasets import load_dataset

from arabic_sentiment.preprocessing import ArabicPreprocessor
from arabic_sentiment.language_model import NgramLanguageModel
from arabic_sentiment.naive_bayes import NaiveBayesClassifier
from arabic_sentiment.evaluation import (
    accuracy,
    confusion_matrix_str,
    precision_recall_f1,
)

# --------------------------------------------------------------------------- #
# Label constants (dataset uses integer labels: 0 = negative, 1 = positive)   #
# --------------------------------------------------------------------------- #
LABEL_MAP = {0: "negative", 1: "positive"}
POSITIVE_LABEL = "positive"
CLASSES = ["positive", "negative"]


# --------------------------------------------------------------------------- #
# Data loading                                                                 #
# --------------------------------------------------------------------------- #


def load_data() -> Tuple[list, list, list, list]:
    """
    Load the Arabic Sentiment Twitter Corpus and return raw splits.

    Returns:
        Tuple of (train_tweets, train_labels, test_tweets, test_labels)
        where labels are string values ("positive" / "negative").
    """
    print("Loading dataset …")
    dataset = load_dataset("arbml/Arabic_Sentiment_Twitter_Corpus")

    train_tweets = [row["tweet"] for row in dataset["train"]]
    train_labels = [LABEL_MAP[row["label"]] for row in dataset["train"]]
    test_tweets = [row["tweet"] for row in dataset["test"]]
    test_labels = [LABEL_MAP[row["label"]] for row in dataset["test"]]

    print(
        f"  Train: {len(train_tweets):,} tweets | "
        f"Test: {len(test_tweets):,} tweets\n"
    )
    return train_tweets, train_labels, test_tweets, test_labels


# --------------------------------------------------------------------------- #
# Part 1 — Preprocessing demo                                                  #
# --------------------------------------------------------------------------- #


def run_preprocessing_demo(
    preprocessor: ArabicPreprocessor,
    raw_tweets: List[str],
) -> None:
    """
    Print a before/after table for 5 sample tweets.

    Args:
        preprocessor: Trained ArabicPreprocessor instance.
        raw_tweets:   List of raw tweet strings to sample from.
    """
    print("=" * 70)
    print("PART 1 — Preprocessing Demo")
    print("=" * 70)

    sample_tweets = raw_tweets[:5]
    col_width = 33

    header = f"{'Before':<{col_width}} │ {'After'}"
    separator = "─" * col_width + "─┼─" + "─" * col_width
    print(header)
    print(separator)

    for tweet in sample_tweets:
        cleaned = preprocessor.preprocess(tweet, tokenize=False)
        before = tweet.replace("\n", " ")[:col_width - 1]
        after = str(cleaned)[:col_width - 1]
        print(f"{before:<{col_width}} │ {after}")

    print()


# --------------------------------------------------------------------------- #
# Part 2 — Language model comparison                                           #
# --------------------------------------------------------------------------- #


def _tokenize_corpus(
    tweets: List[str],
    preprocessor: ArabicPreprocessor,
    use_preprocessing: bool,
) -> List[List[str]]:
    """
    Tokenize a list of tweets either with or without preprocessing.

    Args:
        tweets:            Raw tweet strings.
        preprocessor:      Preprocessor instance.
        use_preprocessing: If True, apply full pipeline; else split on whitespace.

    Returns:
        List of token lists.
    """
    if use_preprocessing:
        return [preprocessor.preprocess(tweet, tokenize=True) for tweet in tweets]
    return [tweet.split() for tweet in tweets]


def run_language_model(
    train_tweets: List[str],
    test_tweets: List[str],
    preprocessor: ArabicPreprocessor,
) -> None:
    """
    Train bigram and trigram LMs on raw and preprocessed text.
    Print a perplexity comparison table and generate sample tweets.

    Args:
        train_tweets: Raw training tweet strings.
        test_tweets:  Raw test tweet strings.
        preprocessor: ArabicPreprocessor instance.
    """
    print("=" * 70)
    print("PART 2 — N-gram Language Model")
    print("=" * 70)

    print("Tokenizing corpora …")
    raw_train = _tokenize_corpus(train_tweets, preprocessor, use_preprocessing=False)
    raw_test = _tokenize_corpus(test_tweets, preprocessor, use_preprocessing=False)
    prep_train = _tokenize_corpus(train_tweets, preprocessor, use_preprocessing=True)
    prep_test = _tokenize_corpus(test_tweets, preprocessor, use_preprocessing=True)

    configurations = [
        ("Bigram",  2, raw_train,  raw_test,  "Raw"),
        ("Bigram",  2, prep_train, prep_test, "Preprocessed"),
        ("Trigram", 3, raw_train,  raw_test,  "Raw"),
        ("Trigram", 3, prep_train, prep_test, "Preprocessed"),
    ]

    # Use first 5 000 sentences for speed; remove slice for full training
    results = {}
    trained_models = {}
    for model_name, order, train_corpus, test_corpus, text_type in configurations:
        key = (model_name, text_type)
        print(f"  Training {model_name} ({text_type}) …")
        lm = NgramLanguageModel(n=order)
        lm.train(train_corpus[:5000])
        ppl = lm.perplexity(test_corpus[:1000])
        results[key] = ppl
        trained_models[key] = lm

    # Perplexity table
    col = 16
    print(f"\n{'Model':<{col}} {'Raw PPL':>{col}} {'Preprocessed PPL':>{col}}")
    print("─" * (col * 3 + 2))
    for model_name in ("Bigram", "Trigram"):
        raw_ppl = results[(model_name, "Raw")]
        prep_ppl = results[(model_name, "Preprocessed")]
        print(f"{model_name:<{col}} {raw_ppl:>{col}.2f} {prep_ppl:>{col}.2f}")

    # Generated samples
    print("\nGenerated tweets (preprocessed bigram model):")
    bigram_model = trained_models[("Bigram", "Preprocessed")]
    for sample_num in range(1, 4):
        generated_text = bigram_model.generate(max_tokens=15)
        print(f"  [{sample_num}] {generated_text}")

    print("\nGenerated tweets (preprocessed trigram model):")
    trigram_model = trained_models[("Trigram", "Preprocessed")]
    for sample_num in range(1, 4):
        generated_text = trigram_model.generate(max_tokens=15)
        print(f"  [{sample_num}] {generated_text}")

    print()


# --------------------------------------------------------------------------- #
# Part 3 — Naïve Bayes classifier                                              #
# --------------------------------------------------------------------------- #


def _run_single_nb_experiment(
    train_docs: List[List[str]],
    train_labels: List[str],
    test_docs: List[List[str]],
    test_labels: List[str],
    raw_test_tweets: List[str],
    label: str,
) -> None:
    """
    Train, evaluate, and print results for one NB configuration.

    Args:
        train_docs:      Tokenized training documents.
        train_labels:    Training labels.
        test_docs:       Tokenized test documents (100 samples).
        test_labels:     Test labels (100 samples).
        raw_test_tweets: Original (un-tokenized) test tweets for display.
        label:           Description string e.g. "Raw" or "Preprocessed".
    """
    classifier = NaiveBayesClassifier(k=1.0)
    classifier.train(train_docs, train_labels)
    predictions = classifier.predict(test_docs)

    acc = accuracy(predictions, test_labels)
    precision, recall, f1_score = precision_recall_f1(
        predictions, test_labels, positive_label=POSITIVE_LABEL
    )
    matrix_str = confusion_matrix_str(predictions, test_labels, labels=CLASSES)

    print(f"\n  ── {label} ──")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1_score:.4f}")
    print(f"\n{matrix_str}\n")

    # Top features
    top = classifier.top_features(n=10)
    for class_label, word_scores in top.items():
        top_words = ", ".join(word for word, _ in word_scores[:5])
        print(f"  Top words for '{class_label}': {top_words}")

    # Correct and incorrect examples
    correct_examples = []
    incorrect_examples = []
    for idx, (predicted, expected) in enumerate(zip(predictions, test_labels)):
        entry = (raw_test_tweets[idx], expected, predicted)
        if predicted == expected and len(correct_examples) < 5:
            correct_examples.append(entry)
        elif predicted != expected and len(incorrect_examples) < 5:
            incorrect_examples.append(entry)

    print("\n  5 Correct predictions:")
    for tweet, expected, predicted in correct_examples:
        print(f"    [{expected}→{predicted}] {tweet[:60]}")

    print("\n  5 Incorrect predictions:")
    for tweet, expected, predicted in incorrect_examples:
        print(f"    [{expected}→{predicted}] {tweet[:60]}")


def run_naive_bayes(
    train_tweets: List[str],
    train_labels: List[str],
    test_sample_tweets: List[str],
    test_sample_labels: List[str],
    preprocessor: ArabicPreprocessor,
) -> None:
    """
    Train NB classifier and print evaluation results for both
    raw and preprocessed text.

    Args:
        train_tweets:        Raw training tweet strings.
        train_labels:        Training labels.
        test_sample_tweets:  100 sampled raw test tweets.
        test_sample_labels:  Corresponding labels for the 100 samples.
        preprocessor:        ArabicPreprocessor instance.
    """
    print("=" * 70)
    print("PART 3 — Naïve Bayes Classifier")
    print("=" * 70)

    # Raw (whitespace tokenization)
    raw_train_docs = [tweet.split() for tweet in train_tweets]
    raw_test_docs = [tweet.split() for tweet in test_sample_tweets]

    # Preprocessed
    prep_train_docs = [
        preprocessor.preprocess(tweet, tokenize=True) for tweet in train_tweets
    ]
    prep_test_docs = [
        preprocessor.preprocess(tweet, tokenize=True) for tweet in test_sample_tweets
    ]

    _run_single_nb_experiment(
        raw_train_docs,
        train_labels,
        raw_test_docs,
        test_sample_labels,
        test_sample_tweets,
        label="Raw text",
    )

    _run_single_nb_experiment(
        prep_train_docs,
        train_labels,
        prep_test_docs,
        test_sample_labels,
        test_sample_tweets,
        label="Preprocessed text",
    )
    print()


# --------------------------------------------------------------------------- #
# Part 4 — Library baseline                                                    #
# --------------------------------------------------------------------------- #


def run_library_baseline(
    train_tweets: List[str],
    train_labels: List[str],
    test_sample_tweets: List[str],
    test_sample_labels: List[str],
    preprocessor: ArabicPreprocessor,
) -> None:
    """
    Train sklearn TF-IDF + MultinomialNB pipeline and compare to
    the from-scratch implementation.

    Args:
        train_tweets:        Raw training tweet strings.
        train_labels:        Training labels.
        test_sample_tweets:  100 sampled raw test tweets.
        test_sample_labels:  Corresponding labels for the 100 samples.
        preprocessor:        ArabicPreprocessor instance.
    """
    print("=" * 70)
    print("PART 4 — Library Baseline (sklearn)")
    print("=" * 70)

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics import classification_report
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import Pipeline
    except ImportError:
        print("  sklearn is not installed. Skipping bonus part.\n")
        return

    # Preprocess to strings for TF-IDF
    prep_train_strings = [
        preprocessor.preprocess(tweet, tokenize=False) for tweet in train_tweets
    ]
    prep_test_strings = [
        preprocessor.preprocess(tweet, tokenize=False) for tweet in test_sample_tweets
    ]

    sklearn_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50_000)),
        ("nb",    MultinomialNB(alpha=1.0)),
    ])

    print("  Training sklearn TF-IDF + MultinomialNB pipeline …")
    sklearn_pipeline.fit(prep_train_strings, train_labels)
    sklearn_predictions = sklearn_pipeline.predict(prep_test_strings)

    print("\n  sklearn classification report:")
    report = classification_report(
        test_sample_labels,
        sklearn_predictions,
        target_names=CLASSES,
    )
    print(report)
    print()


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #


def main() -> None:
    """
    Run the full Arabic Sentiment Analysis pipeline end-to-end.

    Steps:
        1. Load dataset
        2. Preprocessing demo
        3. Language model comparison
        4. Naïve Bayes (raw vs preprocessed)
        5. Library baseline
    """
    random.seed(42)

    print("=" * 70)
    print("Arabic Sentiment Analysis — Full Pipeline")
    print("=" * 70)
    print()

    # Step 1: Load data
    train_tweets, train_labels, test_tweets, test_labels = load_data()

    # Step 2: Preprocessing demo
    preprocessor = ArabicPreprocessor()
    run_preprocessing_demo(preprocessor, train_tweets)

    # Step 3: Language model comparison
    run_language_model(train_tweets[:10_000], test_tweets[:2_000], preprocessor)

    # Step 4: Sample exactly 100 test examples (reproducible)
    random.seed(42)
    test_indices = random.sample(range(len(test_tweets)), 100)
    test_sample_tweets = [test_tweets[idx] for idx in test_indices]
    test_sample_labels = [test_labels[idx] for idx in test_indices]

    run_naive_bayes(
        train_tweets,
        train_labels,
        test_sample_tweets,
        test_sample_labels,
        preprocessor,
    )

    # Step 5: Library baseline
    run_library_baseline(
        train_tweets,
        train_labels,
        test_sample_tweets,
        test_sample_labels,
        preprocessor,
    )

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
