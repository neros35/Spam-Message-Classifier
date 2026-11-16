import csv
import io
import os
import urllib.request
import zipfile
from pathlib import Path

from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATA_FILE = DATA_DIR / "SMSSpamCollection"

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
MIRROR_URL = "https://raw.githubusercontent.com/veermshah/cs6350-sms-spam-dataset/main/SMSSpamCollection"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024  # Limit incoming form payloads to 16 KB.


def download_dataset() -> None:
    """Download the original UCI dataset, with a public mirror as fallback."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    errors = []

    try:
        request_obj = urllib.request.Request(
            UCI_ZIP_URL,
            headers={"User-Agent": "Mozilla/5.0 (compatible; SpamClassifier/1.0)"},
        )
        with urllib.request.urlopen(request_obj, timeout=30) as response:
            archive_bytes = response.read()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            dataset_bytes = archive.read("SMSSpamCollection")
        DATA_FILE.write_bytes(dataset_bytes)
        return
    except Exception as exc:  # Try the mirror if the UCI download is blocked.
        errors.append(f"UCI download: {exc}")

    try:
        request_obj = urllib.request.Request(
            MIRROR_URL,
            headers={"User-Agent": "Mozilla/5.0 (compatible; SpamClassifier/1.0)"},
        )
        with urllib.request.urlopen(request_obj, timeout=30) as response:
            DATA_FILE.write_bytes(response.read())
        return
    except Exception as exc:
        errors.append(f"Dataset mirror: {exc}")

    raise RuntimeError(
        "Could not download the SMS dataset. Put the UCI file named "
        "'SMSSpamCollection' in the project's data/ folder and restart the container. "
        + " | ".join(errors)
    )


def load_dataset():
    """Read the tab-separated UCI file: label<TAB>message."""
    if not DATA_FILE.exists():
        download_dataset()

    messages, labels = [], []
    with DATA_FILE.open("r", encoding="utf-8", errors="replace", newline="") as file:
        reader = csv.reader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            if len(row) >= 2:
                label = row[0].strip().lower()
                message = "\t".join(row[1:]).strip()
                if label in {"ham", "spam"} and message:
                    labels.append(label)
                    messages.append(message)

    if len(messages) < 100 or len(set(labels)) != 2:
        raise RuntimeError(
            f"Dataset was loaded, but only {len(messages)} valid rows were found. "
            "Check that data/SMSSpamCollection is the original UCI dataset file."
        )

    return messages, labels


def train_model():
    messages, labels = load_dataset()

    x_train, x_test, y_train, y_test = train_test_split(
        messages,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )

    # TF-IDF converts the cleaned text into weighted word and bigram features.
    # The pipeline applies the same vectorizer to training and new messages.
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            ("naive_bayes", MultinomialNB(alpha=0.5)),
        ]
    )

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])

    metrics = {
        "accuracy": accuracy * 100,
        "total": len(messages),
        "train_total": len(x_train),
        "test_total": len(x_test),
        "ham_count": labels.count("ham"),
        "spam_count": labels.count("spam"),
        "tn": int(matrix[0][0]),
        "fp": int(matrix[0][1]),
        "fn": int(matrix[1][0]),
        "tp": int(matrix[1][1]),
    }
    return model, metrics


try:
    classifier, model_metrics = train_model()
    training_error = None
except Exception as error:
    classifier, model_metrics = None, None
    training_error = str(error)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    message = ""
    probability = None
    error = training_error

    if request.method == "POST" and classifier is not None:
        message = request.form.get("message", "").strip()
        if not message:
            error = "Enter a message before clicking Classify."
        else:
            prediction = classifier.predict([message])[0]
            probabilities = classifier.predict_proba([message])[0]
            spam_index = list(classifier.classes_).index("spam")
            probability = float(probabilities[spam_index]) * 100

    return render_template(
        "index.html",
        metrics=model_metrics,
        prediction=prediction,
        message=message,
        probability=probability,
        error=error,
    )


if __name__ == "__main__":
    # Listen on all interfaces so Podman can publish the container port.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), debug=False)
