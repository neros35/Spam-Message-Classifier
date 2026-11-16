# Spam Message Classifier (Podman / Red Hat UBI)

A Flask web application that classifies SMS text as **spam** or **ham** using TF-IDF features and a Multinomial Naive Bayes model. It displays held-out test accuracy and a confusion matrix.

## Project structure

```text
spam-message-classifier/
├── Containerfile
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── (SMSSpamCollection is downloaded here if it is not present)
└── templates/
    └── index.html
```

## Run with Podman

Run these commands from the project directory in the Red Hat classroom terminal.

```bash
podman build -t spam-classifier .
podman run -d --name spam-app -p 8080:8080 spam-classifier
podman ps
```

Open the classroom's forwarded port / preview for **8080**. If the classroom provides a port-forwarding panel, expose port 8080 there.

To view application logs:

```bash
podman logs -f spam-app
```

To stop and remove the container:

```bash
podman stop spam-app
podman rm spam-app
```

To rebuild after changing code:

```bash
podman stop spam-app
podman rm spam-app
podman build -t spam-classifier .
podman run -d --name spam-app -p 8080:8080 spam-classifier
```

## Dataset download / offline classroom fallback

On first startup, the app attempts to download the UCI SMS Spam Collection automatically. The dataset contains 5,574 labelled SMS messages. If the classroom blocks outbound downloads, download the dataset on a machine with internet access, extract the file named `SMSSpamCollection`, and copy it into this project's `data/` directory. Then rebuild and run the image again.

If `curl` and `unzip` are available in the classroom, you can download it directly:

```bash
mkdir -p data
curl -L 'https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip' -o /tmp/sms-spam.zip
python3 -c "import zipfile; zipfile.ZipFile('/tmp/sms-spam.zip').extract('SMSSpamCollection', 'data')"
```

Then build and run as above. The application reads `data/SMSSpamCollection` and does not need to download the dataset again.

## Useful Podman checks

```bash
podman images
podman ps -a
podman port spam-app
podman inspect spam-app
```

If port 8080 is already in use, use another host port, for example `-p 8081:8080`, and open port 8081 in the classroom preview.

## What to explain in a demo / viva

- **Ham** means a legitimate message; **spam** means an unwanted/promotional or fraudulent message.
- **TF-IDF** converts text into numerical features, giving higher weight to informative terms and lower weight to common terms.
- **Naive Bayes** estimates the class probability from the message's word features using Bayes' theorem and a conditional-independence assumption.
- **Accuracy** is the proportion of test messages classified correctly.
- **Confusion matrix** uses rows for actual labels and columns for predicted labels:
  - TN: actual ham predicted ham.
  - FP: actual ham predicted spam.
  - FN: actual spam predicted ham.
  - TP: actual spam predicted spam.
- The model is evaluated on a separate stratified 20% test split, not on the training data.

## Dataset attribution

Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84. Licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).
