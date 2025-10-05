# Fake-news-ML

Fake News Dashboard

A simple web application that predicts whether a news article is Real or Fake using a trained deep learning model. Built with Flask, TensorFlow, and Keras.

📁 Project Structure
fake-news-dashboard/
│
├── app.py                # Flask app
├── model.h5              # Pre-trained fake news detection model
├── tokenizer.pkl         # Tokenizer used during model training
└── templates/
    └── index.html        # HTML template for the dashboard

⚙️ Features

User-friendly web interface to input news text.

Predicts whether the news is Real or Fake.

Shows prediction results instantly.

Can be extended to highlight keywords or improve UI.


📦 How It Works

User enters news text in the dashboard.

Flask backend preprocesses the text using the saved tokenizer.pkl.

Preprocessed text is passed into the model.h5.

Model predicts Real or Fake news.

Result is displayed on the dashboard.

📌 Notes

Make sure model.h5 and tokenizer.pkl are in the same folder as app.py.

The accuracy depends on your trained model. If predictions are off, retrain the model with more data.

For production, consider using Docker, Gunicorn, or NGINX.

🔮 Future Improvements

Add a confidence score for predictions.

Improve frontend with Bootstrap or Tailwind CSS.

Support multiple languages or news categories.

Deploy to Heroku, Vercel, or AWS.
