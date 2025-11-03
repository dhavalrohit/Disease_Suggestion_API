# Disease Suggestion  API

This API gives disease suggestion based on given set of symptoms using a trained **Logistic Regression model**. It uses **WordNet-based synonym expansion**, **lemmatization** and Vectorization: Symptom presence binary encoding.

##  Installation & Setup Instructions
1.Python Version: 3.10.0

2.Open the terminal and run the following commands:

```
pip install flask pandas nltk scikit-learn waitress
```
## Run the App
```
python app.py
```
## API Endpoints
### 1.Post /predict
```
http://127.0.0.1:6000/predict
```
### Example Input/Request Body (JSON)
```
{
  "symptoms": ["fever", "headache", "body pain"]
}
```
### Example Output
```
{
  "predictions": {
    "Dengue": 85.76,
    "Malaria": 10.44,
    "Typhoid": 2.33,
    "Flu": 1.12,
    "COVID-19": 0.35
  }
}
```
### POST /receive
Used to receive and store new symptoms->disease data to training set and for training model.
### Input/Request Body (JSON):
```
{
  "symptoms": ["fever", "joint pain", "rash"],
  "final_diagnosis_by_doctor": ["Dengue"]
}
```
### Output/Response:
```
{
  "status": "success",
  "rows_added": 1,
  "received_data": {
    "symptoms": ["fever", "joint pain", "rash"],
    "final_diagnosis_by_doctor": ["Dengue"]
  }
}
```
