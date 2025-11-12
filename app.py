from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import os
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import logging
from pathlib import Path
import threading
import time
import re

# ------------------- Flask App -------------------
app = Flask(__name__)

# ------------------- Logging Setup -------------------
LOG_PATH = Path(__file__).resolve().parent / "api.log"
logging.Formatter.converter = time.localtime
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@app.before_request
def log_request_info():
    logger.info(f" Received {request.method} request at {request.path}")
    if request.is_json:
        logger.info(f"Request JSON: {request.get_json()}")
    else:
        logger.info("Request has no JSON body")

@app.after_request
def log_response_info(response):
    logger.info(f" Responded with {response.status} for {request.path}")
    return response

# ------------------- NLTK Setup -------------------
nltk_data_dir = Path("/usr/local/nltk_data")
nltk.data.path.append(str(nltk_data_dir))

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
splitter = RegexpTokenizer(r'\w+')
synonym_cache = {}

# ------------------- Path Configuration -------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Dataset"
DATA_DIR.mkdir(exist_ok=True)

DATA_PATH = DATA_DIR / "diseasesymp_updated.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
ENCODER_PATH = BASE_DIR / "label_encoder.pkl"

# ------------------- Helper: Synonyms -------------------
def synonyms(term):
    """Return cached WordNet-based synonyms for a term."""
    if term in synonym_cache:
        return synonym_cache[term]
    synonym_set = set()
    for syn in wordnet.synsets(term):
        for lemma in syn.lemmas():
            synonym_set.add(lemma.name().replace('_', ' '))
    synonym_cache[term] = synonym_set
    return synonym_set

# ------------------- Model Training -------------------
def train_and_save_model():
    """Retrain model from dataset and save updated .pkl files."""
    start_time = time.time()
    logger.info(" Retraining model started...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, encoding='latin1')

    X = df.drop(columns=['label_dis'])
    Y = df['label_dis']

    encoder = LabelEncoder()
    Y_encoded = encoder.fit_transform(Y)

    model = LogisticRegression(max_iter=200, n_jobs=-1)
    model.fit(X, Y_encoded)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    duration = round(time.time() - start_time, 2)
    logger.info(f" Model retrained successfully in {duration}s.")

    return model, encoder, list(X.columns)

# ------------------- Background Retraining -------------------
retraining_lock = threading.Lock()
is_retraining = False

def retrain_in_background():
    """Background thread for retraining the model."""
    global lr_model, encoder, dataset_symptoms, is_retraining
    try:
        logger.info(" Background retraining thread started.")
        new_model, new_encoder, new_symptoms = train_and_save_model()
        with retraining_lock:
            lr_model, encoder, dataset_symptoms = new_model, new_encoder, new_symptoms
        logger.info(" Background retraining completed and model updated in memory.")
    except Exception as e:
        logger.exception(f" Retraining failed: {e}")
    finally:
        is_retraining = False

# ------------------- Initial Load -------------------
if MODEL_PATH.exists() and ENCODER_PATH.exists() and DATA_PATH.exists():
    logger.info(" Loading existing model, encoder, and dataset...")
    lr_model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    df = pd.read_csv(DATA_PATH, encoding='latin1')
    dataset_symptoms = list(df.drop(columns=['label_dis']).columns)
else:
    logger.info(" Model or dataset not found — training new model...")
    lr_model, encoder, dataset_symptoms = train_and_save_model()

# ------------------- Prediction Endpoint -------------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_symptoms = data.get("symptoms", [])

    if not user_symptoms:
        return jsonify({"error": "No symptoms provided."}), 400

    # Step 1: Preprocess
    processed_user_symptoms = []
    for sym in user_symptoms:
        sym = sym.strip().replace('_', ' ').replace('-', ' ').replace("'", '')
        sym = ' '.join([lemmatizer.lemmatize(word) for word in splitter.tokenize(sym)])
        processed_user_symptoms.append(sym)

    # Step 2: Synonym expansion
    expanded_symptoms = []
    for user_sym in processed_user_symptoms:
        words = user_sym.split()
        str_sym = set(words)
        for word in words:
            str_sym.update(synonyms(word))
        expanded_symptoms.append(' '.join(str_sym))

    # Step 3: Match symptoms
    found_symptoms = set()
    for data_sym in dataset_symptoms:
        for user_sym in expanded_symptoms:
            if data_sym.replace('_', ' ') in user_sym:
                found_symptoms.add(data_sym)

    # Step 4: Input vector
    sample_x = [0] * len(dataset_symptoms)
    for val in found_symptoms:
        if val in dataset_symptoms:
            sample_x[dataset_symptoms.index(val)] = 1

    # Step 5: Predict top 5 diseases
    # Step 5: Predict top 5 diseases with normalized high-confidence probabilities
    input_df = pd.DataFrame([sample_x], columns=dataset_symptoms)
    prediction = lr_model.predict_proba(input_df)[0]# Step 5: Predict top 5 diseases with normalized high-confidence probabilities
    input_df = pd.DataFrame([sample_x], columns=dataset_symptoms)
    prediction = lr_model.predict_proba(input_df)[0]
    k = 5
    diseases = encoder.classes_
    topk = prediction.argsort()[-k:][::-1]

    topk_probs = prediction[topk]
    total_prob = sum(topk_probs)
    normalized_probs = (topk_probs / total_prob * 100) if total_prob > 0 else [0] * len(topk_probs)
    boosted_probs = [round((p * 0.9) + 10, 2) if p < 90 else round(p, 2) for p in normalized_probs]
    
    #topk_dict = {diseases[t]: prob for t, prob in zip(topk, boosted_probs)}
    topk_dict = {diseases[t]: float(prob) for t, prob in zip(topk, boosted_probs)}

    logger.info(f" Predicted diseases: {topk_dict}")
    return jsonify({"predictions": topk_dict})

# ------------------- Receive + Async Retrain Endpoint -------------------
@app.route('/receive', methods=['POST'])
def receive_data():
    data = request.json
    logger.info(f" Received new training data: {data}")

    symptoms = data.get("symptoms", [])
    doctor_diseases = data.get("final_diagnosis_by_doctor", [])

    if not symptoms or not doctor_diseases:
        return jsonify({"error": "Missing symptoms or diagnosis."}), 400

    # Load or create dataset
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, encoding='latin1')
    else:
        df = pd.DataFrame(columns=['label_dis'])

    # Normalize symptom names
    def normalize_symptom(sym):
        return sym.strip().lower().replace('-', '_').replace(' ', '_').replace("'", "")

    normalized_symptoms = [normalize_symptom(s) for s in symptoms]

    # Ensure all columns exist
    for symptom in normalized_symptoms:
        if symptom not in df.columns:
            if "label_dis" in df.columns:
                insert_idx = df.columns.get_loc("label_dis")
                df.insert(insert_idx, symptom, 0)
            else:
                df[symptom] = 0

    if "label_dis" not in df.columns:
        df["label_dis"] = ""

    # ------------------ Normalize Disease Names ------------------
    def normalize_disease_name(disease):
        """
        Clean and normalize disease name into a consistent 'Title Case' format.
        Handles mixed cases, underscores, punctuation, numbers, and extra spaces.
        Example:
          "__CHICKEN-pox_19" -> "Chicken Pox 19"
          "covid-19" -> "Covid 19"
          "  malaria!! " -> "Malaria"
        """
        if not isinstance(disease, str):
            return "Unknown"

        # Remove leading/trailing spaces
        clean = disease.strip()

        # Remove unwanted characters: underscores, dashes, equals, punctuations, etc.
        clean = re.sub(r"[_\-=\+*/\\,.;:!?@#%^&(){}[\]<>~`|]", " ", clean)

        # Replace multiple spaces with single space
        clean = re.sub(r"\s+", " ", clean)

        # Remove non-alphanumeric characters except spaces (for safety)
        clean = re.sub(r"[^a-zA-Z0-9 ]", "", clean)

        # Convert to lowercase first, then title case for readability
        clean = clean.lower().title()

        # Extra cleanup — remove accidental double spaces
        clean = re.sub(r"\s+", " ", clean).strip()

        return clean if clean else "Unknown"

    # ------------------ Add New Rows (Allow Duplicates) ------------------
    new_rows = []
    for disease in doctor_diseases:
        normalized_disease = normalize_disease_name(disease)

        new_row = {col: 0 for col in df.columns}
        for symptom in normalized_symptoms:
            if symptom in df.columns:
                new_row[symptom] = 1
        new_row["label_dis"] = normalized_disease
        new_rows.append(new_row)

    # Append all rows (duplicates allowed)
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    # Safe save
    temp_path = DATA_PATH.with_suffix(".csv.tmp")
    df.to_csv(temp_path, index=False, encoding='latin1')
    os.replace(temp_path, DATA_PATH)

    # Start background retraining
    global is_retraining
    if not is_retraining:
        is_retraining = True
        thread = threading.Thread(target=retrain_in_background, daemon=True)
        thread.start()
        logger.info(" Retraining triggered in background thread.")
    else:
        logger.info(" Retraining already in progress — new data queued for next cycle.")

    return jsonify({
        "status": "success",
        "rows_added": len(new_rows),
        "normalized_symptoms": normalized_symptoms,
        "message": "Data saved. Model retraining in background."
    })

# ------------------- Health Check -------------------
@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "running", "message": "Flask API is active"})

# ------------------- Run App -------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
