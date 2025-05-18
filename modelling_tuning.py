import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import logging

# Konfigurasi logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Definisikan nama folder dan file dataset
data_folder = 'mental_health_preprocessing'
X_train_path = f'{data_folder}/X_train.csv'
X_test_path = f'{data_folder}/X_test.csv'
y_train_path = f'{data_folder}/y_train.csv'
y_test_path = f'{data_folder}/y_test.csv'

def load_data(X_train_path, X_test_path, y_train_path, y_test_path):
    """Fungsi untuk memuat data dari file CSV."""
    logging.info("Memulai proses memuat data.")
    try:
        X_train = pd.read_csv(X_train_path)
        X_test = pd.read_csv(X_test_path)
        y_train = pd.read_csv(y_train_path)['MH_final'].values.ravel()
        y_test = pd.read_csv(y_test_path)['MH_final'].values.ravel()
        logging.info("Data berhasil dimuat.")
        return X_train, X_test, y_train, y_test
    except FileNotFoundError as e:
        logging.error(f"Error: File not found - {e}")
        return None, None, None, None

def train_and_evaluate_tuned_model(X_train, X_test, y_train, y_test):
    """Fungsi untuk melatih dan mengevaluasi model RandomForestClassifier dengan hyperparameter tuning."""
    if X_train is None:
        logging.error("Data training tidak tersedia. Proses pelatihan dibatalkan.")
        return None

    logging.info("Memulai proses hyperparameter tuning menggunakan GridSearchCV.")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 3, 5]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')

    with mlflow.start_run():
        mlflow.set_tag("model_name", "RandomForestClassifier")
        mlflow.set_tag("tuning_method", "GridSearchCV")

        # Manual logging parameter terbaik
        logging.info(f"Parameter terbaik yang ditemukan: {best_params}")
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)

        # Manual logging metrik
        logging.info(f"Metrik hasil tuning: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1 Score={f1:.4f}")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Manual logging model terbaik
        mlflow.sklearn.log_model(best_model, "best_random_forest_model")
        logging.info("Model terbaik berhasil dicatat ke MLflow.")

        print("Hasil Tuning Model Terbaik:")
        print(f"Parameter Terbaik: {best_params}")
        print(f"Akurasi: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("\nModel terbaik, parameter, dan metrik telah dicatat ke MLflow Tracking UI (secara lokal).")

        return best_model

    if __name__ == "__main__":
        # Set nama eksperimen MLflow
        mlflow.set_experiment("Mental Health Prediction with Tuning")

        # Memuat data
        X_train, X_test, y_train, y_test = load_data(X_train_path, X_test_path, y_train_path, y_test_path)

        # Melatih dan mengevaluasi model dengan tuning
        trained_model = train_and_evaluate_tuned_model(X_train, X_test, y_train, y_test)