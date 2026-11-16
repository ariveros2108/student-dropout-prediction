import pickle
import re

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

DATA_PATH = 'data/data.csv'
MODEL_PATH = 'model.bin'

# Parámetros elegidos en notebook.ipynb (mejor AUC en validación)
XGB_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 4,
    'random_state': 1,
}


def clean_column(name):
    """'Curricular units 1st sem (grade)' -> 'curricular_units_1st_sem_grade'"""
    name = name.strip().lower().replace("'", "")
    return re.sub(r'[^a-z0-9]+', '_', name).strip('_')


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, sep=';', encoding='utf-8-sig')
    df.columns = [clean_column(c) for c in df.columns]
    # El CSV de UCI llama 'Target' a la columna; algunas copias la llaman 'Status'
    df = df.rename(columns={'status': 'target'})
    df['dropout'] = (df['target'] == 'Dropout').astype(int)
    del df['target']
    return df


def main():
    df = load_data()
    df_full_train, df_test = train_test_split(
        df, test_size=0.2, random_state=1, stratify=df['dropout']
    )

    y_full_train = df_full_train.pop('dropout').values
    y_test = df_test.pop('dropout').values

    pipeline = make_pipeline(
        DictVectorizer(sparse=False),
        XGBClassifier(**XGB_PARAMS),
    )
    pipeline.fit(df_full_train.to_dict(orient='records'), y_full_train)

    y_pred = pipeline.predict_proba(df_test.to_dict(orient='records'))[:, 1]
    print(f'AUC en test: {roc_auc_score(y_test, y_pred):.4f}')

    with open(MODEL_PATH, 'wb') as f_out:
        pickle.dump(pipeline, f_out)
    print(f'Modelo guardado en {MODEL_PATH}')


if __name__ == '__main__':
    main()
