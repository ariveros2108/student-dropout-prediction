import pickle

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

MODEL_PATH = 'model.bin'

with open(MODEL_PATH, 'rb') as f_in:
    pipeline = pickle.load(f_in)


class Student(BaseModel):
    marital_status: int
    application_mode: int
    application_order: int
    course: int
    daytime_evening_attendance: int
    previous_qualification: int
    previous_qualification_grade: float
    nacionality: int
    mothers_qualification: int
    fathers_qualification: int
    mothers_occupation: int
    fathers_occupation: int
    admission_grade: float
    displaced: int
    educational_special_needs: int
    debtor: int
    tuition_fees_up_to_date: int
    gender: int
    scholarship_holder: int
    age_at_enrollment: int
    international: int
    curricular_units_1st_sem_credited: int
    curricular_units_1st_sem_enrolled: int
    curricular_units_1st_sem_evaluations: int
    curricular_units_1st_sem_approved: int
    curricular_units_1st_sem_grade: float
    curricular_units_1st_sem_without_evaluations: int
    curricular_units_2nd_sem_credited: int
    curricular_units_2nd_sem_enrolled: int
    curricular_units_2nd_sem_evaluations: int
    curricular_units_2nd_sem_approved: int
    curricular_units_2nd_sem_grade: float
    curricular_units_2nd_sem_without_evaluations: int
    unemployment_rate: float
    inflation_rate: float
    gdp: float

    # Ejemplo que aparece en /docs (un estudiante del set de test que desertó)
    model_config = ConfigDict(json_schema_extra={'examples': [{
        'marital_status': 1, 'application_mode': 1, 'application_order': 3,
        'course': 9500, 'daytime_evening_attendance': 1,
        'previous_qualification': 1, 'previous_qualification_grade': 130.0,
        'nacionality': 1, 'mothers_qualification': 3,
        'fathers_qualification': 1, 'mothers_occupation': 2,
        'fathers_occupation': 5, 'admission_grade': 133.8, 'displaced': 0,
        'educational_special_needs': 0, 'debtor': 0,
        'tuition_fees_up_to_date': 1, 'gender': 0, 'scholarship_holder': 0,
        'age_at_enrollment': 19, 'international': 0,
        'curricular_units_1st_sem_credited': 0,
        'curricular_units_1st_sem_enrolled': 7,
        'curricular_units_1st_sem_evaluations': 0,
        'curricular_units_1st_sem_approved': 0,
        'curricular_units_1st_sem_grade': 0.0,
        'curricular_units_1st_sem_without_evaluations': 0,
        'curricular_units_2nd_sem_credited': 0,
        'curricular_units_2nd_sem_enrolled': 8,
        'curricular_units_2nd_sem_evaluations': 0,
        'curricular_units_2nd_sem_approved': 0,
        'curricular_units_2nd_sem_grade': 0.0,
        'curricular_units_2nd_sem_without_evaluations': 0,
        'unemployment_rate': 7.6, 'inflation_rate': 2.6, 'gdp': 0.32,
    }]})


class Prediction(BaseModel):
    dropout_probability: float
    dropout: bool


app = FastAPI(title='Student Dropout Prediction')


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict', response_model=Prediction)
def predict(student: Student):
    proba = float(pipeline.predict_proba([student.model_dump()])[0, 1])
    return Prediction(dropout_probability=round(proba, 4), dropout=proba >= 0.5)
