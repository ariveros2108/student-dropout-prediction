import requests

URL = 'http://localhost:9696/predict'

student = {
    'marital_status': 1, 'application_mode': 17, 'application_order': 1,
    'course': 9670, 'daytime_evening_attendance': 1,
    'previous_qualification': 1, 'previous_qualification_grade': 133.1,
    'nacionality': 1, 'mothers_qualification': 1, 'fathers_qualification': 37,
    'mothers_occupation': 4, 'fathers_occupation': 9, 'admission_grade': 96.0,
    'displaced': 1, 'educational_special_needs': 0, 'debtor': 0,
    'tuition_fees_up_to_date': 1, 'gender': 0, 'scholarship_holder': 1,
    'age_at_enrollment': 25, 'international': 0,
    'curricular_units_1st_sem_credited': 0,
    'curricular_units_1st_sem_enrolled': 6,
    'curricular_units_1st_sem_evaluations': 8,
    'curricular_units_1st_sem_approved': 6,
    'curricular_units_1st_sem_grade': 11.29,
    'curricular_units_1st_sem_without_evaluations': 0,
    'curricular_units_2nd_sem_credited': 0,
    'curricular_units_2nd_sem_enrolled': 6,
    'curricular_units_2nd_sem_evaluations': 10,
    'curricular_units_2nd_sem_approved': 5,
    'curricular_units_2nd_sem_grade': 12.2,
    'curricular_units_2nd_sem_without_evaluations': 0,
    'unemployment_rate': 15.5, 'inflation_rate': 2.8, 'gdp': -4.06,
}

response = requests.post(URL, json=student, timeout=10)
response.raise_for_status()
print(response.json())
