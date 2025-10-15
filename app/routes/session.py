# Necesary imports
import os
import re
import time
import json
import csv

from pathlib import Path
import os
import pandas as pd
import traceback
import re
<<<<<<< HEAD
import requests
=======
from flask import Flask, request, Response, send_file
>>>>>>> 42a70612727088340cf95589066fb593eb246472

# Local imports from app
from app.services.storage import save_file_locally
from app.models.session import Session

# from app.services import database as db
from app.services import gaze_tracker


# Constants
ALLOWED_EXTENSIONS = {"txt", "webm"}
COLLECTION_NAME = "session"

# Initialize Flask app
app = Flask(__name__)


# def allowed_file(filename):
#     return '.' in filename and \
#         filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# def create_session():
#     # # Get files from request
#     if 'webcamfile' not in request.files or 'screenfile' not in request.files:
#         return Response('Error: Files not found on the request', status=400, mimetype='application/json')

#     webcam_file = request.files['webcamfile']
#     screen_file = request.files['screenfile']
#     title = request.form['title']
#     description = request.form['description']
#     website_url = request.form['website_url']
#     user_id = request.form['user_id']
#     calib_points = json.loads(request.form['calib_points'])
#     iris_points = json.loads(request.form['iris_points'])
#     timestamp = time.time()
#     session_id = re.sub(r"\s+", "", f'{timestamp}{title}')

#     # Check if extension is valid
#     if webcam_file and allowed_file(webcam_file.filename) and screen_file and allowed_file(screen_file.filename):
#         webcam_url = save_file_locally(webcam_file, f'/{session_id}')
#         screen_url = save_file_locally(screen_file, f'/{session_id}')
#     else:
#         return Response('Error: Files do not follow the extension guidelines', status=400, mimetype='application/json')

#     # Save session on database
#     session = Session(
#         id=session_id,
#         title=title,
#         description=description,
#         user_id=user_id,
#         created_date=timestamp,
#         website_url=website_url,
#         screen_record_url=screen_url,
#         webcam_record_url=webcam_url,
#         heatmap_url='',
#         calib_points=calib_points,
#         iris_points=iris_points
#     )

#     db.create_document(COLLECTION_NAME, session_id, session.to_dict())

#     # Generate csv dataset of calibration points
#     os.makedirs(
#         f'{Path().absolute()}/public/training/{session_id}/', exist_ok=True)
#     csv_file = f'{Path().absolute()}/public/training/{session_id}/train_data.csv'
#     csv_columns = ['timestamp', 'left_iris_x', 'left_iris_y',
#                    'right_iris_x', 'right_iris_y', 'mouse_x', 'mouse_y']
#     try:
#         with open(csv_file, 'w') as csvfile:
#             writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
#             writer.writeheader()
#             for data in calib_points:
#                 writer.writerow(data)
#     except IOError:
#         print("I/O error")

#     # Generate csv of iris points of session
#     os.makedirs(
#         f'{Path().absolute()}/public/sessions/{session_id}/', exist_ok=True)
#     csv_file = f'{Path().absolute()}/public/sessions/{session_id}/session_data.csv'
#     csv_columns = ['timestamp', 'left_iris_x', 'left_iris_y',
#                    'right_iris_x', 'right_iris_y']
#     try:
#         with open(csv_file, 'w') as csvfile:
#             writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
#             writer.writeheader()
#             for data in iris_points:
#                 writer.writerow(data)
#     except IOError:
#         print("I/O error")

#     return Response('Session Created!', status=201, mimetype='application/json')


# def get_user_sessions():
#     user_id = request.args.__getitem__('user_id')
#     field = u'user_id'
#     op = u'=='
#     docs = db.get_documents(COLLECTION_NAME, field, op, user_id)
#     sessions = []
#     for doc in docs:
#         sessions.append(
#             doc.to_dict()
#         )
#     return Response(json.dumps(sessions), status=200, mimetype='application/json')


# def get_session_by_id():
#     session_id = request.args.__getitem__('id')
#     doc = db.get_document(COLLECTION_NAME, doc_id=session_id)

#     if doc.exists:
#         session = doc.to_dict()
#         return Response(json.dumps(session), status=200, mimetype='application/json')
#     else:
#         return Response('Session does not exist', status=404, mimetype='application/json')


# def delete_session_by_id():
#     session_id = request.args.__getitem__('id')
#     db.delete_document(COLLECTION_NAME, session_id)
#     return Response(f'Session deleted with id {session_id}', status=200, mimetype='application/json')


# def update_session_by_id():
#     id = request.form['id']
#     title = request.form['title']
#     description = request.form['description']

#     data = {
#         u'title': title,
#         u'description': description,
#     }

#     db.update_document(COLLECTION_NAME, id, data)
#     return Response(f'Session updated with id {id}', status=200, mimetype='application/json')


def calib_results():
    from_ruxailab = json.loads(request.form['from_ruxailab'])
    file_name = json.loads(request.form['file_name'])
    fixed_points = json.loads(request.form['fixed_circle_iris_points'])
    calib_points = json.loads(request.form['calib_circle_iris_points'])
    screen_height = json.loads(request.form['screen_height'])
    screen_width = json.loads(request.form['screen_width'])
    k = json.loads(request.form['k'])

    # Generate csv dataset of calibration points
    os.makedirs(
        f"{Path().absolute()}/app/services/calib_validation/csv/data/", exist_ok=True
    )

    # Generate csv of calibration points with following columns
    calib_csv_file = f"{Path().absolute()}/app/services/calib_validation/csv/data/{file_name}_fixed_train_data.csv"
    csv_columns = [
        "left_iris_x",
        "left_iris_y",
        "right_iris_x",
        "right_iris_y",
        "point_x",
        "point_y",
        "screen_height",
        "screen_width",
    ]

    # Save calibration points to CSV file
    try:
        # Open CSV file
        with open(calib_csv_file, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()

            # Write calibration points to CSV file
            for data in fixed_points:
                data["screen_height"] = screen_height
                data["screen_width"] = screen_width
                writer.writerow(data)

    # Handle I/O error
    except IOError:
        print("I/O error")

    # Generate csv of iris points of session
    os.makedirs(
        f"{Path().absolute()}/app/services/calib_validation/csv/data/", exist_ok=True
    )
    predict_csv_file = f"{Path().absolute()}/app/services/calib_validation/csv/data/{file_name}_predict_train_data.csv"
    csv_columns = ["left_iris_x", "left_iris_y", "right_iris_x", "right_iris_y"]
    try:
        with open(predict_csv_file, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in calib_points:
                # print(data)
                writer.writerow(data)
    except IOError:
        print("I/O error")

    # Run prediction
    data = gaze_tracker.predict(calib_csv_file, calib_csv_file, k)

    if from_ruxailab:
        try:
            payload = {
                "session_id": file_name,
                "model": data,
                "screen_height": screen_height,
                "screen_width": screen_width,
                "k": k
            }

            RUXAILAB_WEBHOOK_URL = "https://receivecalibration-ffptzpxikq-uc.a.run.app"

            print("file_name:", file_name)

            resp = requests.post(RUXAILAB_WEBHOOK_URL, json=payload)
            print("Enviado para RuxaiLab:", resp.status_code, resp.text)
        except Exception as e:
            print("Erro ao enviar para RuxaiLab:", e)

    return Response(json.dumps(data), status=200, mimetype='application/json')



def batch_predict():
    try:
        data = request.get_json()

        iris_data = data['iris_tracking_data']
        k = data.get('k', 3)
        screen_height = data.get('screen_height')
        screen_width = data.get('screen_width')

        base_path = Path().absolute() / 'app/services/calib_validation/csv/data'
        calib_csv_path = base_path / 'vcczxvzxcv_fixed_train_data.csv'
        predict_csv_path = base_path / 'temp_batch_predict.csv'

        print(f"Calib CSV Path: {calib_csv_path}")
        print(f"Predict CSV Path: {predict_csv_path}")
        print(f"Iris data sample (até 3): {iris_data[:3]}")

        # Debug: colunas do CSV de calibração
        df_calib = pd.read_csv(calib_csv_path)
        print("Colunas do CSV de calibração:", df_calib.columns.tolist())

        # Cria CSV temporário com dados de íris para predição
        with open(predict_csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[
                'left_iris_x', 'left_iris_y', 'right_iris_x', 'right_iris_y'
            ])
            writer.writeheader()
            for item in iris_data:
                writer.writerow({
                    'left_iris_x': item['left_iris_x'],
                    'left_iris_y': item['left_iris_y'],
                    'right_iris_x': item['right_iris_x'],
                    'right_iris_y': item['right_iris_y']
                })

        # Chama a predição com os dois CSVs de calibração
        predictions = gaze_tracker.predict(
            calib_csv_path,
            calib_csv_path,
            k
        )

        # Verifica se o retorno é lista, dicionário ou outro
        if isinstance(predictions, list):
            # Se for lista, adiciona timestamp e metadados a cada item
            for i in range(len(predictions)):
                predictions[i]['timestamp'] = iris_data[i].get('timestamp')
                if screen_height is not None:
                    predictions[i]['screen_height'] = screen_height
                if screen_width is not None:
                    predictions[i]['screen_width'] = screen_width
        elif isinstance(predictions, dict):
            # Se for dicionário, anexa metadados gerais (exemplo)
            if screen_height is not None:
                predictions['screen_height'] = screen_height
            if screen_width is not None:
                predictions['screen_width'] = screen_width
            # Timestamp pode não fazer sentido em dicionário com estrutura complexa
        else:
            print("Retorno da predição tem tipo inesperado:", type(predictions))

        return Response(json.dumps(predictions), status=200, mimetype='application/json')

    except Exception as e:
        print("Erro na batch_predict:", e)
        traceback.print_exc()
        return Response("Erro interno na predição", status=500)

# def session_results():
#     session_id = request.args.__getitem__('id')

#     # Train Model
#     data = gaze_tracker.train_model(session_id)

#     # To do: return gaze x and y on response as json
#     gaze = []
#     for i in range(len(data['x'])):
#         gaze.append({
#             'x': data['x'][i],
#             'y': data['y'][i]
#         })

#     return Response(json.dumps(gaze), status=200, mimetype='application/json')


# def session_results_record():
#     session_id = request.args.__getitem__('id')
#     doc = db.get_document(COLLECTION_NAME, doc_id=session_id)
#     if doc.exists:
#         session = doc.to_dict()

#     return send_file(f'{Path().absolute()}/public/videos/{session["screen_record_url"]}', mimetype='video/webm')
