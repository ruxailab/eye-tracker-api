from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import pandas as pd
from app.services.metrics import EyeTrackingBenchmark

# Local imports from app
from app.routes import session as session_route


# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)


# @app.route('/', methods=['GET'])
# def welcome():
#     return Response(f'Welcome to EyeLab!', status=200, mimetype='application/json')

# @app.route('/api/user/sessions', methods=['GET'])
# def get_user_sessions():
#     # Get user sessions
#     if request.method == 'GET':
#         return session_route.get_user_sessions()

#     return Response('Invalid request method for route', status=405, mimetype='application/json')

# @app.route('/api/session', methods=['GET','POST','PATCH','DELETE'])
# def session():
#     # Get by ID
#     if request.method == 'GET':
#         return session_route.get_session_by_id()

#     # Create Session
#     elif request.method == 'POST':
#         return session_route.create_session()

#     # Delete by ID
#     elif request.method == 'DELETE':
#         return session_route.delete_session_by_id()

#     # Update by ID
#     elif request.method == 'PATCH':
#         return session_route.update_session_by_id()

#     return Response('Invalid request method for route', status=405, mimetype='application/json')

# @app.route('/api/session/results/record', methods=['GET'])
# def manage_recording():
#     if request.method == 'GET':
#         return session_route.session_results_record()
#     return Response('Invalid request method for route', status=405, mimetype='application/json')

# @app.route('/api/session/results', methods=['GET'])
# def manage_results():
#     if request.method == 'GET':
#         return session_route.session_results()
#     return Response('Invalid request method for route', status=405, mimetype='application/json')

@app.route('/api/session/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'}), 200

# Route for validating calibration
@app.route("/api/session/calib_validation", methods=["POST"])
def calib_validation():
    """
    Validates the calibration request.

    Returns:
        If the request method is 'POST', it calls the `calib_results` function from the `session_route` module.
        Otherwise, it returns a `Response` object with an error message and status code 405.
    """
    if request.method == "POST":
        return session_route.calib_results()
    return Response('Invalid request method for route', status=405, mimetype='application/json')

@app.route('/api/session/batch_predict', methods=['POST'])
def batch_predict():
    if request.method == 'POST':
        return session_route.batch_predict()
    return Response('Invalid request method for route', status=405, mimetype='application/json')

"""
POST /api/session/benchmark

Runs eye-tracking benchmark evaluation.

Expected JSON:
{
    "screen_width_px": int,
    "screen_width_cm": float,
    "viewing_distance_cm": float,
    "samples": [
        {
            "True X": float,
            "True Y": float,
            "Predicted X": float,
            "Predicted Y": float
        }
    ]
}
"""
"""
POST /api/session/benchmark

Evaluates eye-tracking accuracy and precision.
"""
@app.route('/api/session/benchmark', methods=['POST'])
def run_benchmark():
    try:
        data = request.get_json()

        required_keys = {
            "samples",
            "screen_width_px",
            "screen_width_cm",
            "viewing_distance_cm"
        }

        if not required_keys.issubset(data.keys()):
            missing = required_keys - set(data.keys())
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        df = pd.DataFrame(data["samples"])

        benchmark = EyeTrackingBenchmark(
            df=df,
            screen_width_px=data["screen_width_px"],
            screen_width_cm=data["screen_width_cm"],
            viewing_distance_cm=data["viewing_distance_cm"]
        )

        results = {
            "overall": benchmark.evaluate(),
            "per_target": benchmark.evaluate_per_target()
        }

        return jsonify(results), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
