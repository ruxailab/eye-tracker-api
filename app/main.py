from flask import Flask, jsonify
from flask_cors import CORS

# Local imports from app
from app.routes import session as session_route


# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)


@app.route('/api/session/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'}), 200

# Route for validating calibration
@app.route("/api/session/calib_validation", methods=["POST"])
def calib_validation():
    """
    Validates the calibration request.

    Returns:
        If the request body is valid, it delegates to session_route.calib_results().
    """
    return session_route.calib_results()

@app.route('/api/session/batch_predict', methods=['POST'])
def batch_predict():
    return session_route.batch_predict()
