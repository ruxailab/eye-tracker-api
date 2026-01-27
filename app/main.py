from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, join_room, emit
from collections import defaultdict
import time

# Local imports from app
from app.routes import session as session_route


# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

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

# TODO: replace with persistent storage for replay & analytics
gaze_buffer = defaultdict(list)

@socketio.on("join_session")
def handle_join(data):
    session_id = data.get("session_id")
    print("OBSERVER JOIN:", session_id, flush=True)
    if session_id:
        join_room(session_id)
        emit("joined", {"session_id": session_id})


@app.route("/api/session/gaze", methods=["POST"])
def receive_gaze():
    data = request.get_json()

    if not data:
        return Response("Invalid JSON", status=400)

    session_id = data.get("session_id")
    x = data.get("x")
    y = data.get("y")
    timestamp = data.get("timestamp", time.time())

    if not session_id or x is None or y is None:
        return Response("Invalid payload", status=400)

    point = {
        "session_id": session_id,
        "x": x,
        "y": y,
        "timestamp": timestamp,
        "phase": data.get("phase"),
    }

    print("GAZE:", point, flush=True)

    # store for replay later
    gaze_buffer[session_id].append(point)
    socketio.emit("gaze_point", point, room=session_id)

    return jsonify({"status": "ok"})

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)

