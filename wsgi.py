"""
This script is the entry point for running the web application.

It imports the `app` object from the `main` module and starts the Flask development server.

Debug:
    The server runs in debug mode if the `debug` argument is set to `True`.

Environment Variables:
    PORT: The port number on which the server should listen. Defaults to 5000 if not provided.
"""

import os
from app.main import app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
