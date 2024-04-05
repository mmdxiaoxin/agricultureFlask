import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import logging

from flask import Flask
from flask_cors import CORS

from config import Config
from model import db
from routes import routes_bp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)
cors = CORS(app)

if __name__ == "__main__":
    # 注册Blueprint到Flask应用中
    app.register_blueprint(routes_bp)
    app.run(debug=True, host="0.0.0.0", port=5000)
    print('the server is closed')
