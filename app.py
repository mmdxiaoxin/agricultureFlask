import logging

from flask import Blueprint
from flask import Flask
from flask_cors import CORS

from config import Config
from model import db
from routes import routes_bp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)
cors = CORS(app)

# 创建名为"agriculture_bp"的Blueprint，并指定前缀
agriculture_bp = Blueprint('agriculture', __name__, url_prefix='/agriculture')

if __name__ == "__main__":
    # 注册Blueprint到Flask应用中
    app.register_blueprint(routes_bp)
    app.register_blueprint(agriculture_bp)
    app.run(debug=True, host="0.0.0.0", port=5000)
    print('the db is closed')
