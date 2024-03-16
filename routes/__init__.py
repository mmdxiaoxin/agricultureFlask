# routes/__init__.py

from flask import Blueprint

# 创建一个名为 routes_bp 的蓝图
routes_bp = Blueprint('routes_bp', __name__, url_prefix='/agriculture')

# 导入路由模块并注册到蓝图
from routes import predict_routes, user_routes

# 将路由模块注册到蓝图
routes_bp.register_blueprint(predict_routes.predict_bp)
routes_bp.register_blueprint(user_routes.user_bp)
