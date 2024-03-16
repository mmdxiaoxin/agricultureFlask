from flask import Blueprint
from flask import request, jsonify

from model import AgriUser, db

user_bp = Blueprint('user_bp', __name__)


# 处理用户登录请求
@user_bp.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    # 查询用户
    user = AgriUser.query.filter_by(username=username, password=password).first()

    if user:
        code = 200
        access_token = 'bqddxxwqmfncffacvbpkuxvwvqrhln'  # 生成访问令牌的逻辑
        message = '成功'
    else:
        code = 500
        access_token = None
        message = '用户名或密码错误'

    response_data = {
        'code': code,
        'data': {'access_token': access_token},
        'message': message
    }

    return jsonify(response_data)


# 处理用户注册请求
@user_bp.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    # 检查用户名是否已存在
    existing_user = AgriUser.query.filter_by(username=username).first()
    if existing_user:
        code = 500
        message = '用户名已存在'
    else:
        # 创建新用户
        new_user = AgriUser(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        code = 200
        message = '注册成功'

    # 构建注册响应 JSON
    response_data = {
        'code': code,
        'message': message
    }

    return jsonify(response_data)


# 处理用户注销请求
@user_bp.route('/logout', methods=['POST'])
def logout():
    return jsonify({"code": 200, "message": "成功"})


# 处理用户数据导入请求
@user_bp.route('/user/import', methods=['POST'])
def import_data():
    return jsonify({"code": 200, "message": "成功"})


# 处理用户数据导出请求
@user_bp.route('/user/export', methods=['POST'])
def export_data():
    return jsonify({"code": 200, "message": "成功"})
