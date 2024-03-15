import json
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import pymysql
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from flask import Blueprint
from flask import Flask, request, jsonify
from flask_caching import Cache
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)
cors = CORS(app)

# 配置缓存
app.config['CACHE_TYPE'] = 'simple'  # 使用内存缓存
cache = Cache(app)

# 数据库连接参数
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "122600",
    "database": "eviroment_data",
    "charset": "utf8",
}

# 最大连接数
MAX_DB_CONNECTIONS = 1000

# 全局数据库连接池
db_connections = []

# 创建名为"agriculture_bp"的Blueprint，并指定前缀
agriculture_bp = Blueprint('agriculture', __name__, url_prefix='/agriculture')

# 全局变量，用于存储预测方法和模型
method = "0"
model = "0"


# 函数用于获取数据库连接
def get_db_connection():
    if len(db_connections) < MAX_DB_CONNECTIONS:
        conn = pymysql.connect(**DB_CONFIG)
        if conn is None:
            logger.error("Failed to create a database connection.")
            raise Exception("Failed to create a database connection.")
        db_connections.append(conn)
        logger.info(f"New database connection created. Total connections: {len(db_connections)}")
        return conn
    else:
        # 如果连接池已满，等待并尝试重新获取连接
        for conn in db_connections:
            if not conn.open:
                conn.ping(reconnect=True)
                logger.info(f"Database connection reopened. Total connections: {len(db_connections)}")
                return conn
        time.sleep(1)  # 等待1秒后重试
        return get_db_connection()


@agriculture_bp.route('/', methods=['GET'])
def hello_world():
    return "欢迎使用农业监控系统"


# 处理用户登录请求
@agriculture_bp.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    # 验证用户名和密码
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            sql = "SELECT * FROM users WHERE username = %s AND password = %s"
            cursor.execute(sql, (username, password))
            user = cursor.fetchone()

        if user:
            code = 200
            access_token = 'bqddxxwqmfncffacvbpkuxvwvqrhln'  # 生成你的访问令牌逻辑
            message = '成功'
        else:
            code = 500
            access_token = None
            message = '用户名或密码错误'
    finally:
        conn.close()

    # 构建登录响应 JSON
    response_data = {
        'code': code,
        'data': {'access_token': access_token},
        'message': message
    }

    return jsonify(response_data)


# 处理用户注册请求
@agriculture_bp.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    # 检查用户名是否已存在
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # 查询数据库中是否已存在相同的用户名
            sql = "SELECT * FROM users WHERE username = %s"
            cursor.execute(sql, (username,))
            existing_user = cursor.fetchone()

        if existing_user:
            code = 500
            message = '用户名已存在'
        else:
            # 如果用户名不存在，将新用户插入数据库
            with conn.cursor() as cursor:
                sql = "INSERT INTO users (username, password) VALUES (%s, %s)"
                cursor.execute(sql, (username, password))
                conn.commit()

            code = 200
            message = '注册成功'
    finally:
        conn.close()

    # 构建注册响应 JSON
    response_data = {
        'code': code,
        'message': message
    }

    return jsonify(response_data)


# 处理用户注销请求
@agriculture_bp.route('/logout', methods=['POST'])
def logout():
    return jsonify({"code": 200, "message": "成功"})


# 处理用户数据导入请求
@agriculture_bp.route('/user/import', methods=['POST'])
def import_data():
    return jsonify({"code": 200, "message": "成功"})


# 处理用户数据导出请求
@agriculture_bp.route('/user/export', methods=['POST'])
def export_data():
    return jsonify({"code": 200, "message": "成功"})


# 获取用户设备列表并缓存结果
@agriculture_bp.route("/user/deviceList", methods=['GET'])
@cache.memoize(timeout=3600)  # 缓存结果，设置缓存时间（秒）
def get_device_list():
    """
    用于获取用户设备列表的GET请求，并缓存结果。

    Returns:
        JSON: 包含设备列表的JSON响应
    """
    conn = None  # 初始化连接为 None，以确保无论如何都能关闭连接
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # 查询所有站点
        cursor.execute('SELECT * FROM address;')
        sites = cursor.fetchall()

        # 查询每个站点的设备
        device_list = []

        for site in sites:
            site_id = site[0]
            site_name = site[1]

            cursor.execute('SELECT * FROM device WHERE address_id = %s;', (site_id,))
            devices = cursor.fetchall()

            # 构建站点信息和设备信息的数据结构
            site_info = {
                'id': str(site_id),
                'name': site_name,
                'isSite': True,  # 添加isSite字段
                'children': [{'id': str(device[0]), 'name': device[2], 'isDevice': True} for device in devices]
            }

            device_list.append(site_info)

        # 构建响应 JSON
        response = {
            'code': 200,
            'data': device_list,
            'message': '成功'
        }

        return jsonify(response)

    except Exception as e:
        response = {
            'code': 500,
            'message': '服务器错误: {}'.format(str(e))
        }
        return jsonify(response), 500

    finally:
        # 在 finally 块中关闭连接，确保无论如何都会关闭连接
        if conn and conn.open:
            conn.close()


@agriculture_bp.route('/menu/list', methods=['GET'])
def mock_response():
    headers = request.headers
    access_token = headers.get('X-Access-Token')
    if access_token == 'bqddxxwqmfncffacvbpkuxvwvqrhln':
        with open('config.json', 'r', encoding='utf-8') as file:
            response_data = json.load(file)['menu']
        return jsonify({"code": 200, "data": response_data, "message": "成功"})
    else:
        return jsonify({"code": 401, "message": "Unauthorized"})


@agriculture_bp.route("/address/select", methods=['GET'])
@cache.memoize(timeout=3600)  # 缓存结果，设置缓存时间（秒）
def address_select():
    conn = None  # 初始化连接为 None，以确保无论如何都能关闭连接
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT * FROM address;')
        results = cur.fetchall()
        rows = []

        for row in results:
            site_id = row[0]
            name = row[1]

            row_dict = {
                'id': site_id,
                'name': name
            }

            rows.append(row_dict)

        response_data = {
            'code': 200,
            'data': rows,
            'message': 'Success'
        }

        return jsonify(response_data)
    except Exception as e:
        response_data = {
            'code': 500,
            'message': f'Server Error: {str(e)}'
        }
        return jsonify(response_data), 500
    finally:
        # 在 finally 块中关闭连接，确保无论如何都会关闭连接
        if conn and conn.open:
            conn.close()


# 返回仪表盘数据
@agriculture_bp.route('/device/count', methods=['GET'])
def get_device_count():
    conn = None
    try:
        conn = get_db_connection()  # 获取数据库连接

        # 执行查询获取设备数量
        with conn.cursor() as cursor:
            cursor.execute('SELECT COUNT(*) FROM device;')
            device_count = cursor.fetchone()[0]  # 获取设备数量

        # 执行查询获取站点数量
        with conn.cursor() as cursor:
            cursor.execute('SELECT COUNT(*) FROM env_db;')
            site_count = cursor.fetchone()[0]  # 获取站点数量

        # 查询mihoutao39表的数据记录数量
        with conn.cursor() as cursor:
            cursor.execute('SELECT COUNT(*) FROM mihoutao39;')
            mihoutao39_count = cursor.fetchone()[0]  # 获取mihoutao39表的数据记录数量

        # 查询pingguo42表的数据记录数量
        with conn.cursor() as cursor:
            cursor.execute('SELECT COUNT(*) FROM pingguo42;')
            pingguo42_count = cursor.fetchone()[0]  # 获取pingguo42表的数据记录数量

        # 查询putao41表的数据记录数量
        with conn.cursor() as cursor:
            cursor.execute('SELECT COUNT(*) FROM putao41;')
            putao41_count = cursor.fetchone()[0]  # 获取putao41表的数据记录数量

        # 查询shucai44表的数据记录数量
        with conn.cursor() as cursor:
            cursor.execute('SELECT COUNT(*) FROM shucai44;')
            shucai44_count = cursor.fetchone()[0]  # 获取shucai44表的数据记录数量

        total_number_of_device_records = mihoutao39_count + pingguo42_count + putao41_count + shucai44_count

        response_data = {
            'code': 200,
            'message': '成功',
            'data': {
                'deviceCount': device_count,
                'totalDeviceDataCount': total_number_of_device_records,
                'siteCount': site_count,
                'siteValues':
                    [
                        {
                            'name': "武功猕猴桃试验站",
                            'value': mihoutao39_count
                        },
                        {
                            'name': "白水苹果试验站",
                            'value': pingguo42_count
                        },
                        {
                            'name': "临渭葡萄研究所",
                            'value': putao41_count
                        },
                        {
                            'name': "泾阳蔬菜示范站",
                            'value': shucai44_count
                        }
                    ]
            }
        }

        return jsonify(response_data)

    except Exception as e:
        response_data = {
            'code': 500,
            'message': f'服务器错误: {str(e)}'
        }
        return jsonify(response_data), 500

    finally:
        if conn and conn.open:
            conn.close()


# 返回传感器设备列表（根据地区）
@agriculture_bp.route("/device/select", methods=['GET'])
def device_select():
    conn = None  # 初始化连接为 None，以确保无论如何都能关闭连接
    try:
        conn = get_db_connection()
        # 创建数据库游标
        cur = conn.cursor()
        address_id = request.args.get("address_id", default=1, type=str)
        cur.execute("SELECT id, device_name, business_id, device_id, collect_run FROM device WHERE address_id = %s;",
                    address_id)
        conn.commit()
        results = cur.fetchall()
        rows = []

        # 遍历结果集
        for row in results:
            site_id = row[0]
            device_name = row[1]
            business_id = row[2]
            device_id = row[3]
            collect = row[4]

            # 创建字典对象，表示每行结果
            row_dict = {
                'id': site_id,
                'device_name': device_name,
                'business_id': business_id,
                'device_id': device_id,
                'collect_run': collect
            }
            # 将每行结果的字典添加到列表中
            rows.append(row_dict)

        # 将结果列表转换为JSON格式
        json_data = json.dumps(rows)
        return json_data

    except Exception as e:
        # 处理异常，您可以根据需要进行记录或其他操作
        response_data = {
            'code': 500,
            'message': '服务器错误: {}'.format(str(e))
        }
        return jsonify(response_data), 500

    finally:
        # 在 finally 块中关闭连接，确保无论如何都会关闭连接
        if conn and conn.open:
            conn.close()


# 根据参数返回数据库中的数据
@agriculture_bp.route("/data/show", methods=['GET'])
def data_base():
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            device_id = request.args.get("deviceId", default=1, type=str)
            hour = request.args.get("hour", default=24, type=int)
            columns_param = request.args.get("columns")

            dbname = get_database_name(cur, device_id)
            if not table_exists(cur, dbname):
                response_data = {
                    'code': 404,
                    'message': 'No data found',
                }
                return jsonify(response_data)

            all_columns = get_all_columns(cur, dbname)
            columns_to_select = get_columns_to_select(columns_param, all_columns)

            select_clause = construct_select_clause(columns_to_select)
            results = execute_query(cur, dbname, select_clause, hour)

            json_data = convert_to_json(results, columns_to_select)

            response_data = {
                'code': 200,
                'message': 'Success',
                'data': json_data
            }

            return jsonify(response_data)

    except IndexError:
        response_data = {
            'code': 404,
            'message': 'No data found',
        }
        return jsonify(response_data)

    except Exception as e:
        print("服务器错误:", str(e))
        response_data = {
            'code': 500,
            'message': '服务器错误: {}'.format(str(e))
        }
        return jsonify(response_data), 500


# 选择合适的预处理方式
@agriculture_bp.route('/predict/predict-method', methods=['POST'])
def predict_method():
    global method
    method_value = request.json.get('method')
    if method_value is not None:
        method = method_value
        return jsonify({'code': 200, 'msg': f"Method updated to {method}"}), 200
    else:
        return jsonify({'code': 400, 'msg': "No method parameter provided in the request"}), 400


# 选择合适的模型
@agriculture_bp.route('/predict/models', methods=['POST'])
def chose_model():
    global model
    model_value = request.json.get('model')
    if model_value is not None:
        model = model_value
        return jsonify({'code': 200, 'msg': f"Model updated to: {model}"}), 200
    else:
        return jsonify({'code': 400, 'msg': "No model parameter provided in the request"}), 400


# 给出模型预测结果
@agriculture_bp.route('/predict/upload-file', methods=['POST'])
def upload_file():
    global model, method
    received_file = request.files.get('input_image')  # 使用get方法获取文件，避免出错
    if received_file:
        image_file_name = received_file.filename
        # 加载模型参数
        model_path = ""
        predictions = []
        class_names = ()
        if model == 0:
            model_path = './saved_model/Res_RGB.pt'
            class_names = ('褐斑病', '斑点落叶病', '花叶病', '健康', '锈病')
            # 加载 ResNet 模型
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = ResNet(5)  # 实例化模型对象
            model.load_state_dict(torch.load(model_path, map_location=device))  # 加载模型参数
            model.eval()  # 设置模型为评估模式
            # 随机生成一个数据并进行预测
            img = torch.randn(1, 3, 64, 64)  # 第一个1是batch_size，这里随机生成了一个数据
            result = model(img)  # 传入图像返回类别序号
            probabilities = torch.softmax(result, dim=1).tolist()[0]
            predictions = [{class_names[i]: probabilities[i]} for i in range(len(class_names))]
        elif model == 1:
            model_path = './saved_model/Net2_59.pt'
            class_names = ('花叶病', '健康', '锈病')
            # 加载 Net2 模型
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = Net2(125, 3)  # 实例化模型对象
            model.load_state_dict(torch.load(model_path, map_location=device))  # 加载模型参数
            model.eval()  # 设置模型为评估模式
            # 随机生成一个数据并进行预测
            img = torch.randn(1, 1, 125, 64, 64)  # 第一个1是batch_size，这里随机生成了一个数据
            result = model(img)  # 传入图像返回类别序号
            probabilities = torch.softmax(result, dim=1).tolist()[0]
            predictions = [{class_names[i]: probabilities[i]} for i in range(len(class_names))]

        return jsonify({'code': 200, 'data': {'predictions': predictions}, 'msg': "文件上传成功"})
    else:
        return jsonify({'code': 400, 'msg': "No file provided in the request"})


# 沿波段方向绘制某一像素波谱曲线
def spectra_plot(img, position, image_file_name):
    x, y = position
    # 提取光谱数据
    spectra = img[x, y, :].reshape(img.shape[2])

    # 创建波段索引
    wavelengths = np.arange(0, img.shape[2])

    # 绘制光谱曲线
    plt.plot(wavelengths, spectra)
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.title(f'Spectral Curve of ({x},{y})')
    file_name = f"./result/spectral_curve_image/{image_file_name}"
    plt.savefig(file_name)
    return file_name


class ResNet(nn.Module):
    def __init__(self, class_num):
        super(ResNet, self).__init__()
        self.net = models.resnet18(pretrained=False)
        self.net.fc = nn.Linear(in_features=512, out_features=class_num, bias=True)

    def forward(self, x):
        x = self.net(x)
        x = F.softmax(x, dim=1)
        return x


class InceptionResBlock(nn.Module):
    def __init__(self, in_channels):
        super(InceptionResBlock, self).__init__()
        self.branch1x1 = nn.Conv3d(1, in_channels, kernel_size=1, stride=(1, 1, 1), padding=0)
        self.branch2_1 = nn.Conv3d(1, 1, kernel_size=1, stride=(1, 1, 1), padding=0)
        self.branch3x3 = nn.Conv3d(1, in_channels, kernel_size=3, stride=(1, 1, 1), padding=1)
        self.branch3_1 = nn.Conv3d(1, 1, kernel_size=1, stride=(1, 1, 1), padding=0)
        self.branch5x5 = nn.Conv3d(1, in_channels, kernel_size=5, stride=(1, 1, 1), padding=2)
        self.conv1x1 = nn.Conv3d(in_channels * 3, 1, kernel_size=1, stride=(1, 1, 1), padding=0)
        self.bn = nn.BatchNorm3d(1)
        self.scale = 0.1
        self.init_weights()

    def forward(self, x):
        x1 = self.branch1x1(x)
        x2 = self.branch3x3(self.branch2_1(x))
        x3 = self.branch5x5(self.branch3_1(x))
        out = torch.cat((x1, x2, x3), dim=1)
        out = self.bn(self.conv1x1(out))
        out = x + self.scale * out
        out = F.leaky_relu(out)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class InceptionResBlock_SE(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(InceptionResBlock_SE, self).__init__()
        self.branch1x1 = nn.Conv3d(1, in_channels, kernel_size=1, stride=(1, 1, 1), padding=0)
        self.branch2_1 = nn.Conv3d(1, 1, kernel_size=1, stride=(1, 1, 1), padding=0)
        self.branch3x3 = nn.Conv3d(1, in_channels, kernel_size=3, stride=(1, 1, 1), padding=1)
        self.branch3_1 = nn.Conv3d(1, 1, kernel_size=1, stride=(1, 1, 1), padding=0)
        self.branch5x5 = nn.Conv3d(1, in_channels, kernel_size=5, stride=(1, 1, 1), padding=2)
        self.conv1x1 = nn.Conv3d(in_channels * 3, 1, kernel_size=1, stride=(1, 1, 1), padding=0)
        self.bn = nn.BatchNorm3d(1)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.LeakyReLU(inplace=True)
        )
        self.scale = 0.1

    def forward(self, x):
        n, _, b, w, h = x.shape
        se_weight = self.se(x.view(n, b, w, h))
        x = x * se_weight.view(n, -1, b, 1, 1)
        x1 = self.branch1x1(x)
        x2 = self.branch3x3(self.branch2_1(x))
        x3 = self.branch5x5(self.branch3_1(x))
        out = torch.cat((x1, x2, x3), dim=1)
        out = self.bn(self.conv1x1(out))
        out = x + self.scale * out
        out = F.leaky_relu(out)
        return out


class Net2(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Net2, self).__init__()
        self.num_classes = num_classes
        self.block1 = InceptionResBlock_SE(in_channels)
        self.block2 = InceptionResBlock_SE(in_channels)
        self.block3 = InceptionResBlock(in_channels)
        self.block4 = InceptionResBlock(in_channels)
        self.pool = nn.AdaptiveAvgPool3d((in_channels, 1, 1))
        self.fc = nn.Conv2d(in_channels, num_classes, (1, 1))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        n, _, b, w, h = x.shape
        x = x.reshape(n, b, w, h)
        x = F.leaky_relu(self.fc(x))
        x = x.view(x.size(0), -1)
        x = F.softmax(x, dim=1)
        return x


def get_database_name(cur, device_id):
    cur.execute("SELECT name FROM env_db WHERE deviceId = %s;", (device_id,))
    return str(cur.fetchall()[0][0]) + device_id


def table_exists(cur, dbname):
    cur.execute("SHOW TABLES LIKE %s;", (dbname,))
    return cur.fetchone()


def get_all_columns(cur, dbname):
    cur.execute("SHOW COLUMNS FROM " + dbname + ";")
    result_columns = cur.fetchall()
    return [i[0] for i in result_columns]


def get_columns_to_select(columns_param, all_columns):
    if columns_param:
        columns_to_select = columns_param.split(',')
    else:
        columns_to_select = []
    return columns_to_select if columns_to_select else all_columns


def construct_select_clause(columns_to_select):
    return ", ".join(columns_to_select)


def execute_query(cur, dbname, select_clause, hour):
    cur.execute(f"SELECT {select_clause} FROM " + dbname + " ORDER BY id DESC LIMIT %s;", (hour,))
    return cur.fetchall()


def convert_to_json(results, columns_to_select):
    rows = []
    for row in results:
        row_dict = dict(zip(columns_to_select, row))
        rows.append(row_dict)
    return rows


# 返回实时数据或者设备属性等等
@agriculture_bp.route("/device/api", methods=['GET'])
def Device_api():
    conn = None  # 初始化连接为 None，以确保无论如何都能关闭连接
    try:
        conn = get_db_connection()
        # 创建数据库游标
        cur = conn.cursor()
        device_id = request.args.get("id", default=1, type=str)
        method = request.args.get("method", default=1, type=str)
        cur.execute("SELECT api,business_id, device_id, equipment, version, collect_run FROM device WHERE id = %s;",
                    device_id)
        conn.commit()
        res_device = cur.fetchone()
        if res_device is None:
            print("未查询到此设备")
            response = {
                "code": 404,
                "message": "未查询到此设备"
            }
            return jsonify(response), 404

        if res_device[5] == '0':
            response = {
                "code": 403,
                "message": "没有访问权限"
            }
            return jsonify(response), 403

        res_device[0] + '/' + method + '?' + 'Version=' + res_device[4] + '&Business=' + res_device[
            1] + '&Equipment=' + res_device[3] + '&RequestTime=' + str(
            int(time.time())) + '&Value={ "page": 1,"length": 5,"deviceId":' + res_device[2] + '}'
        # device_api_last_index = res_api.rfind("/")
        device_url = res_device[0] + '/' + method
        value = "{ 'page': 1,'length': 5, 'deviceId': " + res_device[2] + " }"
        device_params = {
            "Version": res_device[4],
            "Business": res_device[1],
            "Equipment": res_device[3],
            "RequestTime": str(int(time.time())),
            "Value": value
        }
        result_data = requests.get(device_url, device_params)
        result = json.loads(result_data.text)

        return jsonify(result)

    except Exception as e:
        # 处理异常，您可以根据需要进行记录或其他操作
        response_data = {
            'code': 500,
            'message': '服务器错误: {}'.format(str(e))
        }
        return jsonify(response_data), 500

    finally:
        # 在 finally 块中关闭连接，确保无论如何都会关闭连接
        if conn and conn.open:
            conn.close()


if __name__ == "__main__":
    # 注册Blueprint到Flask应用中
    app.register_blueprint(agriculture_bp)
    app.run(debug=True, host="0.0.0.0", port=5000)
    print('the db is closed')
