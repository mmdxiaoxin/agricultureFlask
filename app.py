import json
import logging
import time

import pymysql
import requests
from flask import Flask, request, Blueprint, jsonify
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
def Hello_World():
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
def importData():
    return jsonify({"code": 200, "message": "成功"})


# 处理用户数据导出请求
@agriculture_bp.route('/user/export', methods=['POST'])
def exportData():
    return jsonify({"code": 200, "message": "成功"})


# 获取用户设备列表并缓存结果
@agriculture_bp.route("/user/deviceList", methods=['GET'])
@cache.memoize(timeout=3600)  # 缓存结果，设置缓存时间（秒）
def getDeviceList():
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
        deviceList = []

        for site in sites:
            site_id = site[0]
            site_name = site[1]

            cursor.execute('SELECT * FROM device WHERE address_id = %s;', (site_id,))
            devices = cursor.fetchall()

            # 构建站点信息和设备信息的数据结构
            site_info = {
                'id': str(site_id),
                'name': site_name,
                'children': [{'id': str(device[0]), 'name': device[2]} for device in devices]
            }

            deviceList.append(site_info)

        # 构建响应 JSON
        response = {
            'code': 200,
            'data': deviceList,
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
        response_data = [
            {
                "path": "/home/index",
                "name": "home",
                "component": "/home/index",
                "meta": {
                    "icon": "HomeFilled",
                    "title": "首页",
                    "isLink": "",
                    "isHide": False,
                    "isFull": False,
                    "isAffix": True,
                    "isKeepAlive": True
                }
            },
            {
                "path": "/dashboard/dataVisualize",
                "name": "dataVisualize",
                "component": "/dashboard/dataVisualize/index",
                "meta": {
                    "icon": "Odometer",
                    "title": "仪表盘",
                    "isLink": "",
                    "isHide": False,
                    "isFull": False,
                    "isAffix": False,
                    "isKeepAlive": True
                }
            },
            {
                "path": "/diseaseWarning",
                "name": "diseaseWarning",
                "redirect": "/diseaseWarning/dataVisualization",
                "meta": {
                    "icon": "WarningFilled",
                    "title": "农业预警",
                    "isLink": "",
                    "isHide": False,
                    "isFull": False,
                    "isAffix": False,
                    "isKeepAlive": True
                },
                "children": [
                    {
                        "path": "/diseaseWarning/dataVisualization",
                        "name": "dataVisualization",
                        "component": "/diseaseWarning/dataVisualization/index",
                        "meta": {
                            "icon": "Histogram",
                            "title": "数据可视化",
                            "isLink": "",
                            "isHide": False,
                            "isFull": False,
                            "isAffix": False,
                            "isKeepAlive": True
                        }
                    },
                    {
                        "path": "/diseaseWarning/real-timeData",
                        "name": "real-timeData",
                        "component": "/diseaseWarning/real-timeData/index",
                        "meta": {
                            "icon": "Grid",
                            "title": "实时数据",
                            "isLink": "",
                            "isHide": False,
                            "isFull": False,
                            "isAffix": False,
                            "isKeepAlive": True
                        }
                    },
                    {
                        "path": "/diseaseWarning/dataImport",
                        "name": "dataImport",
                        "component": "/diseaseWarning/dataImport/index",
                        "meta": {
                            "icon": "upload",
                            "title": "数据导入",
                            "isLink": "",
                            "isHide": False,
                            "isFull": False,
                            "isAffix": False,
                            "isKeepAlive": True
                        }
                    },
                    {
                        "path": "/diseaseWarning/reportExport",
                        "name": "reportExport",
                        "component": "/diseaseWarning/reportExport/index",
                        "meta": {
                            "icon": "download",
                            "title": "报表导出",
                            "isLink": "",
                            "isHide": False,
                            "isFull": False,
                            "isAffix": False,
                            "isKeepAlive": True
                        }
                    },
                    {
                        "path": "/diseaseWarning/notificationSystem",
                        "name": "notificationSystem",
                        "component": "/diseaseWarning/notificationSystem/index",
                        "meta": {
                            "icon": "bell",
                            "title": "消息通知",
                            "isLink": "",
                            "isHide": False,
                            "isFull": False,
                            "isAffix": False,
                            "isKeepAlive": True
                        }
                    }
                ]
            },
            {
                "path": "/link",
                "name": "link",
                "redirect": "/link/gitee",
                "meta": {
                    "icon": "Paperclip",
                    "title": "外部链接",
                    "isLink": "",
                    "isHide": False,
                    "isFull": False,
                    "isAffix": False,
                    "isKeepAlive": True
                },
                "children": [{
                    "path": "/link/gitee",
                    "name": "gitee",
                    "component": "/link/gitee/index",
                    "meta": {
                        "icon": "Menu",
                        "title": "Gitee 仓库",
                        "isLink": "https://gitee.com/haojingxin/Agricultural",
                        "isHide": False,
                        "isFull": False,
                        "isAffix": False,
                        "isKeepAlive": True
                    }
                },
                    {
                        "path": "/link/github",
                        "name": "github",
                        "component": "/link/github/index",
                        "meta": {
                            "icon": "Menu",
                            "title": "GitHub 仓库",
                            "isLink": "https://github.com/mmdxiaoxin/Agricultural",
                            "isHide": False,
                            "isFull": False,
                            "isAffix": False,
                            "isKeepAlive": True
                        }
                    }
                ]
            },
            {
                "path": "/about/index",
                "name": "about",
                "component": "/about/index",
                "meta": {
                    "icon": "InfoFilled",
                    "title": "关于项目",
                    "isLink": "",
                    "isHide": False,
                    "isFull": False,
                    "isAffix": False,
                    "isKeepAlive": True
                }
            }
        ]
        return jsonify({"code": 200, "data": response_data, "message": "成功"})
    else:
        return jsonify({"code": 401, "message": "Unauthorized"})


@agriculture_bp.route("/address/select", methods=['GET'])
@cache.memoize(timeout=3600)  # 缓存结果，设置缓存时间（秒）
def Address_Select():
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

        total_numberOfDeviceRecords = mihoutao39_count + pingguo42_count + putao41_count + shucai44_count

        response_data = {
            'code': 200,
            'message': '成功',
            'data': {
                'deviceCount': device_count,
                'totalDeviceDataCount': total_numberOfDeviceRecords,
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
def Device_Select():
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
