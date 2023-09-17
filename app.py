import json
import time
import pymysql
import requests
from flask import Flask, request, Blueprint, jsonify
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)

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
MAX_DB_CONNECTIONS = 10

# 全局数据库连接池
db_connections = []

# 创建名为"agriculture_bp"的Blueprint，并指定前缀
agriculture_bp = Blueprint('agriculture', __name__, url_prefix='/agriculture')


# 函数用于获取数据库连接
def get_db_connection():
    if len(db_connections) < MAX_DB_CONNECTIONS:
        conn = pymysql.connect(**DB_CONFIG)
        db_connections.append(conn)
        return conn
    else:
        # 如果连接池已满，等待并尝试重新获取连接
        for conn in db_connections:
            if not conn.open:
                conn.ping(reconnect=True)
                return conn
        time.sleep(1)  # 等待1秒后重试
        return get_db_connection()


@agriculture_bp.route('/', methods=['GET'])
def Hello_World():
    return "欢迎使用农业监控系统"


@agriculture_bp.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if ((username == 'admin' and password == 'e10adc3949ba59abbe56e057f20f883e') or
            (username == 'user' and password == 'e10adc3949ba59abbe56e057f20f883e')):
        code = 200
        access_token = 'bqddxxwqmfncffacvbpkuxvwvqrhln' if username == 'admin' else 'unufvdotdqxuzfbdygovfmsbftlvbn'
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


@agriculture_bp.route('/logout', methods=['POST'])
def logout():
    return jsonify({"code": 200, "message": "成功"})


@agriculture_bp.route('/user/import', methods=['POST'])
def importData():
    return jsonify({"code": 200, "message": "成功"})


@agriculture_bp.route('/user/export', methods=['POST'])
def exportData():
    return jsonify({"code": 200, "message": "成功"})


@agriculture_bp.route("/user/deviceList", methods=['GET'])
def getDeviceList():
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
        cursor.close()


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
                "path": "/dataScreen",
                "name": "dataScreen",
                "component": "/dataScreen/index",
                "meta": {
                    "icon": "Histogram",
                    "title": "数据大屏",
                    "isLink": "",
                    "isHide": False,
                    "isFull": True,
                    "isAffix": False,
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
                "children": [{
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
                        "path": "/diseaseWarning/diseaseWarningSys",
                        "name": "diseaseWarningSys",
                        "component": "/diseaseWarning/diseaseWarningSys/index",
                        "meta": {
                            "icon": "warning",
                            "title": "病害预警",
                            "isLink": "",
                            "isHide": False,
                            "isFull": False,
                            "isAffix": False,
                            "isKeepAlive": True
                        }
                    },
                    {
                        "path": "/diseaseWarning/preventionAndControlInformation",
                        "name": "preventionAndControlInformation",
                        "component": "/diseaseWarning/preventionAndControlInformation/index",
                        "meta": {
                            "icon": "QuestionFilled",
                            "title": "防治信息",
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
                        "path": "/diseaseWarning/scheduledTasks",
                        "name": "scheduledTasks",
                        "component": "/diseaseWarning/scheduledTasks/index",
                        "meta": {
                            "icon": "clock",
                            "title": "定时任务",
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
                "path": "/system",
                "name": "system",
                "redirect": "/system/accountManage",
                "meta": {
                    "icon": "Tools",
                    "title": "系统管理",
                    "isLink": "",
                    "isHide": False,
                    "isFull": False,
                    "isAffix": False,
                    "isKeepAlive": True
                },
                "children": [{
                    "path": "/system/accountManage",
                    "name": "accountManage",
                    "component": "/system/accountManage/index",
                    "meta": {
                        "icon": "Menu",
                        "title": "账号管理",
                        "isLink": "",
                        "isHide": False,
                        "isFull": False,
                        "isAffix": False,
                        "isKeepAlive": True
                    }
                },
                    {
                        "path": "/system/menuMange",
                        "name": "menuMange",
                        "component": "/system/menuMange/index",
                        "meta": {
                            "icon": "Menu",
                            "title": "菜单管理",
                            "isLink": "",
                            "isHide": False,
                            "isFull": False,
                            "isAffix": False,
                            "isKeepAlive": True
                        }
                    },
                    {
                        "path": "/system/systemLog",
                        "name": "systemLog",
                        "component": "/system/systemLog/index",
                        "meta": {
                            "icon": "Menu",
                            "title": "系统日志",
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
def Address_Select():
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
        cur.close()


# 返回传感器设备列表（根据地区）
@agriculture_bp.route("/device/select", methods=['GET'])
def Device_Select():
    conn = get_db_connection()
    # 创建数据库游标
    cur = conn.cursor()
    address_id = request.args.get("address_id", default=1, type=str)
    cur.execute("SELECT id, device_name, business_id, device_id, collect_run FROM device WHERE address_id = %s;",
                address_id)
    conn.commit()
    results = cur.fetchall()
    rows = []
    cur.close()
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
    # 关闭数据库游标

    return json_data


@agriculture_bp.route("/data/show", methods=['GET'])
def Data_Base():
    conn = get_db_connection()
    # 创建数据库游标
    cur = conn.cursor()
    deviceId = request.args.get("deviceId", default=1, type=str)
    hour = request.args.get("hour", default=24, type=int)  # 获取 hour 参数，如果没有传递参数，默认值为 24 条数据
    columns_param = request.args.get("columns")  # 获取要选择的列，作为一个字符串

    # 检查是否存在 columns 参数，并以逗号分隔的方式拆分列名
    if columns_param:
        columns_to_select = columns_param.split(',')
    else:
        columns_to_select = []

    cur.execute("SELECT name FROM env_db WHERE deviceId = %s;", (deviceId,))
    conn.commit()
    dbname = str(cur.fetchall()[0][0]) + deviceId

    # 判断是否存在数据表
    cur.execute("SHOW TABLES LIKE %s;", (dbname,))
    conn.commit()
    table_exists = cur.fetchone()

    if not table_exists:
        cur.close()
        response_data = {
            'code': 404,
            'message': 'No data found',
        }
        return jsonify(response_data)

    cur.execute("SHOW COLUMNS FROM " + dbname + ";")
    conn.commit()
    result_columns = cur.fetchall()
    all_columns = [i[0] for i in result_columns]

    # 构建 SQL 查询以选择特定的列
    if not columns_to_select:
        # 如果没有指定要选择的列，默认选择所有列
        columns_to_select = all_columns
    else:
        # 验证所请求的列是否存在于数据表中（忽略大小写）
        invalid_columns = [col for col in columns_to_select if col.lower() not in [c.lower() for c in all_columns]]
        if invalid_columns:
            cur.close()
            response_data = {
                'code': 400,
                'message': f'Invalid columns: {", ".join(invalid_columns)}',
            }
            return jsonify(response_data)

    # 构建 SELECT 子句
    select_clause = ", ".join(columns_to_select)

    # 使用 LIMIT 子句来限制返回的数据条数
    cur.execute(f"SELECT {select_clause} FROM " + dbname + " ORDER BY id DESC LIMIT %s;", (hour,))
    conn.commit()
    results = cur.fetchall()
    rows = []
    cur.close()

    for row in results:
        row_dict = dict(zip(columns_to_select, row))
        rows.append(row_dict)

    # 将结果列表转换为 JSON 格式
    json_data = rows

    response_data = {
        'code': 200,
        'message': 'Success',
        'data': json_data
    }

    return jsonify(response_data)


# 实时数据返回
@agriculture_bp.route("/device/api", methods=['GET'])
def Device_api():
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


if __name__ == "__main__":
    # 注册Blueprint到Flask应用中
    app.register_blueprint(agriculture_bp)
    app.run(debug=True, host="0.0.0.0", port=5000)
    print('the db is closed')
