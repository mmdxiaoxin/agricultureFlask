import json
import time
from datetime import datetime

import pymysql
import requests
from flask import Flask, request, Blueprint, jsonify
from flask_cors import CORS

from camera_get import CameraUrl
from camera_get import Camera_Control_Start
from camera_get import Camera_Control_Stop

app = Flask(__name__)

cors = CORS(app)
# 主机名
HOSTNAME = "127.0.0.1"
# 端口号
PORT = 3306
# 账号
USERNAME = "root"
# 密码
PASSWORD = "122600"
# 数据库
DATABASE = "eviroment_data"
# 线程池
THREADS = {}
# 线程池是一个字典，key为设备id，value为线程对象

# 使用pymysql创建数据库连接
global con
con = pymysql.connect(host='localhost', password=PASSWORD, port=PORT, user=USERNAME, charset='utf8',
                      database=DATABASE)

# 创建名为"agriculture_bp"的Blueprint，并指定前缀
agriculture_bp = Blueprint('agriculture', __name__, url_prefix='/agriculture')


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


# 返回所有地区列表
@agriculture_bp.route("/address/select", methods=['GET'])
def Address_Select():
    global con
    if not con.ping():
        con = pymysql.connect(host='localhost', password=PASSWORD, port=PORT, user=USERNAME, charset='utf8',
                              database=DATABASE)
    cur = con.cursor()
    affected_rows = cur.execute(
        'SELECT * FROM address;'
    )
    con.commit()
    results = cur.fetchall()
    cur.close()
    rows = []
    for row in results:
        id = row[0]
        name = row[1]

        row_dict = {
            'id': id,
            'name': name
        }

        rows.append(row_dict)

    # 构建响应数据
    response_data = {
        'code': 200,  # 可根据需要设置不同的状态码
        'data': rows,
        'message': 'Success'  # 可根据需要设置不同的消息
    }

    # 使用 Flask 的 jsonify 函数将响应数据转换为 JSON 格式并返回
    return jsonify(response_data)


# 返回传感器设备列表（根据地区）
@agriculture_bp.route("/device/select", methods=['GET'])
def Device_Select():
    global con
    if not con.ping():
        con = pymysql.connect(host='localhost', password=PASSWORD, port=PORT, user=USERNAME, charset='utf8',
                              database=DATABASE)
    # 创建数据库游标
    cur = con.cursor()
    address_id = request.args.get("address_id", default=1, type=str)
    cur.execute("SELECT id, device_name, business_id, device_id, collect_run FROM device WHERE address_id = %s;",
                (address_id))
    con.commit()
    results = cur.fetchall()
    rows = []
    cur.close()
    # 遍历结果集
    for row in results:
        id = row[0]
        device_name = row[1]
        business_id = row[2]
        device_id = row[3]
        collect = row[4]

        # 创建字典对象，表示每行结果
        row_dict = {
            'id': id,
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


@agriculture_bp.route("/user/deviceList", methods=['GET'])
def getDeviceList():
    global con
    try:
        if not con.ping():
            con = pymysql.connect(host='localhost', password=PASSWORD, port=PORT, user=USERNAME, charset='utf8',
                                  database=DATABASE)
        cursor = con.cursor()

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


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()  # 将 datetime 对象转换为 ISO 8601 格式的字符串
        return super().default(obj)


# 返回最新的100条历史数据（按传感器设备）
@agriculture_bp.route("/data/show", methods=['GET'])
def Data_Base():
    global con
    if not con.ping():
        con = pymysql.connect(host='localhost', password=PASSWORD, port=PORT, user=USERNAME, charset='utf8',
                              database=DATABASE)
    # 创建数据库游标
    cur = con.cursor()
    deviceId = request.args.get("deviceId", default=1, type=str)
    cur.execute("SELECT name FROM env_db WHERE deviceId = %s;", (deviceId))
    con.commit()
    dbname = str(cur.fetchall()[0][0]) + deviceId

    cur.execute("SHOW COLUMNS FROM " + dbname + ";")
    con.commit()
    result_columns = cur.fetchall()
    # print("!!!!!!!!")
    # print(result_columns)
    columns = []
    for i in result_columns:
        columns.append(i[0])
    cur.execute("SELECT * FROM " + dbname + " ORDER BY id DESC LIMIT 0,100;")
    con.commit()
    results = cur.fetchall()
    rows = []
    cur.close()
    # print(results)

    for row in results:
        row_dict = dict(zip(columns, row))
        rows.append(row_dict)

    # 将结果列表转换为JSON格式
    json_data = json.dumps(rows, cls=DateTimeEncoder)
    # 关闭数据库游标

    return json_data


@agriculture_bp.route("/camera/select", methods=['GET'])
def Camera_Select():
    global con
    if not con.ping():
        con = pymysql.connect(host='localhost', password=PASSWORD, port=PORT, user=USERNAME, charset='utf8',
                              database=DATABASE)
    # 创建数据库游标
    cur = con.cursor()
    address_id = request.args.get("address_id", default=1, type=str)
    cur.execute("SELECT id, appkey, appsecret, deviceserial FROM camera WHERE address_id = %s;", (address_id))
    con.commit()
    results = cur.fetchall()
    rows = []
    cur.close()
    # 遍历结果集
    for row in results:
        id = row[0]
        appkey = row[1]
        appsecret = row[2]
        deviceserial = row[3]

        # 创建字典对象，表示每行结果
        row_dict = {
            'id': id,
            'appkey': appkey,
            'appsecret': appsecret,
            'deviceserial': deviceserial,
        }
        # 将每行结果的字典添加到列表中
        rows.append(row_dict)

    # 将结果列表转换为JSON格式
    json_data = json.dumps(rows)
    # 关闭数据库游标

    return json_data


@agriculture_bp.route("/camera/url", methods=['GET'])
def Camera_url():
    global con
    if not con.ping():
        con = pymysql.connect(host='localhost', password=PASSWORD, port=PORT, user=USERNAME, charset='utf8',
                              database=DATABASE)
    # 创建数据库游标
    cur = con.cursor()
    id = request.args.get("id", default=1, type=str)
    cur.execute("SELECT appkey, appsecret, deviceserial FROM camera WHERE id = %s;", (id))
    con.commit()
    results = cur.fetchall()
    # print(results)
    cur.close()
    rows = []
    for row in results:
        appkey = row[0]
        appsecret = row[1]
        deviceserial = row[2]
        # 创建字典对象，表示每行结果
        row_dict = {
            'appkey': appkey,
            'appsecret': appsecret,
            'deviceserial': deviceserial,
        }
        # 将每行结果的字典添加到列表中
        rows.append(row_dict)
    print(rows)
    appkey = rows[0]['appkey']
    appsecret = rows[0]['appsecret']
    deviceserial = rows[0]['deviceserial']
    cameraurl = CameraUrl(appkey, appsecret, deviceserial)
    # print(cameraurl)
    return cameraurl


@agriculture_bp.route("/camera/control", methods=['GET'])
def Camera_Control():
    global con
    if not con.ping():
        con = pymysql.connect(host='localhost', password=PASSWORD, port=PORT, user=USERNAME, charset='utf8',
                              database=DATABASE)
    # 创建数据库游标
    cur = con.cursor()
    id = request.args.get("id", default=1, type=str)
    direction = request.args.get("direction", default=1, type=int)
    sp = request.args.get("speed", default=1, type=int)
    cur.execute("SELECT appkey, appsecret, deviceserial FROM camera WHERE id = %s;", (id))
    con.commit()
    results = cur.fetchall()
    # print(results)
    cur.close()
    rows = []
    for row in results:
        appkey = row[0]
        appsecret = row[1]
        deviceserial = row[2]
        # 创建字典对象，表示每行结果
        row_dict = {
            'appkey': appkey,
            'appsecret': appsecret,
            'deviceserial': deviceserial,
        }
        # 将每行结果的字典添加到列表中
        rows.append(row_dict)
    # print(rows)
    appkey = rows[0]['appkey']
    appsecret = rows[0]['appsecret']
    deviceserial = rows[0]['deviceserial']
    control_start_res = Camera_Control_Start(appkey, appsecret, deviceserial, direction, sp)
    print(control_start_res)
    time.sleep(0.5)
    control_stop_res = Camera_Control_Stop(appkey, appsecret, deviceserial)
    print(control_stop_res)
    return control_stop_res


# 实时数据返回
@agriculture_bp.route("/device/api", methods=['GET'])
def Device_api():
    global con
    if not con.ping():
        con = pymysql.connect(host='localhost', password=PASSWORD, port=PORT, user=USERNAME, charset='utf8',
                              database=DATABASE)
    # 创建数据库游标
    cur = con.cursor()
    device_id = request.args.get("id", default=1, type=str)
    method = request.args.get("method", default=1, type=str)
    cur.execute("SELECT api,business_id, device_id, equipment, version, collect_run FROM device WHERE id = %s;",
                (device_id))
    con.commit()
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
    device_api = res_device[0] + '/' + method + '?' + 'Version=' + res_device[4] + '&Business=' + res_device[
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


# 注册Blueprint到Flask应用中
app.register_blueprint(agriculture_bp)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
    # 关闭数据库连接
    con.close()
    print('the db is closed')
