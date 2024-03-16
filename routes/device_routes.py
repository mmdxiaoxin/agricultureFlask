import time
from datetime import timedelta

import requests
from flask import Blueprint
from flask import request, jsonify

from model import AgriDevice, AgriSite, AgriSensorData

device_bp = Blueprint('device_bp', __name__)


# 获取用户设备列表
@device_bp.route("/user/deviceList", methods=['GET'])
def get_device_list():
    try:
        # 查询所有站点
        sites = AgriSite.query.all()

        # 查询每个站点的设备
        device_list = []

        for site in sites:
            site_info = {
                'id': str(site.id),
                'name': site.site_name,
                'isSite': True,  # 添加isSite字段
                'children': [{'id': str(device.id), 'name': device.device_name, 'isDevice': True} for device in
                             site.devices]
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


# 返回站点列表
@device_bp.route("/address/select", methods=['GET'])
def address_select():
    try:
        # 查询所有地址
        sites = AgriSite.query.all()

        # 构建响应数据
        rows = [{'id': site.id, 'name': site.site_name} for site in sites]
        response_data = {'code': 200, 'data': rows, 'message': 'Success'}
        return jsonify(response_data)
    except Exception as e:
        response_data = {'code': 500, 'message': f'Server Error: {str(e)}'}
        return jsonify(response_data), 500


# 返回仪表盘数据
@device_bp.route('/device/count', methods=['GET'])
def get_site_device_data_count():
    try:
        # 查询所有站点
        sites = AgriSite.query.all()

        # 初始化响应数据
        response_data = {
            'code': 200,
            'message': '成功',
            'data': {
                'deviceCount': 0,
                'totalDeviceDataCount': 0,
                'siteCount': len(sites),
                'siteValues': []
            }
        }

        total_device_count = 0
        total_device_data_count = 0

        for site in sites:
            # 获取站点名称
            site_name = site.site_name

            # 查询站点对应设备的总数
            device_count = AgriDevice.query.filter_by(site_id=site.id).count()

            # 获取站点对应设备的总数据条数
            device_data_count = AgriSensorData.query.join(AgriDevice).filter(AgriDevice.site_id == site.id).count()

            # 更新总设备数和总数据条数
            total_device_count += device_count
            total_device_data_count += device_data_count

            # 构建站点数据信息
            site_info = {
                'name': site_name,
                'value': device_data_count
            }

            # 添加到响应数据中
            response_data['data']['siteValues'].append(site_info)

        # 更新响应数据中的总设备数和总数据条数
        response_data['data']['deviceCount'] = total_device_count
        response_data['data']['totalDeviceDataCount'] = total_device_data_count

        return jsonify(response_data)

    except Exception as e:
        # 处理异常
        response_data = {
            'code': 500,
            'message': '服务器错误: ' + str(e),
            'data': {}
        }
        return jsonify(response_data)


# 返回传感器设备列表（根据地区）
@device_bp.route("/device/select", methods=['GET'])
def device_select():
    try:
        # 获取请求参数
        address_id = request.args.get("address_id", default=1, type=int)

        # 查询符合条件的设备列表
        devices = AgriDevice.query.filter_by(site_id=address_id).all()

        # 构建响应数据
        rows = [{'id': device.id, 'device_name': device.device_name,
                 'business_id': device.business_id, 'device_id': device.device_id,
                 'collect_run': device.collect_run} for device in devices]

        response_data = {'code': 200, 'data': rows, 'message': 'Success'}
        return jsonify(response_data)
    except Exception as e:
        response_data = {'code': 500, 'message': f'Server Error: {str(e)}'}
        return jsonify(response_data), 500


# 根据参数返回数据库中的数据
@device_bp.route("/data/show", methods=['GET'])
def data_base():
    try:
        device_id = request.args.get("deviceId", default=1, type=int)
        hour = request.args.get("hour", default=12, type=int)  # 获取传递的小时数，默认为12
        columns = request.args.get("columns", None)  # 获取用户指定的列

        # 查询特定设备的最新数据的时间
        latest_data_time = AgriSensorData.query.filter_by(device_id=device_id).order_by(
            AgriSensorData.createTime.desc()).first().createTime

        # 计算起始时间，即最新数据时间往前推12小时
        start_time = latest_data_time - timedelta(hours=hour)

        # 构建查询条件，即设备ID和时间范围
        query = AgriSensorData.query.filter_by(device_id=device_id)

        # 添加时间范围过滤条件
        query = query.filter(AgriSensorData.createTime >= start_time)

        # 执行查询
        datas = query.all()

        # 获取用户指定的列，如果没有指定则返回所有列
        columns_to_select = columns.split(",") if columns else [column.name for column in
                                                                AgriSensorData.__table__.columns]
        # 构建响应数据
        response_data = {
            'code': 200,
            'message': '成功',
            'data': [data.to_dict(columns_to_select) for data in datas]  # 将查询结果转换为字典列表
        }
        return jsonify(response_data)

    except Exception as e:
        # 处理异常情况
        response_data = {
            'code': 500,
            'message': '服务器错误: {}'.format(str(e))
        }
        return jsonify(response_data), 500


# 返回实时数据或者设备属性等等
@device_bp.route("/device/api", methods=['GET'])
def device_api():
    try:
        device_id = request.args.get("id", default=1, type=int)
        api_method = request.args.get("method", default=1, type=str)

        # 查询设备信息
        device = AgriDevice.query.filter_by(id=device_id).first()
        if not device:
            return jsonify({'code': 404, 'message': '未查询到此设备'}), 404

        # 检查权限
        if device.collect_run == '0':
            return jsonify({'code': 403, 'message': '没有访问权限'}), 403

        # 构建请求参数
        params = {
            "Version": device.version,
            "Business": device.business_id,
            "Equipment": device.equipment,
            "RequestTime": str(int(time.time())),
            "Value": '{ "page": 1, "length": 5, "deviceId": ' + str(device.device_id) + ' }'
        }

        # 发送请求
        response = requests.get(device.api + '/' + api_method, params=params)
        result = response.json()

        return jsonify(result)

    except Exception as e:
        return jsonify({'code': 500, 'message': f'服务器错误: {str(e)}'}), 500
