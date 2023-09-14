# 连接数据库，从接口中获取数据返回数据库

import json
import time

import pymysql
import requests

# 线程池
global THREADS
THREADS = {}


def make_params(version, business, equipment, deviceId, requesttime="1681722094"):
    value = "{ 'page': 1,'length': 5, 'deviceId': " + str(deviceId) + " }"
    params = {
        "Version": str(version),
        "Business": str(business),
        "Equipment": str(equipment),
        "RequestTime": str(requesttime),
        "Value": value
    }
    return params


def insert_sql(API_URL, API_PARAMS, times, DATABASE, ID):
    global THREADS
    THREADS[str(ID)] = 1
    # 向接口发起session
    Url = API_URL
    Params = API_PARAMS
    Database = DATABASE
    # 连接数据库
    con = pymysql.connect(host='localhost', password='123456', port=3306, user='root', charset='utf8',
                          database='eviroment_data')

    for i in range(times):
        if THREADS[str(ID)] == 0:
            print(str(ID) + '线程结束')
            return
        # 发起get请求，获取数据
        resultData = requests.get(Url, params=Params)
        # params格式如下
        ####################################################################################
        # Params = {"Version": "2020-08-10",                                              #
        #            "Business": "1678341180638",                                          #
        #            "Equipment": "AJ0012931",                                             #
        #            "RequestTime": "1681722094",                                          #
        #            "Value": "{ 'page': 1,'length': 5, 'deviceId': 1093892941990735872 }"}#
        ####################################################################################
        # 分析数据
        result = json.loads(resultData.text)
        rows = []
        columns = []
        # print(result)
        for flag in result['data']['deviceDataList']:
            rows.append(flag['value'])
            columns.append(flag['sign'])
        rows.append("'" + result['data']['deviceDataList'][0]['createTime'] + "'")
        columns.append('createTime')

        if not con.ping():
            con = pymysql.connect(host='localhost', password='123456', port=3306, user='root', charset='utf8',
                                  database='eviroment_data')
        # 创建数据库游标
        cur = con.cursor()
        sql_rows_str = ','.join(rows)
        sql_columns_str = ','.join(columns)
        sql_str = 'insert into ' + Database + '(' + sql_columns_str + ') ' + 'values (' + sql_rows_str + ');'
        print(sql_str)
        affected_rows = cur.execute(
            sql_str
        )
        if affected_rows == 1:
            print(str(ID) + '一条数据新增成功')
            # 提交到数据库中
            con.commit()
        # 关闭数据库游标
        cur.close()
        # return (affected_rows == 1)

        # 每一小时请求一次
        time.sleep(3600)
    # 关闭数据库连接
    con.close()


def data_collect(version, business, equipment, deviceId, API_URL, times, DATABASE, ID):
    params = make_params(version, business, equipment, deviceId)
    insert_sql(API_URL, params, times, DATABASE, ID)


def collect_stop(ID):
    global THREADS
    THREADS[str(ID)] = 0;
    print(THREADS)


if __name__ == '__main__':
    day_hours = 24
    days = int(input('请输入您要监测数据天数'))
    params = make_params("2020-08-10", "1678341180638", "AJ0012931", "1093892941990735872")
    insert_sql("http://139.196.24.82/treatment/fields/deviceDataHandler", params, days * day_hours, 'mihoutao39')
