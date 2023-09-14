# 连接数据库，从接口中获取数据返回数据库

import sys
import time

import requests
import json
import pymysql
# reload(sys)
# sys.setdefaultencoding('utf-8')


def insert_sql(times):
    # 向接口发起session
    url = "http://139.196.24.82/treatment/fields/deviceDataHandler"
    # 白水苹果
    Params1 = {"Version":"2020-08-10",
            "Business":"1678341180638",
            "Equipment":"AJ0012931",
            "RequestTime":"1681722094",
            "Value":"{ 'page': 1,'length': 5, 'deviceId': 1093904346861154304 }"}
    # 武功猕猴桃
    Params2 = {"Version": "2020-08-10",
               "Business": "1678341180638",
               "Equipment": "AJ0012931",
               "RequestTime": "1681722094",
               "Value": "{ 'page': 1,'length': 5, 'deviceId': 1093892941990735872 }"}
    # 临渭葡萄
    Params3 = {"Version": "2020-08-10",
               "Business": "1678341180638",
               "Equipment": "AJ0012931",
               "RequestTime": "1681722094",
               "Value": "{ 'page': 1,'length': 5, 'deviceId': 1093567370681266176 }"}
    # 泾阳蔬菜
    Params4 = {"Version": "2020-08-10",
               "Business": "1678341180638",
               "Equipment": "AJ0012931",
               "RequestTime": "1681722094",
               "Value": "{ 'page': 1,'length': 5, 'deviceId': 1093497790671761408 }"}
    # ss = requests.session()

    # 连接数据库
    con = pymysql.connect(host='localhost', password='123456', port=3306, user='root', charset='utf8',
                          database='eviroment_data')

    for i in range(times):
        # 发起get请求，获取数据
        resultData_1 = requests.get(url, params=Params1)
        resultData_2 = requests.get(url, params=Params2)
        resultData_3 = requests.get(url, params=Params3)
        resultData_4 = requests.get(url, params=Params4)
        # 分析数据
# 1
        result = json.loads(resultData_1.text)
        print(result)
        AD1 = result['data']['deviceDataList'][0]['value']
        AL1 = result['data']['deviceDataList'][1]['value']
        AF1 = result['data']['deviceDataList'][2]['value']
        AE1 = result['data']['deviceDataList'][3]['value']
        AC1 = result['data']['deviceDataList'][4]['value']
        AB1 = result['data']['deviceDataList'][5]['value']
        AA1 = result['data']['deviceDataList'][6]['value']
        AH1 = result['data']['deviceDataList'][7]['value']
        AI1 = result['data']['deviceDataList'][8]['value']
        AJ1 = result['data']['deviceDataList'][9]['value']
        Create_time = result['data']['deviceDataList'][4]['createTime']

        if not con.ping():
            con = pymysql.connect(host='localhost', password='123456', port=3306, user='root', charset='utf8',
                                  database='eviroment_data')
        # 创建数据库游标
#         cur = con.cursor()
#
#         affected_rows = cur.execute(
#             'insert into PINGGUO(AD1,AL1,AF1,AE1,AC1,AB1,AA1,AH1,AI1,AJ1,createTime) values (%s, %s, %s, %s, %s, %s, %s,%s, %s, %s, %s)',(AD1,AL1,AF1,AE1,AC1,AB1,AA1,AH1,AI1,AJ1,Create_time)
#         )
#         if affected_rows == 1:
#             print('站点1新增成功')
#             # 提交到数据库中
#             con.commit()
#         # 关闭数据库游标
#         cur.close()
# # 2
#         result = json.loads(resultData_2.text)
#         # print(result)
#         AD1 = result['data']['deviceDataList'][0]['value']
#         AL1 = result['data']['deviceDataList'][1]['value']
#         AF1 = result['data']['deviceDataList'][2]['value']
#         AE1 = result['data']['deviceDataList'][3]['value']
#         AC1 = result['data']['deviceDataList'][4]['value']
#         AB1 = result['data']['deviceDataList'][5]['value']
#         AA1 = result['data']['deviceDataList'][6]['value']
#         AH1 = result['data']['deviceDataList'][7]['value']
#         AI1 = result['data']['deviceDataList'][8]['value']
#         AJ1 = result['data']['deviceDataList'][9]['value']
#         Create_time = result['data']['deviceDataList'][4]['createTime']
#
#         # 创建数据库游标
#         cur = con.cursor()
#
#         affected_rows = cur.execute(
#             'insert into MIHOUTAO(AD1,AL1,AF1,AE1,AC1,AB1,AA1,AH1,AI1,AJ1,createTime) values (%s, %s, %s, %s, %s, %s, %s,%s, %s, %s, %s)',(AD1,AL1,AF1,AE1,AC1,AB1,AA1,AH1,AI1,AJ1,Create_time)
#         )
#         if affected_rows == 1:
#             print('站点2新增成功')
#             # 提交到数据库中
#             con.commit()
#         # 关闭数据库游标
#         cur.close()
# # 3
#         result = json.loads(resultData_3.text)
#         # print(result)
#         AA1 = result['data']['deviceDataList'][0]['value']
#         AH1 = result['data']['deviceDataList'][1]['value']
#         AI1 = result['data']['deviceDataList'][2]['value']
#         AL1 = result['data']['deviceDataList'][3]['value']
#         AB1 = result['data']['deviceDataList'][4]['value']
#         BD1 = result['data']['deviceDataList'][5]['value']
#         Create_time = result['data']['deviceDataList'][4]['createTime']
#
#         # 创建数据库游标
#         cur = con.cursor()
#
#         affected_rows = cur.execute(
#             'insert into PUTAO(AA1,AH1,AI1,AL1,AB1,BD1,createTime) values (%s, %s, %s, %s, %s, %s, %s)',
#             (AA1, AH1, AI1, AL1, AB1, BD1, Create_time)
#         )
#         if affected_rows == 1:
#             print('站点3新增成功')
#             # 提交到数据库中
#             con.commit()
#         # 关闭数据库游标
#         cur.close()
# # 4
#         result = json.loads(resultData_4.text)
#         # print(result)
#         AA1 = result['data']['deviceDataList'][0]['value']
#         AH1 = result['data']['deviceDataList'][1]['value']
#         AI1 = result['data']['deviceDataList'][2]['value']
#         AL1 = result['data']['deviceDataList'][3]['value']
#         AB1 = result['data']['deviceDataList'][4]['value']
#         BD1 = result['data']['deviceDataList'][5]['value']
#         Create_time = result['data']['deviceDataList'][4]['createTime']
#
#         # 创建数据库游标
#         cur = con.cursor()
#
#         affected_rows = cur.execute(
#             'insert into SHUCAI(AA1,AH1,AI1,AL1,AB1,BD1,createTime) values (%s, %s, %s, %s, %s, %s, %s)',
#             (AA1, AH1, AI1, AL1, AB1, BD1, Create_time)
#         )
#         if affected_rows == 1:
#             print('站点4新增成功')
#             # 提交到数据库中
#             con.commit()
#         # 关闭数据库游标
#         cur.close()


        # 每90秒请求一次
        time.sleep(1)

    # 关闭session
    # ss.close()

    # 关闭数据库连接
    con.close()

if __name__ == '__main__':
    day_seconds = 86400
    days = int(input('请输入您要监测数据天数'))
    insert_sql(days * day_seconds)