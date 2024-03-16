import pymysql

# 连接源数据库
source_db_connection = pymysql.connect(
    host="localhost",
    user="root",
    password="122600",
    database="eviroment_data"
)

# 连接目标数据库
target_db_connection = pymysql.connect(
    host="localhost",
    user="root",
    password="122600",
    database="agri_monitor_db"
)

# 创建游标对象
source_cursor = source_db_connection.cursor()
target_cursor = target_db_connection.cursor()

try:
    # 查询源数据库中的数据
    source_cursor.execute("SELECT * FROM pingguo42")

    # 插入数据到目标数据库中
    for row in source_cursor.fetchall():
        data = {
            "AD1": row[0],
            "AL1": row[1],
            "AF1": row[2],
            "AE1": row[3],
            "AB1": row[4],
            "AA1": row[5],
            "AH1": row[6],
            "AI1": row[7],
            "AC1": row[8],
            "AJ1": row[9],
            "createTime": row[10]
        }
        # 手动构造设备ID为1的数据插入语句
        insert_query = "INSERT INTO agri_sensor_data (device_id, AD1, AL1, AF1, AE1, AB1, AA1, AH1, AI1, AC1, AJ1, createTime) VALUES (3, %(AD1)s, %(AL1)s, %(AF1)s, %(AE1)s, %(AB1)s, %(AA1)s, %(AH1)s, %(AI1)s, %(AC1)s, %(AJ1)s, %(createTime)s)"
        target_cursor.execute(insert_query, data)

    # 提交事务
    target_db_connection.commit()
    print("Data migration successful!")

except pymysql.Error as error:
    print("Error:", error)
    # 回滚事务
    target_db_connection.rollback()

finally:
    # 关闭连接和游标
    source_cursor.close()
    target_cursor.close()
    source_db_connection.close()
    target_db_connection.close()
