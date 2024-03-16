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
    source_cursor.execute("SELECT * FROM shucai44")

    # 插入数据到目标数据库中
    for row in source_cursor.fetchall():
        data = {
            "AL1": row[0],
            "AB1": row[1],
            "AA1": row[2],
            "AH1": row[3],
            "AI1": row[4],
            "BD1": row[5],
            "createTime": row[6]
        }
        # 手动构造设备ID为2的数据插入语句
        insert_query = "INSERT INTO agri_sensor_data (device_id, AL1, AB1, AA1, AH1, AI1, BD1, createTime) VALUES (4, %(AL1)s, %(AB1)s, %(AA1)s, %(AH1)s, %(AI1)s, %(BD1)s, %(createTime)s)"
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
