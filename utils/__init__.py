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
