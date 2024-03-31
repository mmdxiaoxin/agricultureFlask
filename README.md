## agricultureFlask

### 项目介绍

该项目是一个基于 Flask 的农业监控系统后端，旨在提供农业数据的采集、预测和用户管理功能。通过该系统，用户可以上传农业数据进行预测。其前端为：[mmdxiaoxin/Agricultural: 使用了Greek-admin模板进行搭建的农业监控系统。 (github.com)](https://github.com/mmdxiaoxin/Agricultural)

**主要特点和功能包括：**

- 使用 Flask 框架搭建，支持 RESTful API。
- 集成了 Flask-CORS 插件，解决跨域访问问题。
- 使用 SQLAlchemy 进行数据库操作。
- 包含了设备、用户、预测等模块，提供了相应的路由处理。
- 实现了模型选择和预测方法选择的功能，用户可以通过接口更改模型和预测方法。
- 支持用户的注册、登录，以及用户上传文件进行预测的功能。

> 以下为接口文档内容

### 用户登录

- **URL：** `/login`
- **请求方法：** POST
- 请求参数：
  - `username` (string): 用户名
  - `password` (string): 密码
- 响应：
  - `code` (int): 响应状态码，200 表示成功，500 表示失败
  - `data` (object): 返回数据，包含 `access_token` 字段
  - `message` (string): 响应信息

### 用户注册

- **URL：** `/register`
- **请求方法：** POST
- 请求参数：
  - `username` (string): 用户名
  - `password` (string): 密码
- 响应：
  - `code` (int): 响应状态码，200 表示成功，500 表示失败
  - `message` (string): 响应信息

### 用户注销

- **URL：** `/logout`
- **请求方法：** POST
- 响应：
  - `code` (int): 响应状态码，200 表示成功
  - `message` (string): 响应信息

### 用户数据导入

- **URL：** `/user/import`
- **请求方法：** POST
- 响应：
  - `code` (int): 响应状态码，200 表示成功
  - `message` (string): 响应信息

### 用户数据导出

- **URL：** `/user/export`
- **请求方法：** POST
- 响应：
  - `code` (int): 响应状态码，200 表示成功
  - `message` (string): 响应信息

### 菜单列表

- **URL：** `/menu/list`
- **请求方法：** GET
- 请求头：
  - `X-Access-Token` (string): 访问令牌
- 响应：
  - `code` (int): 响应状态码，200 表示成功，401 表示未授权
  - `data` (object): 返回数据，包含菜单列表信息
  - `message` (string): 响应信息

### 获取用户设备列表

- **URL：** `/user/deviceList`

- **请求方法：** GET

- 响应：

  - `code` (int): 响应状态码，200 表示成功，500 表示失败

  - ```
    data
    ```

     (list): 设备列表

    - `id` (str): 站点或设备 ID

    - `name` (str): 站点或设备名称

    - `isSite` (bool): 是否为站点

    - ```
      children
      ```

       (list): 子设备列表，每个元素包含以下字段

      - `id` (str): 设备 ID
      - `name` (str): 设备名称
      - `isDevice` (bool): 是否为设备

  - `message` (string): 响应信息

### 返回站点列表

- **URL：** `/address/select`

- **请求方法：** GET

- 响应：

  - `code` (int): 响应状态码，200 表示成功，500 表示失败

  - ```
    data
    ```

     (list): 地址列表，每个元素包含以下字段

    - `id` (int): 地址 ID
    - `name` (str): 地址名称

  - `message` (string): 响应信息

### 返回仪表盘数据

- **URL：** `/device/count`

- **请求方法：** GET

- 响应：

  - `code` (int): 响应状态码，200 表示成功，500 表示失败

  - ```
    data
    ```

     (object): 仪表盘数据

    - `deviceCount` (int): 设备总数

    - `totalDeviceDataCount` (int): 总数据条数

    - `siteCount` (int): 站点总数

    - ```
      siteValues
      ```

       (list): 站点数据列表，每个元素包含以下字段

      - `name` (str): 站点名称
      - `value` (int): 设备数据条数

  - `message` (string): 响应信息

### 返回传感器设备列表（根据地区）

- **URL：** `/device/select`

- **请求方法：** GET

- 请求参数：

  - `address_id` (int): 地址 ID

- 响应：

  - `code` (int): 响应状态码，200 表示成功，500 表示失败

  - ```
    data
    ```

     (list): 设备列表，每个元素包含以下字段

    - `id` (int): 设备 ID
    - `device_name` (str): 设备名称
    - `business_id` (str): 业务 ID
    - `device_id` (str): 设备 ID
    - `collect_run` (str): 数据采集运行状态

  - `message` (string): 响应信息

### 根据参数返回数据库中的数据

- **URL：** `/data/show`
- **请求方法：** GET
- 请求参数：
  - `deviceId` (int): 设备 ID
  - `hour` (int): 小时数，默认为 12
  - `columns` (string): 用户指定的列，逗号分隔
- 响应：
  - `code` (int): 响应状态码，200 表示成功，500 表示失败
  - `data` (list): 数据列表，每个元素为字典，包含请求的列数据
  - `message` (string): 响应信息

### 返回实时数据或者设备属性等

- **URL：** `/device/api`
- **请求方法：** GET
- 请求参数：
  - `id` (int): 设备 ID
  - `method` (str): API 方法名
- 响应：
  - 根据请求的 API 方法返回相应数据，具体格式由 API 决定。

### 更新预测方法

- **URL：** `/predict/predict-method`

- **请求方法：** POST

- 请求参数：

  - `method` (int): 预测方法编号

- 请求体示例：

  ```json
  {
    "method": 0
  }
  ```

- 响应：

  - `code` (int): 响应状态码，200 表示成功，400 表示请求错误
  - `msg` (string): 响应消息

### 更新选择的模型

- **URL：** `/predict/models`

- **请求方法：** POST

- 请求参数：

  - `model` (int): 模型编号

- 请求体示例：

  ```json
  {
    "model": 0
  }
  ```

- 响应：

  - `code` (int): 响应状态码，200 表示成功，400 表示请求错误
  - `msg` (string): 响应消息

### 上传文件进行预测

- **URL：** `/predict/upload-file`

- **请求方法：** POST

- 请求参数：

  - `input_image` (file): 输入图像文件

- 响应：

  - `code` (int): 响应状态码，200 表示成功，400 表示请求错误

  - ```
    data
    ```

     (object): 预测结果数据

    - ```
      predictions
      ```

       (list): 预测结果列表，每个元素包含以下字段

      - `value` (float): 预测结果概率值
      - `name` (string): 预测结果类别名称

  - `msg` (string): 响应消息

