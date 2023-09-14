## 农业监控系统 API 文档

### 欢迎页面

- **URL:** /
- **请求方法:** GET
- **描述:** 获取欢迎消息。

### 用户登录

- **URL:** /login
- **请求方法:** POST
- **描述:** 用户登录接口，根据用户名和密码验证用户身份并返回访问令牌。
- 请求参数:
  - `username` (字符串) - 用户名。
  - `password` (字符串) - 密码。
- 响应:
  - `code` (整数) - 响应状态码。
  - `data` (对象) - 响应数据对象，包含访问令牌 (`access_token`)。
  - `message` (字符串) - 响应消息。

### 用户登出

- **URL:** /logout
- **请求方法:** POST
- **描述:** 用户登出接口，注销用户。

### 导入用户数据

- **URL:** /user/import
- **请求方法:** POST
- **描述:** 导入用户数据。

### 导出用户数据

- **URL:** /user/export
- **请求方法:** POST
- **描述:** 导出用户数据。

### 获取地区列表

- **URL:** /agriculture/address/select
- **请求方法:** GET
- **描述:** 获取所有地区列表。
- 响应:
  - `code` (整数) - 响应状态码。
  - `data` (数组) - 包含地区信息的数组。
  - `message` (字符串) - 响应消息。

### 获取传感器设备列表

- **URL:** /agriculture/device/select
- **请求方法:** GET
- **描述:** 根据地区获取传感器设备列表。
- 请求参数:
  - `address_id` (字符串) - 地区 ID。
- 响应:
  - `code` (整数) - 响应状态码。
  - `data` (数组) - 包含传感器设备信息的数组。
  - `message` (字符串) - 响应消息。

### 获取设备列表

- **URL:** /agriculture/user/deviceList
- **请求方法:** GET
- **描述:** 获取用户的设备列表。
- 响应:
  - `code` (整数) - 响应状态码。
  - `data` (数组) - 包含设备信息的数组。
  - `message` (字符串) - 响应消息。

### 获取菜单列表

- **URL:** /agriculture/menu/list
- **请求方法:** GET
- **描述:** 获取用户菜单列表。
- 请求头:
  - `X-Access-Token` (字符串) - 用户访问令牌。
- 响应:
  - `code` (整数) - 响应状态码。
  - `data` (数组) - 包含菜单信息的数组。
  - `message` (字符串) - 响应消息。

### 获取历史数据

- **URL:** /agriculture/data/show
- **请求方法:** GET
- **描述:** 获取最新的 100 条历史数据。
- 请求参数:
  - `deviceId` (字符串) - 设备 ID。
- 响应:
  - `code` (整数) - 响应状态码。
  - `data` (数组) - 包含历史数据的数组。
  - `message` (字符串) - 响应消息。

### 获取摄像头列表

- **URL:** /agriculture/camera/select
- **请求方法:** GET
- **描述:** 获取摄像头列表。
- 请求参数:
  - `address_id` (字符串) - 地区 ID。
- 响应:
  - `code` (整数) - 响应状态码。
  - `data` (数组) - 包含摄像头信息的数组。
  - `message` (字符串) - 响应消息。

### 获取摄像头 URL

- **URL:** /agriculture/camera/url
- **请求方法:** GET
- **描述:** 获取摄像头的 URL。
- 请求参数:
  - `id` (字符串) - 摄像头 ID。
- 响应:
  - `code` (整数) - 响应状态码。
  - `data` (对象) - 包含摄像头 URL 信息。
  - `message` (字符串) - 响应消息。

### 控制摄像头

- **URL:** /agriculture/camera/control
- **请求方法:** GET
- **描述:** 控制摄像头的方向和速度。
- 请求参数:
  - `id` (字符串) - 摄像头 ID。
  - `direction` (整数) - 控制方向。
  - `speed` (整数) - 控制速度。
- 响应:
  - `code` (整数) - 响应状态码。
  - `data` (对象) - 包含摄像头控制结果信息。
  - `message` (字符串) - 响应消息。

### 设备 API 请求

- **URL:** /agriculture/device/api
- **请求方法:** GET
- **描述:** 发送设备 API 请求。
- 请求参数:
  - `id` (字符串) - 设备 ID。
  - `method` (字符串) - API 请求方法。
- 响应:
  - `code` (整数) - 响应状态码。
  - `data` (对象) - 包含设备 API 响应数据。
  - `message` (字符串) - 响应消息。

### 关于项目

- **URL:** /about/index
- **请求方法:** GET
- **描述:** 获取关于项目信息。