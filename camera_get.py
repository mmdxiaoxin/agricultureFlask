import time

import requests
import json

Token_url = "https://open.ys7.com/api/lapp/token/get"
Camera_get_url = "https://open.ys7.com/api/lapp/v2/live/address/get"
Control_start_url = "https://open.ys7.com/api/lapp/device/ptz/start"
Control_stop_url = "https://open.ys7.com/api/lapp/device/ptz/stop"
Add_point_url = "https://open.ys7.com/api/lapp/device/preset/add"
Move_point_url = "https://open.ys7.com/api/lapp/device/preset/move"
Clear_point_url = "https://open.ys7.com/api/lapp/device/preset/clear"

appKey = '6cbc6189f35149549f661bcb640ce566'
appSecret = '0c9d7ef49593e134bfebbce178f9c926'
deviceSerial = 'L42185102'

def CameraUrl(key, secret, deviceSerial):
    Token_data = {"appKey": key, "appSecret": secret}
    Token_res = requests.post(url=Token_url, data=Token_data)
    Token_result = json.loads(Token_res.text)
    accessToken = Token_result['data']['accessToken']
    Camera_get_data = {"accessToken": accessToken, "deviceSerial": deviceSerial, "protocol": 2, "quality": 2,
                       "supportH265": 0}
    Camera_get_res = requests.post(url=Camera_get_url, data=Camera_get_data)
    Camera_get_result = json.loads(Camera_get_res.text)
    # print(Camera_get_result)
    # print(type(Camera_get_result))
    Camera_url = Camera_get_result['data']['url']
    print(Camera_url)
    return Camera_url

def Camera_Control_Start(key, secret, deviceSerial, direction, speed):
    Token_data = {"appKey": key, "appSecret": secret}
    Token_res = requests.post(url=Token_url, data=Token_data)
    Token_result = json.loads(Token_res.text)
    accessToken = Token_result['data']['accessToken']
    Control_start_data = {"accessToken": accessToken, "deviceSerial":deviceSerial, "direction":direction, "speed":speed, "channelNo" : 1}
    Control_start = requests.post(url=Control_start_url, data=Control_start_data)
    res = json.loads(Control_start.text)
    return res

def Camera_Control_Stop(key, secret, deviceSerial):
    Token_data = {"appKey": key, "appSecret": secret}
    Token_res = requests.post(url=Token_url, data=Token_data)
    Token_result = json.loads(Token_res.text)
    accessToken = Token_result['data']['accessToken']
    Control_stop_data = {"accessToken": accessToken, "deviceSerial":deviceSerial, "channelNo": 1}
    Control_stop = requests.post(url=Control_stop_url, data=Control_stop_data)
    res = json.loads(Control_stop.text)
    return res

def Camera_Add_Point(key, secret, deviceSerial):
    Token_data = {"appKey": key, "appSecret": secret}
    Token_res = requests.post(url=Token_url, data=Token_data)
    Token_result = json.loads(Token_res.text)
    accessToken = Token_result['data']['accessToken']
    Add_point_data = {"accessToken" : accessToken, "deviceSerial":deviceSerial, "channelNo": 1}
    Add_point = requests.post(url=Add_point_url, data=Add_point_data)
    res = json.loads(Add_point.text)
    print(res)
    print(res['data']['index'])
    return res['data']['index']

def Camera_Move_Point(key, secret, deviceSerial, index):
    Token_data = {"appKey": key, "appSecret": secret}
    Token_res = requests.post(url=Token_url, data=Token_data)
    Token_result = json.loads(Token_res.text)
    accessToken = Token_result['data']['accessToken']
    Move_point_data = {"accessToken" : accessToken, "deviceSerial":deviceSerial, "channelNo": 1, "index" : index}
    Move_point = requests.post(url=Move_point_url, data=Move_point_data)
    res = json.loads(Move_point.text)
    print(res)
    return res

def Camera_Clear_Point(key, secret, deviceSerial, index):
    Token_data = {"appKey": key, "appSecret": secret}
    Token_res = requests.post(url=Token_url, data=Token_data)
    Token_result = json.loads(Token_res.text)
    accessToken = Token_result['data']['accessToken']
    Clear_point_data = {"accessToken" : accessToken, "deviceSerial":deviceSerial, "channelNo": 1, "index" : index}
    Clear_point = requests.post(url=Clear_point_url, data=Clear_point_data)
    res = json.loads(Clear_point.text)
    print(res)
    return res

if __name__ == "__main__":
    CameraUrl(appKey, appSecret, deviceSerial)
    # for i in range(3):
    #     Camera_Control_Start(appKey, appSecret, deviceSerial, 3, 1)
    #     time.sleep(1)
    #     Camera_Control_Stop(appKey, appSecret, deviceSerial)
    #     Camera_Add_Point(appKey, appSecret, deviceSerial)
    #
    # Camera_Clear_Point(appKey, appSecret, deviceSerial, 2)
    # Camera_Clear_Point(appKey, appSecret, deviceSerial, 3)





