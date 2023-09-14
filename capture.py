import os
import subprocess
import camera_get
import pymysql
import time


HOSTNAME = "127.0.0.1"
# 端口号
PORT = 3306
# 账号
USERNAME = "root"
# 密码
PASSWORD = "123456"
# 数据库
DATABASE = "eviroment_data"
# 图片存储地址
output_folder = "capture"  # 替换为您希望保存帧的文件夹路径

def extract_frames_from_hls(hls_url, output_folder, frame_index):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 使用FFmpeg命令行抽取帧
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', hls_url,  # 输入HLS视频流地址
        '-vf', 'fps=1',  # 指定帧率，这里设置为每秒抽取1帧
        '-vframes', '1',  # 仅抽取一帧
        os.path.join(output_folder, 'frame-'+str(frame_index)+'.jpg')  # 输出图像文件名格式
    ]

    try:
        # 运行FFmpeg命令行
        subprocess.run(ffmpeg_cmd, check=True)
        print("帧抽取完成！")
    except subprocess.CalledProcessError as e:
        print("Error executing FFmpeg command:", e)


if __name__ == "__main__":
    deviceserial = camera_get.deviceSerial
    hls_url = camera_get.CameraUrl(camera_get.appKey, camera_get.appSecret, deviceserial)
    # hls_url = "https://open.ys7.com/v3/openlive/L42185102_1_2.m3u8?expire=1690253470&id=603539671800094720&t=ed505a56f0571fb6a5551603884d2a100236a97bf96f2ab68eff613cecf09601&ev=100"  # 替换为实际的HLS视频流地址
    con = pymysql.connect(host='localhost', password=PASSWORD, port=PORT, user=USERNAME, charset='utf8',
                          database=DATABASE)
    cur = con.cursor()
    cur.execute("SELECT point_index FROM point WHERE deviceserial = %s", (deviceserial))
    con.commit()

    affected_rows = cur.fetchall()
    i = 0
    for row in affected_rows:
        res = camera_get.Camera_Move_Point(camera_get.appKey, camera_get.appSecret, deviceserial, row[0])
        time.sleep(10)
        extract_frames_from_hls(hls_url, output_folder, i)
        i += 1
    cur.close()


