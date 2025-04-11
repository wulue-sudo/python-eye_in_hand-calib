# --- coding: utf-8 ---
# @Time    : 3/5/25 7:13 PM        # 文件创建时间
# @Author  : htLiang
# @Email   : ryzeliang@163.com

#机器人与3D相机协同工作的数据采集功能
#控制一个UR5机器人移动到3D工作空间内的多个特定位置，并在每个位置采集机器人的TCP（工具中心点）位姿数据以及相机的图像数据。
import rtde_receive
import pyrealsense2 as rs
import numpy as np
import time
import cv2
from ur_ import UR_Robot
import os
import rtde_control
rtde_c = rtde_control.RTDEControlInterface("192.168.3.10")
# 初始化路径
img_path = 'IMG'
rt_path = 'RT'
os.makedirs(img_path, exist_ok=True)                 # 创建图像保存路径，如果目录已经存在，exist_ok=True参数会防止抛出异常
os.makedirs(rt_path, exist_ok=True)

# 初始化RealSense相机
pipeline = rs.pipeline()                                                                   # 创建RealSense管道，用于管理相机设备的流数据
config = rs.config()                                                                       # 创建配置对象，用于管理相机参数
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)                        #配置RealSense相机以启用彩色流（rs.stream.color），设置分辨率为640x480像素，图像格式为BGR8（即每个像素用3个8位的值表示，分别对应蓝、绿、红三个通道），帧率为每秒30帧
pipeline.start(config)                                                                     # 启动RealSense相机

# 数据采集函数
def capture_data(index):
    # 获取机器人位姿
    tcp_pose = robot.get_cartesian_info()# rx ry rz 罗德里格斯形式 
    np.save(os.path.join(rt_path, f'{index}.npy'), tcp_pose)               #os.path.join()函数来生成文件的完整路径，rt_path+index.npy表示保存的文件名，tcp_pose表示要保存的数据

    # 获取相机图像
    frames = pipeline.wait_for_frames()                                    # 等待并获取帧集
    color_frame = frames.get_color_frame()                                 # 获取彩色图像帧
    color_image = np.asanyarray(color_frame.get_data())                    # 将彩色图像帧数据转换为NumPy数组
    cv2.imwrite(os.path.join(img_path, f'{index}.png'), color_image)       # 保存图像到文件
    print(f"Captured data pair {index}")



# # User options
# # --------------- Setup options ---------------
# tcp_host_ip = '192.168.243.101'  # IP and port to robot arm as TCP client (UR5)
# workspace_limits = np.asarray([[0.45, 0.55], [0.1, 0.15], [0.15, 0.2]])
# calib_grid_step = 0.02  # 0.05
# tool_orientation = [-np.pi,0,-np.pi/2]  # [0,-2.22,2.22] # [2.22,2.22,0] rpy
# # ---------------------------------------------
# # Construct 3D calibration grid across workspace
# print(1 + (workspace_limits[0][1] - workspace_limits[0][0]) / calib_grid_step)
# gridspace_x = np.linspace(workspace_limits[0][0], workspace_limits[0][1],
#                           int(1 + (workspace_limits[0][1] - workspace_limits[0][0]) / calib_grid_step))
# gridspace_y = np.linspace(workspace_limits[1][0], workspace_limits[1][1],
#                           int(1 + (workspace_limits[1][1] - workspace_limits[1][0]) / calib_grid_step))
# gridspace_z = np.linspace(workspace_limits[2][0], workspace_limits[2][1],
#                           int(1 + (workspace_limits[2][1] - workspace_limits[2][0]) / calib_grid_step))
# calib_grid_x, calib_grid_y, calib_grid_z = np.meshgrid(gridspace_x, gridspace_y, gridspace_z)
# num_calib_grid_pts = calib_grid_x.shape[0] * calib_grid_x.shape[1] * calib_grid_x.shape[2]
#
# calib_grid_x.shape = (num_calib_grid_pts, 1)
# calib_grid_y.shape = (num_calib_grid_pts, 1)
# calib_grid_z.shape = (num_calib_grid_pts, 1)
# calib_grid_pts = np.concatenate((calib_grid_x, calib_grid_y, calib_grid_z), axis=1)
#
# measured_pts = []
# observed_pts = []
# observed_pix = []
#
# print('Connecting to robot...')
# robot = UR_Robot(tcp_host_ip,  workspace_limits, is_use_robotiq85=False)
# # robot.open_gripper()
#
# # Slow down robot
# robot.joint_acc = 1.4
# robot.joint_vel = 1.05
#
# # Move robot to each calibration point in workspace
# print('Collecting data...')
# for calib_pt_idx in range(num_calib_grid_pts):
#     tool_position = calib_grid_pts[calib_pt_idx, :]
#     tool_config = [tool_position[0], tool_position[1], tool_position[2],
#                    tool_orientation[0], tool_orientation[1], tool_orientation[2]]
#     tool_config1 = [tool_position[0], tool_position[1], tool_position[2],
#                     tool_orientation[0], tool_orientation[1], tool_orientation[2]]
#     print(f"tool position and orientation:{tool_config1}")
#     robot.move_j_p(tool_config)
#     time.sleep(2)  # k
#     capture_data(calib_pt_idx)

import numpy as np
import time

# 预设参数
tcp_host_ip = '192.168.3.10'  # IP和端口，机器人控制器额的IP地址
workspace_limits = np.asarray([[-0.05, 0], [-0.35, -0.30], [0.35, 0.40]])        #定义了机器人工作空间的范围，是一个3x2的NumPy数组，表示x、y、z三个方向上的最小值和最大值。
calib_grid_step = 0.02                                                           #标定网格的步长
tool_orientation = [2 * np.pi / 3, 2 * np.pi / 3, 0]                                       #工具的姿态，表示x、y、z三个方向的旋转角度（绕X轴-180度，Y轴无旋转，Z轴旋转-90度）

# 构造工作空间内的3D标定网格
print(1 + (workspace_limits[0][1] - workspace_limits[0][0]) / calib_grid_step)                                     #打印在x方向上，标定网格的格点数
gridspace_x = np.linspace(workspace_limits[0][0], workspace_limits[0][1],
                          int(1 + (workspace_limits[0][1] - workspace_limits[0][0]) / calib_grid_step))            #使用linspace()函数生成在x方向上的标定网格，参数分别是起始值、结束值和生成的点的数量。
gridspace_y = np.linspace(workspace_limits[1][0], workspace_limits[1][1],
                          int(1 + (workspace_limits[1][1] - workspace_limits[1][0]) / calib_grid_step))  
gridspace_z = np.linspace(workspace_limits[2][0], workspace_limits[2][1],
                          int(1 + (workspace_limits[2][1] - workspace_limits[2][0]) / calib_grid_step))
calib_grid_x, calib_grid_y, calib_grid_z = np.meshgrid(gridspace_x, gridspace_y, gridspace_z)                      #生成3D标定网格的点坐标
num_calib_grid_pts = calib_grid_x.size                                                                             #计算标定网格的点数

calib_grid_x = calib_grid_x.reshape((num_calib_grid_pts, 1))                                                       #将标定网格的点坐标转换成一个1xN的数组
calib_grid_y = calib_grid_y.reshape((num_calib_grid_pts, 1))
calib_grid_z = calib_grid_z.reshape((num_calib_grid_pts, 1))                                      
calib_grid_pts = np.concatenate((calib_grid_x, calib_grid_y, calib_grid_z), axis=1)                                 #将标定网格的点坐标合并成一个3xN的数组

measured_pts = []
observed_pts = []
observed_pix = []

print('Connecting to robot...')
print(f"采样次数：{num_calib_grid_pts}")
robot = UR_Robot(tcp_host_ip, workspace_limits, is_use_robotiq85=False)
# robot.open_gripper()

# 调整机器人的运动参数
robot.joint_acc = 0.1
robot.joint_vel = 0.07
# 移动机器人到标定网格的每个点，并在每个点采集数据
print('Collecting data...')
for calib_pt_idx in range(num_calib_grid_pts):
    tool_position = calib_grid_pts[calib_pt_idx, :]

    # 添加小的随机旋转扰动（单位：弧度）
    random_rotation = np.random.uniform(-0.5, 0.5, 3)                       #生成一个3x1的随机数组，范围是-0.1到0.1，表示随机扰动的角度

    modified_orientation = [tool_orientation[0] + random_rotation[0],
                            tool_orientation[1] + random_rotation[1],
                            tool_orientation[2] + random_rotation[2]]

    # 构造工具的位姿配置：[x, y, z, roll, pitch, yaw]
    tool_config = [tool_position[0], tool_position[1], tool_position[2]] + modified_orientation
    print(f"tool position and orientation: {tool_config}")

    # 移动机器人到新的配置点
    # robot.move_l(tool_config)
    rtde_c.moveL(tool_config, 0.1, 0.07)
    time.sleep(2)  # 等待机器人运动到位
    capture_data(calib_pt_idx)


