import os
import cv2
import xlrd2
from math import *
import numpy as np
import glob
import csv
import re
import time 
from scipy.spatial.transform import Rotation as R

def save_matrices_to_csv(matrices, file_name):

    rows, cols = matrices[0].shape
    num_matrices = len(matrices)
    combined_matrix = np.zeros((rows, cols * num_matrices))

    for i, matrix in enumerate(matrices):
        combined_matrix[:, i * cols: (i + 1) * cols] = matrix

    with open(file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in combined_matrix:
            csv_writer.writerow(row)


#欧拉角转换为旋转矩阵
def euler_angles_to_rotation_matrix(rx, ry, rz):
    # 计算旋转矩阵
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])

    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])

    R = Rz @ Ry @ Rx

    return R

#位姿转换为齐次矩阵
def pose_to_homogeneous_matrix(pose):
    x, y, z, rx, ry, rz = pose
    R = euler_angles_to_rotation_matrix(rx, ry, rz)
    t = np.array([x, y, z]).reshape(3, 1)

    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t[:, 0]

    return H

if __name__ == "__main__":
    path = os.path.dirname(__file__)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)         #亚像素角点检测的停止准则，cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS表示终止条件是最大迭代次数达到指定值或者误差小于指定值时停止

    # 获取标定板角点的位置
    objp = np.zeros((11 * 8, 3), np.float32)            #标定板为12*9,这里就输入11*8
    objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y

    obj_points = []  # 存储3D点
    img_points = []  # 存储2D点

    images = glob.glob("./IMG/*.png")  # 修改成自己的图片路径
    print(f"images = {images}")
    count = 0
    for i, fname in enumerate(images):
        print(f"{i}/{len(images) - 1}")
        img = cv2.imread(fname)
        cv2.imshow('img', img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
        print(ret)
        
        if ret:

            obj_points.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
            # print(corners2)
            if [corners2]:
                img_points.append(corners2)
            else:
                img_points.append(corners)

            cv2.drawChessboardCorners(img, (11, 8), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
            cv2.imshow('img', img)
            # time.sleep(0.5)
            cv2.waitKey(20)
        else:
            print(f"chessboard not found: {fname}")

    print(len(img_points))
    print(f"img_points length: {len(img_points)}")
    # print(f"img_points: {img_points}")
    cv2.destroyAllWindows()

    # 标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)   #得到图案在相机坐标系下的位姿,rvecs 和 tvecs 分别是每张图像的旋转向量和平移向量，表示标定板在相机坐标系下的位姿
    print(type(rvecs))
    rvecs = np.array(rvecs)
    print(rvecs.shape)
    rotation_matrices = []
    for rvec in rvecs:
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        print(f"图像的旋转矩阵：{rotation_matrix.shape}")
        rotation_matrices.append(rotation_matrix)
        matrix_transpose = rotation_matrix.T
        # 计算矩阵与其转置的乘积
        product = np.dot(matrix_transpose, rotation_matrix)
        # 定义单位矩阵
        identity_matrix = np.eye(3)
        # 计算乘积矩阵与单位矩阵的误差
        error = np.linalg.norm(product - identity_matrix)
        if error > 1e-4:
            print("该矩阵不是正交矩阵。")
    translation_vectors = []    
    for tvec in tvecs:
        translation_vectors.append(tvec)

    pose_path = "/home/wulue/ur-realsense-eye-in-hand-calibration/all_data.xlsx"
    data = xlrd2.open_workbook(pose_path)                             # 打开excel文件
    # print(data)
    table = data.sheets()[0]                                          # 获取 Excel 文件中的第一个工作表，并对这个工作表进行数据读取和处理
    # print(table)
    N = table.nrows                                                   # 读取Excel文件中除标题行外的数据行数，即位姿数据总数
    matrices = []
    for row in range(table.nrows):                                    #table.nrows 表示 Excel 表格的总行数，table.ncols 表示 Excel 表格的总列数
        x = table.cell_value(row, 0)
        y = table.cell_value(row, 1)
        z = table.cell_value(row, 2)
        tx = table.cell_value(row, 3)
        ty = table.cell_value(row, 4)
        tz = table.cell_value(row, 5)
        pose = [x, y, z, tx, ty, tz]
        pose = pose_to_homogeneous_matrix(pose)
        matrices.append(pose)

    save_matrices_to_csv(matrices, f'RobotToolPose.csv')
    save_matrices_to_csv(rotation_matrices, f'cameraPose.csv')
    save_matrices_to_csv(translation_vectors, f'translationcameraPose.csv')
    csv_file = os.path.join(path,"RobotToolPose.csv")
    tool_pose = np.loadtxt(csv_file,delimiter=',')

    R_tool = []
    t_tool = []

    for i in range(int(N)):

        R_tool.append(tool_pose[0:3,4*i:4*i+3])
        t_tool.append(tool_pose[0:3,4*i+3].reshape(-1, 1))
        #验证旋转矩阵正交性
        R = tool_pose[0:3,4*i:4*i+3]
        print(R.shape)
        matrix_transpose = R.T
        # 计算矩阵与其转置的乘积
        product = np.dot(matrix_transpose, R)
        # 定义单位矩阵
        identity_matrix = np.eye(3)
        # 计算乘积矩阵与单位矩阵的误差
        error = np.linalg.norm(product - identity_matrix)
        if error > 1e-4:
            print("该矩阵不是正交矩阵。")
        # t_tool.append(tool_pose[0:3,4*i+3])


    print(f"R_tool 的大小是: {np.array(R_tool).shape}")
    print(f"t_tool 的大小是: {np.array(t_tool).shape}")
    print(f"rotation_matrices 的大小是: {np.array(rotation_matrices).shape}")
    print(f"translation_vectors 的大小是: {np.array(translation_vectors).shape}")
    R, t = cv2.calibrateHandEye(R_tool, t_tool, rotation_matrices, translation_vectors, cv2.CALIB_HAND_EYE_PARK)
    print(f"旋转矩阵是：\n{R}\n")
    print(f"平移矩阵是：\n{t}\n")
    pose = np.eye(4)
    pose[:3,:3] = R
    pose[:3,3] = t.reshape(-1)
    np.savetxt('camera2gripper_matrix.txt', pose)
    # pose = [-0.10003807886213,	-0.399997280547224,	0.300050147152287,	2.16448537803642,	2.16934381981752,	0.0849566779283456]
    # H = pose_to_homogeneous_matrix(pose)
    # print(H)