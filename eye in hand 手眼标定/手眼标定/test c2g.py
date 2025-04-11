import numpy as np
import os

def transform_point(point, transform_matrix):
    """
    将点通过变换矩阵进行坐标转换
    :param point: 输入的点，形状为 (3,) 或 (4,)
    :param transform_matrix: 变换矩阵，形状为 (4, 4)
    :return: 转换后的点，形状为 (3,)
    """
    if len(point) == 3:
        point = np.append(point, 1)
    transformed_point = np.dot(transform_matrix, point)
    return transformed_point[:3]


def test_hand_eye_calibration(hand_eye_matrix, R_gripper2base_list, t_gripper2base_list, R_target2cam_list,
                              t_target2cam_list):
    """
    测试手眼标定矩阵的准确性
    :param hand_eye_matrix: 手眼标定矩阵，形状为 (4, 4)
    :param R_gripper2base_list: 机器人末端执行器相对于基座的旋转矩阵列表
    :param t_gripper2base_list: 机器人末端执行器相对于基座的平移向量列表
    :param R_target2cam_list: 标定目标相对于相机的旋转矩阵列表
    :param t_target2cam_list: 标定目标相对于相机的平移向量列表
    :return: 平均位置误差
    """
    errors = []
    num_poses = len(R_gripper2base_list)

    for i in range(num_poses):
        R_gripper2base = R_gripper2base_list[i]
        t_gripper2base = t_gripper2base_list[i]
        R_target2cam = R_target2cam_list[i]
        t_target2cam = t_target2cam_list[i]

        # 构建机器人末端执行器到基座的变换矩阵
        T_gripper2base = np.eye(4)
        T_gripper2base[:3, :3] = R_gripper2base
        T_gripper2base[:3, 3] = t_gripper2base.flatten()

        # 构建标定目标到相机的变换矩阵
        T_target2cam = np.eye(4)
        T_target2cam[:3, :3] = R_target2cam
        T_target2cam[:3, 3] = t_target2cam.flatten()

        # 假设目标点在标定目标坐标系下的位置
        target_point_in_target = np.array([0, 0, 0])

        # 计算目标点在相机坐标系下的位置
        target_point_in_cam = transform_point(target_point_in_target, T_target2cam)

        # 计算目标点在机器人末端执行器坐标系下的位置
        target_point_in_ee = transform_point(target_point_in_cam, hand_eye_matrix)

        # 计算目标点在机器人基座坐标系下的位置
        target_point_in_base = transform_point(target_point_in_ee, T_gripper2base)

        # 这里假设已知目标点在机器人基座坐标系下的真实位置
        # 在实际应用中，需要通过其他可靠方式获取真实位置
        true_target_point_in_base = np.array([0, 0, 0])

        # 计算位置误差
        error = np.linalg.norm(target_point_in_base - true_target_point_in_base)
        errors.append(error)

    # 计算平均位置误差
    average_error = np.mean(errors)
    return average_error


# 示例数据
# 假设已经完成手眼标定，得到手眼矩阵
# 读取矩阵数据
pose_loaded = np.loadtxt('camera2gripper_matrix.txt')

print(f"从文件中加载的旋转和平移矩阵是：\n{pose_loaded}\n")

# 如果你需要分别提取旋转矩阵和平移向量，可以这样做：
R_loaded = pose_loaded[:3, :3]
t_loaded = pose_loaded[:3, 3]

print(f"加载的旋转矩阵是：\n{R_loaded}\n")
print(f"加载的平移向量是：\n{t_loaded}\n")



# 机器人末端执行器相对于基座的旋转矩阵列表,平移向量列表
path = os.path.dirname(__file__)
csv_file = os.path.join(path,"RobotToolPose.csv")
tool_pose = np.loadtxt(csv_file,delimiter=',')
R_tool = []
t_tool = []
N = 9                                                    #.XLSX文件中数据的行数
for i in range(int(N)):

    R_tool.append(tool_pose[0:3,4*i:4*i+3])
    t_tool.append(tool_pose[0:3,4*i+3].reshape(-1, 1))




# 标定目标相对于相机的旋转矩阵列表
csv_file = os.path.join(path,"cameraPose.csv")
tool_pose = np.loadtxt(csv_file,delimiter=',')
R_cam = []
for i in range(int(N)):
    R_cam.append(tool_pose[0:3,3*i:3*i+3])


# 标定目标相对于相机的平移向量列表
csv_file = os.path.join(path,"translationcameraPose.csv")
tool_pose = np.loadtxt(csv_file,delimiter=',')
t_cam = []
for i in range(int(N)):
    t_cam.append(tool_pose[0:3,i])

# 测试手眼标定矩阵
average_error = test_hand_eye_calibration(pose_loaded, R_tool, t_tool, R_cam,
                                          t_cam)
print(f"平均位置误差: {average_error}")
    