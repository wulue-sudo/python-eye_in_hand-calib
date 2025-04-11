## 1.运行get_data_new.py，获取标定需要的图片和位姿数据
注意从UR5机械臂的示教器中读取一个，处在正常位置时的xyz和rx,ry,rz，然后以这个数据为参考，将下面的测试范围值设置为一个相对合适的范围。如果设置不合理，会使机械臂移动到一些很奇怪的姿势。               

```python
tcp_host_ip = '192.168.3.10'  # IP和端口，机器人控制器额的IP地址
workspace_limits = np.asarray([[-0.05, 0], [-0.35, -0.30], [0.35, 0.40]])        #定义了机器人工作空间的范围，是一个3x2的NumPy数组，表示x、y、z三个方向上的最小值和最大值。
calib_grid_step = 0.02                                                           #标定网格的步长
tool_orientation = [2 * np.pi / 3, 2 * np.pi / 3, 0]
```

添加的旋转扰动要稍微大一些，大概在接近30度的值，否则求不出手眼矩阵，报错如下：

<font style="color:#DF2A3F;">[ERROR:0@1.733] global calibration_handeye.cpp:335 calibrateHandEyeTsai Hand-eye calibration failed! Not enough informative motions--include larger rotations.</font>

```python
# 添加小的随机旋转扰动（单位：弧度）
    random_rotation = np.random.uniform(-0.5, 0.5, 3)
```

## 2.运行xyz_npy2excel.py，将每一个保存的位姿数据以.npy结尾的形式转化为.xlsx
## 3.运行test.py，计算手眼矩阵
![](https://cdn.nlark.com/yuque/0/2025/png/49878130/1742536423215-7ad177c0-00bc-41e7-979e-7fec27c34537.png)

有些拍摄的标定板图片不完整，因此提取不出完整角点，也就不会得到<font style="color:#DF2A3F;">目标相对于相机</font>的位姿，因此我们要在.xlsx文件里，删除那些没有提取出完整角点的图片对应的位姿。

我们从这里可以看到那些图片是不可用的，比如12.png，然后就去.xlsx里面删除第13行（图片命名以0开始，.xlsx以1开始）

![](https://cdn.nlark.com/yuque/0/2025/png/49878130/1742536587843-eb155781-7d01-4062-9865-f5234c9fe53e.png)

最后要保证相应矩阵的维度保持一致。

![](https://cdn.nlark.com/yuque/0/2025/png/49878130/1742536726269-4278a4af-29aa-4948-8392-ee7a640c26a8.png)

最后使用cv2.calibrateHandEye(R_tool, t_tool, rotation_matrices, translation_vectors, cv2.CALIB_HAND_EYE_PARK)计算出手眼矩阵的旋转和平移


