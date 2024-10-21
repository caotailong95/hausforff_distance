import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def hausdorff_distance_details(set_A, set_B):
    """    
        
                                  b
                              ——————————
                             |         |          b
                           c |         | a    ——————————
                             | NUAA#02 |     |         |
                             ——————————    c |         | a
                                  d          | NUAA#07 |
                                  b          ——————————
                              ——————————          d
                             |         |          b
                           c |         | a    ——————————
                             | NUAA#01 |   c | NUAA#06 | a
                             ——————————      ——————————
                                  d               d
    Y_axis  ↑
            ↑
            ↑
            · → → → 
          O        X_axis
    
    间隙计算：#01.b----#02.d
    计算两个点集之间的豪斯多夫距离，输出最大、最小距离，并返回每个点到另一个集合的最小距离。

    参数:
    set_A : np.ndarray
        点集A，形状为 (m, 3) 的numpy数组，表示m个三维点。
    set_B : np.ndarray
        点集B，形状为 (n, 3) 的numpy数组，表示n个三维点。

    返回:
    tuple
        最大豪斯多夫距离、最小豪斯多夫距离、对应点的坐标、每个点的最小距离。
    """
    # 计算所有点对之间的距离矩阵
    dist_matrix = distance.cdist(set_A, set_B)
    
    # 计算从A到B的最小距离
    min_dist_A_to_B = np.min(dist_matrix, axis=1)
    max_hausdorff_A_to_B = np.max(min_dist_A_to_B)
    max_index_A_to_B = np.argmax(min_dist_A_to_B)
    
    # 计算从B到A的最小距离
    min_dist_B_to_A = np.min(dist_matrix, axis=0)
    max_hausdorff_B_to_A = np.max(min_dist_B_to_A)
    max_index_B_to_A = np.argmax(min_dist_B_to_A)
    
    # 计算最大和最小豪斯多夫距离
    max_hausdorff_distance = max(max_hausdorff_A_to_B, max_hausdorff_B_to_A)
    min_hausdorff_distance = min(min_dist_A_to_B.min(), min_dist_B_to_A.min())
    
    # 找到最大距离对应的点对
    if max_hausdorff_A_to_B > max_hausdorff_B_to_A:
        max_point_A = set_A[max_index_A_to_B]
        min_index = np.argmin(dist_matrix[max_index_A_to_B])
        max_point_B = set_B[min_index]
    else:
        max_point_B = set_B[max_index_B_to_A]
        min_index = np.argmin(dist_matrix[:, max_index_B_to_A])
        max_point_A = set_A[min_index]
    
    # 找到最小距离对应的点对
    min_index_A = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
    min_point_A = set_A[min_index_A[0]]
    min_point_B = set_B[min_index_A[1]]
    
    # 获取每个点的最小距离和对应的点
    closest_points_A_to_B = [set_B[np.argmin(row)] for row in dist_matrix]
    closest_points_B_to_A = [set_A[np.argmin(col)] for col in dist_matrix.T]

    return max_hausdorff_distance, min_hausdorff_distance, (max_point_A, max_point_B), (min_point_A, min_point_B), min_dist_A_to_B, min_dist_B_to_A, closest_points_A_to_B, closest_points_B_to_A

# 从文件中读取点集
def read_points_from_file(file_path):
    return np.loadtxt(file_path)

# 将距离矩阵和最大最小距离保存到文件
def save_distances_to_file(filename, max_distance, min_distance, max_points, min_points, min_dist_A_to_B, min_dist_B_to_A, closest_points_A_to_B, closest_points_B_to_A, set_A, set_B, monitoring_points):
    with open(filename, 'w') as f:         
        # 写入9个监听点及其距离
        f.write("监听点及其距离:\n")
        for point in monitoring_points:
            f.write(f"监听点 = {point[0]}, 距离 = {point[1]}\n")
                
        # 写入最大距离和对应的点对坐标
        f.write(f"最大距离: {max_distance}, 点A: {max_points[0]}, 点B: {max_points[1]}\n")
        # 写入最小距离和对应的点对坐标
        f.write(f"最小距离: {min_distance}, 点A: {min_points[0]}, 点B: {min_points[1]}\n")
        
        # 写入每个点的豪斯多夫距离及对应的点坐标
        f.write("点集A到点集B的最小距离:\n")
        for i, (dist, point_B) in enumerate(zip(min_dist_A_to_B, closest_points_A_to_B)):
            f.write(f"点A[{i}] = {set_A[i]}, 最小距离 = {dist}, 对应点B = {point_B}\n")
        
        f.write("点集B到点集A的最小距离:\n")
        for i, (dist, point_A) in enumerate(zip(min_dist_B_to_A, closest_points_B_to_A)):
            f.write(f"点B[{i}] = {set_B[i]}, 最小距离 = {dist}, 对应点A = {point_A}\n")
                   
    print(f"距离数据已保存到文件: {filename}")

# 计算9个监听点
def calculate_monitoring_points(points_A, points_B):
    """
    根据XZ坐标划分计算9个监听点的位置。
    """
    # 获取X和Z坐标的最大最小值
    x_min, x_max = points_A[:, 0].min(), points_A[:, 0].max()
    z_min, z_max = points_A[:, 2].min(), points_A[:, 2].max()
    
    # 使用linspace均匀分割XZ轴，获取中间三个值
    x_coords = np.linspace(x_min, x_max, 5)[1:-1]  # 获取中间三个x值
    z_coords = np.linspace(z_min, z_max, 5)[1:-1]  # 获取中间三个z值
    
    print("x_coords:", x_coords)
    print("z_coords:", z_coords)
    
    # 记录9个监听点及其最小豪斯多夫距离
    monitoring_points = []
    
    for x in x_coords:
        for z in z_coords:
            # 在点集A中找到最近的点，并获取其完整坐标
            distances = np.sqrt((points_A[:, 0] - x) ** 2 + (points_A[:, 2] - z) ** 2)
            idx_A = np.argmin(distances)
            nearest_point_A = points_A[idx_A]
            
            # 计算该点到点集B的最小豪斯多夫距离
            dist_to_B = np.sqrt(np.sum((points_B - nearest_point_A) ** 2, axis=1))
            min_distance_to_B = np.min(dist_to_B)
            
            # 记录监听点和距离
            monitoring_points.append((nearest_point_A, min_distance_to_B))
            
    # 输出监听点的坐标和对应的最小豪斯多夫距离
    for point, distance in monitoring_points:
        print(f"监听点坐标: {point}, 最小哈斯多夫距离: {distance}")
    
    return sorted(monitoring_points, key=lambda p: p[1])
   
# 文件路径
file_A = '01b.txt'
file_B = '02d.txt'

# 读取点集A和B
points_A = read_points_from_file(file_A)
points_B = read_points_from_file(file_B)

# 计算豪斯多夫距离并输出每个点的最小距离
max_d_H, min_d_H, max_points, min_points, min_dist_A_to_B, min_dist_B_to_A, closest_points_A_to_B, closest_points_B_to_A = hausdorff_distance_details(points_A, points_B)

# 计算9个监听点
monitoring_points = calculate_monitoring_points(points_A, points_B)

print(f"点集A和点集B之间的最大豪斯多夫距离是: {max_d_H}")
print(f"点集A和点集B之间的最小豪斯多夫距离是: {min_d_H}")

# 保存结果到文件
save_distances_to_file("hausdorff_distance_gap.txt", max_d_H, min_d_H, max_points, min_points, min_dist_A_to_B, min_dist_B_to_A, closest_points_A_to_B, closest_points_B_to_A, points_A, points_B, monitoring_points)

def visualize_points(set_A, set_B, monitoring_points):
    fig = plt.figure(figsize=(10, 8))  # 增加图像大小
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制点集A和点集B
    ax.scatter(set_A[:, 0], set_A[:, 1], set_A[:, 2], c='b', marker='o', label='Set A', alpha=0.6)
    ax.scatter(set_B[:, 0], set_B[:, 1], set_B[:, 2], c='r', marker='^', label='Set B', alpha=0.6)
    
    # 绘制监听点
    for point, _ in monitoring_points:
        ax.scatter(point[0], point[1], point[2], c='k', marker='x', s=100, label='Monitoring Points', alpha=1)
    
    # 删除重复的图例条目
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = set(labels)
    unique_handles = [handles[labels.index(label)] for label in unique_labels]
    ax.legend(unique_handles, unique_labels, loc='upper right')  # 将图例放置在右上角
    
    # 设置坐标轴标签
    # 调整坐标轴的范围，以同时考虑点集A和点集B
    x_min, x_max = min(set_A[:, 0].min(), set_B[:, 0].min()), max(set_A[:, 0].max(), set_B[:, 0].max())
    y_min, y_max = min(set_A[:, 1].min(), set_B[:, 1].min()), max(set_A[:, 1].max(), set_B[:, 1].max())
    z_min, z_max = min(set_A[:, 2].min(), set_B[:, 2].min()), max(set_A[:, 2].max(), set_B[:, 2].max())
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    
    # 设置适合的视角
    # ax.view_init(elev=20, azim=30)  # 设定一个合适的视角来观察图形
    
    # 显示网格
    ax.grid(True)
    
    # 设置轴标签
    ax.set_xlabel('X_Axis')
    ax.set_ylabel('Y_Axis')
    ax.set_zlabel('Z_Axis')
    
    # 显示图形
    plt.show()

# 调用可视化函数
visualize_points(points_A, points_B, monitoring_points)

