import numpy as np

def calculate_monitoring_points(points_A, points_B):
    x_min, x_max = points_A[:, 0].min(), points_A[:, 0].max()
    y_min, y_max = points_A[:, 1].min(), points_A[:, 1].max()
    
    # 增加网格点的数量以生成更多监测点
    num_points = 6  # 增加到6个网格点
    x_coords = np.linspace(x_min, x_max, num_points)
    y_coords = np.linspace(y_min, y_max, num_points)
    
    print("x_coords:", x_coords)
    print("y_coords:", y_coords)
    
    monitoring_points = []
    
    for x in x_coords:
        for y in y_coords:
            distances = np.sqrt((points_A[:, 0] - x) ** 2 + (points_A[:, 1] - y) ** 2)
            idx_A = np.argmin(distances)
            nearest_point_A = points_A[idx_A]
            dist_to_B = np.sqrt(np.sum((points_B - nearest_point_A) ** 2, axis=1))
            min_distance_to_B = np.min(dist_to_B)
            monitoring_points.append((nearest_point_A, min_distance_to_B))
    
    # 打印每个计算的监测点
    print("All monitoring points before deduplication:")
    for i, point in enumerate(monitoring_points):
        print(f"Monitoring point {i + 1}: {point[0]}, Distance: {point[1]}")
    
    # 移除重复的监测点
    unique_monitoring_points = []
    seen_points = set()
    for point in monitoring_points:
        point_str = str(point[0]) + str(point[1])
        if point_str not in seen_points:
            seen_points.add(point_str)
            unique_monitoring_points.append(point)
    
    # 打印唯一的监测点信息
    print("Total unique monitoring points calculated:", len(unique_monitoring_points))
    for i, point in enumerate(unique_monitoring_points):
        print(f"Unique Monitoring point {i + 1}: {point[0]}, Distance: {point[1]}")
    
    return unique_monitoring_points

# 示例代码来测试
if __name__ == "__main__":
    # 示例数据
    points_A = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ])
    points_B = np.array([
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4]
    ])
    
    monitoring_points = calculate_monitoring_points(points_A, points_B)
    print("Calculated monitoring points:", monitoring_points)
