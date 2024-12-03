import pybullet as p
import pybullet_data
import os
import time

# Đường dẫn đến dữ liệu dự án
PROJECT_DATA = "./ur_e_description/urdf"
ROBOT_URDF_PATH = os.path.join(PROJECT_DATA, "robot/ur5e.urdf")
TABLE_URDF_PATH = os.path.join(PROJECT_DATA, "table/table.urdf")
CUBE_URDF_PATH = os.path.join(PROJECT_DATA, "cube/red.urdf")
TRAY_URDF_PATH = [os.path.join(PROJECT_DATA, f"tray/tray_{color}.urdf") for color in ['red', 'green', 'blue']]

# Khởi tạo PyBullet GUI
physics_client = p.connect(p.GUI)

# Thêm đường dẫn mặc định (nếu cần)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Đặt trọng lực
p.setGravity(0, 0, -9.8)

# Tải mặt phẳng cơ bản
plane_id = p.loadURDF("plane.urdf")

# Tải mô hình robot UR5e
robot_start_position = [0, 0, 0.5]
robot_id = p.loadURDF(ROBOT_URDF_PATH, robot_start_position)

# Tải bàn
table_position = [1, 0, 0]
# table_id = p.loadURDF(TABLE_URDF_PATH, table_position)

# Tải khối lập phương
cube_position = [1, 0.5, 0.75]
cube_id = p.loadURDF(CUBE_URDF_PATH, cube_position)

# Tải các khay (mỗi khay một màu)
tray_positions = [[1.5, 0, 0.5], [1.5, 0.5, 0.5], [1.5, -0.5, 0.5]]
tray_ids = []
for tray_path, tray_position in zip(TRAY_URDF_PATH, tray_positions):
    tray_id = p.loadURDF(tray_path, tray_position)
    tray_ids.append(tray_id)

# Vòng lặp chính để giữ GUI mở
while True:
    p.stepSimulation()
    time.sleep(1. / 240.)

# Ngắt kết nối khi kết thúc (nếu cần thoát)
p.disconnect()
