import numpy as np
import math
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# 用來看兩腳距離跟身長比例
def pixel_distance(landmarker_list ,lm_1, lm_2, video_w, video_h):
    dx = (landmarker_list[lm_1].x - landmarker_list[lm_2].x) * video_w
    dy = (landmarker_list[lm_1].y - landmarker_list[lm_2].y) * video_h
    return math.sqrt(dx**2 + dy**2)

def angle_with_ground(landmarker_list, side = "right"):
    #計算向量跟地面有無平行
    #用來計算Max ER的
    if side == "right":
        wrist, elbow = np.array([landmarker_list[16].x, landmarker_list[16].y]), np.array([landmarker_list[14].x, landmarker_list[14].y])
    else:
        wrist, elbow = np.array([landmarker_list[15].x, landmarker_list[15].y]), np.array([landmarker_list[13].x, landmarker_list[13].y])

    # 1. 計算前臂向量 (V_forearm)
    v_forearm = wrist - elbow

    # 2. 定義垂直向下的向量
    v_vertical = np.array([0, -1])

    # 3. 計算點積 (Dot Product)
    dot_product = np.dot(v_forearm, v_vertical)

    # 4. 計算兩個向量的長度 (Magnitude / Norm)
    norm_forearm = np.linalg.norm(v_forearm)
    norm_vertical = np.linalg.norm(v_vertical) # 這個值其實就是 1

    # 5. 計算夾角的 cosine 值
    # 加上 np.clip 是為了避免浮點數誤差導致 arccos 計算失敗 (例如 1.000001)
    cos_theta = np.clip(dot_product / (norm_forearm * norm_vertical), -1.0, 1.0)

    # 6. 計算角度 (單位是「弧度」 radians)
    angle_rad = np.arccos(cos_theta)

    # 7. 將弧度轉換為「角度」 (degrees)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def draw_landmarks_on_image(rgb_image, detection_result):
    '''
    z 值越小（負越多），代表該點越靠近攝影機。
    z 值越大（正越多），代表該點離攝影機越遠。
    '''
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
    # 取得影像的寬度和高度
    image_height, image_width, _ = annotated_image.shape
    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        for i in range(11,33):
            if i in [17, 18, 19 ,20, 21, 22]:
                continue
            landmark = pose_landmarks[i] 
            
            # 1. 將正規化座標 (0.0-1.0) 轉換為像素座標
            pixel_x = int(landmark.x * image_width)
            pixel_y = int(landmark.y * image_height)

            # 2. 準備要顯示的文字 (Z 座標，取到小數點後2位)
            text = f"X:{landmark.x:.2f} Y:{landmark.y:.2f} Z:{landmark.z:.2f}"

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


# 計算水平外旋
def calculate_horizontal_abduction(landmarker_list, side='right'):
    # 抓方向跟抓出肩肘向量
    if side == "right":
        left_shoulder, right_shoulder, elbow = np.array([landmarker_list[11].x, landmarker_list[11].z]), np.array(
            [landmarker_list[12].x, landmarker_list[12].z]), np.array([landmarker_list[14].x, landmarker_list[14].z])
        v_shoulder = left_shoulder - right_shoulder
        v_arm = elbow - right_shoulder
    elif side == "left":
        left_shoulder, right_shoulder, elbow = np.array([landmarker_list[11].x, landmarker_list[11].z]), np.array(
            [landmarker_list[12].x, landmarker_list[12].z]), np.array([landmarker_list[13].x, landmarker_list[13].z])
        v_shoulder = left_shoulder - right_shoulder
        v_arm = elbow - left_shoulder
    else:
        return print(f"This is not a correct parameter {side}")

    # 使用 atan2 計算角度
    angle_shoulder = np.arctan2(v_shoulder[1], v_shoulder[0])
    angle_arm = np.arctan2(v_arm[1], v_arm[0])

    # 計算有向角度差 (範圍在 -pi 到 +pi)
    angle_diff_rad = angle_arm - angle_shoulder
    angle_diff_rad = (angle_diff_rad + np.pi) % (2 * np.pi) - np.pi

    angle_diff_deg = np.degrees(angle_diff_rad)

    # angle_diff_deg 在 -180 到 +180 之間
    # 180 (或 -180) = 側面 (0° 外展)
    # 90 = 後方 (90° 外展)
    # -90 = 前方 (90° 內收)
    # 0 = 胸前 (180° 內收)

    if angle_diff_deg > 0:  # 0 到 180 (胸前 -> 側面 -> 後方)
        # 這是 "後方" 區域
        final_angle_deg = 180 - angle_diff_deg
        # 180 -> 0
        # 90 -> 90
    else:  # -180 到 0 (側面 -> 前方 -> 胸前)
        # 這是 "前方" 區域
        final_angle_deg = -(180 + angle_diff_deg)
        # -180 -> 0
        # -90 -> -90

    return final_angle_deg


def calculate_body_angle(landmarker_list, start_idx, center_idx, end_idx):
    # setting points from landmarker_list
    start_point, center_point, end_point = np.array([landmarker_list[start_idx].x, landmarker_list[start_idx].y, landmarker_list[start_idx].z]), \
        np.array([landmarker_list[center_idx].x, landmarker_list[center_idx].y, landmarker_list[center_idx].z]), \
        np.array([landmarker_list[end_idx].x, landmarker_list[end_idx].y, landmarker_list[end_idx].z])

    '''
    theta = arccos(a*b/|a||b|)
    angle = theta * (180/pi)
    '''
    # calculate vector
    v_a = start_point - center_point
    v_b = end_point - center_point

    # calculate dot product
    dp = np.dot(v_a, v_b)

    # calculate Magnitude of vector
    mag = np.linalg.norm(v_a) * np.linalg.norm(v_b)
    cosine_angle = dp / mag
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    # calculte theta
    angle = np.degrees(np.arccos(cosine_angle))

    return angle


def calculate_body_angle_2d(landmarker_list, start_idx, center_idx, end_idx):
    # setting points from landmarker_list 2d
    start_point, center_point, end_point = np.array(
        [landmarker_list[start_idx].x, landmarker_list[start_idx].y]), np.array(
        [landmarker_list[center_idx].x, landmarker_list[center_idx].y]), np.array(
        [landmarker_list[end_idx].x, landmarker_list[end_idx].y])

    '''
    theta = arccos(a*b/|a||b|)
    angle = theta * (180/pi)
    '''
    # calculate vector
    v_a = start_point - center_point
    v_b = end_point - center_point

    # calculate dot product
    dp = np.dot(v_a, v_b)

    # calculate Magnitude of vector
    mag = np.linalg.norm(v_a) * np.linalg.norm(v_b)
    cosine_angle = dp / mag
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    # calculte theta
    angle = np.degrees(np.arccos(cosine_angle))

    return angle

