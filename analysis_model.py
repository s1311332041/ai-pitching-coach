import numpy as np
import math
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python
import cv2
import json
import os
import time

# GCS 下載影片
import requests
import tempfile 

# Gemini API 套件
from google import genai
from google.genai import types


def pixel_distance(landmarker_list ,lm_1, lm_2, video_w, video_h):
    dx = (landmarker_list[lm_1].x - landmarker_list[lm_2].x) * video_w
    dy = (landmarker_list[lm_1].y - landmarker_list[lm_2].y) * video_h
    return math.sqrt(dx**2 + dy**2)

def angle_with_ground(landmarker_list, side = "right"):
    if side == "right":
        wrist, elbow = np.array([landmarker_list[16].x, landmarker_list[16].y]), np.array([landmarker_list[14].x, landmarker_list[14].y])
    else:
        wrist, elbow = np.array([landmarker_list[15].x, landmarker_list[15].y]), np.array([landmarker_list[13].x, landmarker_list[13].y])
    v_forearm = wrist - elbow
    v_vertical = np.array([0, -1])
    dot_product = np.dot(v_forearm, v_vertical)
    norm_forearm = np.linalg.norm(v_forearm)
    norm_vertical = np.linalg.norm(v_vertical)
    cos_theta = np.clip(dot_product / (norm_forearm * norm_vertical), -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
    image_height, image_width, _ = annotated_image.shape
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        for i in range(11,33):
            if i in [17, 18, 19 ,20, 21, 22]:
                continue
            landmark = pose_landmarks[i] 
            pixel_x = int(landmark.x * image_width)
            pixel_y = int(landmark.y * image_height)
            text = f"X:{landmark.x:.2f} Y:{landmark.y:.2f} Z:{landmark.z:.2f}"
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

def calculate_horizontal_abduction(landmarker_list, side='right'):
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
    angle_shoulder = np.arctan2(v_shoulder[1], v_shoulder[0])
    angle_arm = np.arctan2(v_arm[1], v_arm[0])
    angle_diff_rad = angle_arm - angle_shoulder
    angle_diff_rad = (angle_diff_rad + np.pi) % (2 * np.pi) - np.pi
    angle_diff_deg = np.degrees(angle_diff_rad)
    if angle_diff_deg > 0: 
        final_angle_deg = 180 - angle_diff_deg
    else: 
        final_angle_deg = -(180 + angle_diff_deg)
    return final_angle_deg

def calculate_body_angle(landmarker_list, start_idx, center_idx, end_idx):
    start_point, center_point, end_point = np.array([landmarker_list[start_idx].x, landmarker_list[start_idx].y, landmarker_list[start_idx].z]), \
        np.array([landmarker_list[center_idx].x, landmarker_list[center_idx].y, landmarker_list[center_idx].z]), \
        np.array([landmarker_list[end_idx].x, landmarker_list[end_idx].y, landmarker_list[end_idx].z])
    v_a = start_point - center_point
    v_b = end_point - center_point
    dp = np.dot(v_a, v_b)
    mag = np.linalg.norm(v_a) * np.linalg.norm(v_b)
    cosine_angle = dp / mag
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return angle


def format_timestamp(frame_number, fps):
    if fps <= 0: return "00:00"
    seconds = frame_number / fps
    m, s = divmod(seconds, 60)
    # 回傳 "00:05.33" 格式 (分:秒.毫秒)
    return f"{int(m):02d}:{s:05.2f}"

def convert2Json(peak_leg_tuple, foot_plant_tuple, maxER_tuple, ball_release_tuple,
           left_elbow_angle,right_elbow_angle,left_knee_angle,right_knee_angle,
           right_shoulder_abduction,left_shoulder_abduction,horizontal_abduction, 
           fps): 
    
    peak_leg = {
        "Frame": peak_leg_tuple[0],
        "Time": format_timestamp(peak_leg_tuple[0], fps)       
    }
    foot_plant = {
        "Frame": foot_plant_tuple[0],
        "Time": format_timestamp(foot_plant_tuple[0], fps),    
        "stride_percentage": foot_plant_tuple[1]
    }
    max_ER = {
        "Frame": maxER_tuple[0],
        "Time": format_timestamp(maxER_tuple[0], fps),         
        "ER(External Rotation)angle": maxER_tuple[1]
    }
    ball_release = {
        "Frame": ball_release_tuple[0],
        "Time": format_timestamp(ball_release_tuple[0], fps)   
    }

    keyframe = {
        "peak leg frame": peak_leg,
        "foot plant frame": foot_plant,
        "max ER frame": max_ER,
        "ball release frame": ball_release
    }
    frame_landmark = {
        "left_elbow_angle": left_elbow_angle,
        "right_elbow_angle": right_elbow_angle,
        "left_knee_angle": left_knee_angle,
        "right_knee_angle": right_knee_angle,
        "right_shoulder_abduction": right_shoulder_abduction,
        "left_shoulder_abduction": left_shoulder_abduction,
        "horizontal_abduction": horizontal_abduction,
        "keyframe": keyframe,
    }
    
    # 移除 local file write
    # output_json_path = "json/Pitcher_pose_data.json"
    # with open(output_json_path, 'w', encoding='utf-8') as f:
    #     json.dump(frame_landmark, f, indent=4)
    # print(f"成功將姿勢數據儲存至: {output_json_path}")
    
    # 回傳 Python 字典
    return frame_landmark


# 將模型路徑改為「相對路徑」
model_full_path = 'models/pose_landmarker_full.task'
model_heavy_path = 'models/pose_landmarker_heavy.task'
model_lite_path = "models/pose_landmarker_lite.task"

# 將「規則」移到函式外部，作為全域常數
RULES_PROMPT = '''
注意!!!偵測出來可能會有正負10度的誤差!!!
================================================================================================================
第一時期：預備期 (Wind-up)
目標：建立穩定平衡的起始動作。
規則 1.1 - [重心穩定]：
    IF 身體重心（骨盆中心點）在抬腿過程中，水平位移過大，THEN 提示「起始平衡不佳，晃動過多可能導致力量流失」。
規則 1.2 - [軸心腳支撐角度]：
    IF 軸心腳（ex:左投是左腳，反之）的膝蓋角度大於 160度（過於伸直）或小於 120度（過於彎曲），THEN 提示「軸心腳膝蓋角度不佳，不利於後續力量的儲存與爆發」。
選擇依據：
規則 1.1：
    選擇監測重心，是因為「平衡是動力鏈的起點」。
    如果在第一步就失去平衡，後續所有階段的發力都會被迫用來「修正」而非「加速」，導致力量傳遞中斷。
規則 1.2：
    選擇檢查軸心腳膝蓋，是因為此為發力的「起點」。膝蓋過度伸直(大於170度)會使腿部肌肉僵硬，無法產生彈性與爆發力；
    過度彎曲(小於110度)則會讓重心過低、姿勢不穩。此規則確保投手處於一個「隨時可發力」的運動準備狀態。
    而數據(150度~120度)是因為根據論文所說軸心腳微彎，加上數據端的誤差所設定的。
================================================================================================================
第二時期：跨步期(Stride)
目標：線性地將力量引導至本壘板。
規則 2.1 - [跨步距離]：
    IF前腳落地時，雙腳踝之間為身高的75%(誤差為正負5%)，THEN 提示「跨步距離可能過短/過長，影響動力鏈的順暢度」。
規則 2.2 - [落地腳穩定性]：
    IF前腳落地瞬間，膝蓋角度小於150度(過度彎曲)或大於 120度(接近伸直），THEN提示「前腳落地時膝蓋角度過於伸直或是彎曲導致不穩，可能造成力量中斷或增加膝關節壓力」。
規則 2.3 - [投球臂肩外展]：  
    IF肩外展角度小於80度大於100度，THEN提示「肩需要夾一點或開一點」。
選擇依據:
規則 2.1:
    跨步長度之所以需要達到這個特定的比例，是為了確保投球動作中關鍵的生物力學、時間協調和能量傳遞能達到最佳狀態，同時降低受傷風險。正確的跨步長度也對於建立穩定的投球基礎很重要。
規則 2.2:
    角落地時，前膝蓋彎曲來吸收著地時的衝擊力，以保護膝蓋免受傷害。也能帶來一定的重心穩地度。
規則 2.3:
    為了確保投手有正確手部擺放位置，以防止投手手部負擔。
    假如肩膀水平外展過多，可能導致肩關節囊韌帶撕裂。
================================================================================================================
第三時期：上臂舉球期 (Arm Cocking)
目標：最大化身體的扭轉與彈性能量儲存（球速的關鍵）。
規則 3.1 - [手臂位置]：
    IF 投球手臂的肩膀外展角度遠小於 85度 或大於 110度，THEN提示「手臂與身體的夾角不當，可能導致『手臂拖曳 (Arm Drag)』，增加肩膀受傷風險」。
規則 3.2 - [肩外旋角度]：
    IF 投球側的肩膀最大外旋(Max ER)角度明顯不足（小於 160度），THEN提示「手臂向後伸展幅度不足，影響力量的完全儲存」。
選擇依據:
規則 3.1:
    根據論文研究顯示肩外展在90度會有最大的活動範圍
規則 3.2:
    想像手臂為橡皮筋，被拉升到最長，並釋放的過程，假如外旋不足會導致肩膀內旋速度低，進而導致球速低。
================================================================================================================
第四時期：上臂加速期 (Arm Acceleration)
目標：將儲存的能量依序、高效地釋放。
規則 4.1 - [前腳支撐]：
    IF在球離手瞬間，前腳膝蓋角度仍小於140度（彎曲過多），THEN提示「前腳支撐腿過軟，未能形成穩固的支點，造成力量流失」。
規則 4.2 - [出手時的手臂角度]：
    IF在球離手瞬間，手肘彎曲角度小於135度，THEN提示「出手時手臂伸展不完全，力量未完全釋放，動作更像是『推球』而非『甩鞭』」。
規則 4.3 - [身體前傾]：
    IF在球離手瞬間，身體軀幹前傾角度不足（小於 35度），THEN提示「出手時身體跟進不足，未能有效利用全身的體重去加速」。
選擇依據:
規則 4.1:
    前導膝的「鎖定」作用，它作為一個穩固的支點，將下半身的動能傳遞到軀幹和手臂。
規則 4.2:
    要讓手肘並非完全伸直，仍保留輕微彎曲。還會保留一點彈性，避免手肘伸直而鎖死受傷。
規則 4.3:
    軀幹在釋放球時的精確傾斜角度，這有助於最大化能量傳遞和穩定性。
================================================================================================================
第五時期 / 第六時期：減速期 (Deceleration / Fallow Through)
目標：安全地吸收巨大的手臂動能，避免受傷。
規則 5.1 - [手臂順勢動作]：
    IF 手臂在球離手後，沒有順勢劃過身體到對側膝蓋附近，而是過早停止，THEN 提示「減速不完全！這會讓肩膀與手肘承受巨大的衝擊力，是受傷的高風險動作」。
規則 5.2 - [身體協助減速]：
    IF 在此階段，身體軀幹的最大前傾角度不足（小於 40度），THEN 提示「身體沒有充分前傾來幫助吸收手臂的減速力量，壓力過度集中在手臂上」。
'''

def get_gemini_report_from_video(video_gcs_url, side, gemini_api_key):
    """
    這是 app.py 會呼叫的「主要函式」。
    它負責串聯整個 AI 流程：
    1. 從 GCS 下載影片到暫存檔
    2. 執行 MediaPipe 分析
    3. 產生原始 JSON (in memory)
    4. 上傳影片到 Gemini
    5. 呼叫 Gemini API (含影片 + JSON)
    6. 回傳 Gemini 生成的 Markdown 文字
    """
    
    local_temp_video_path = None
    try:
        # -------------------------------------------------
        # 步驟 1: 從 GCS URL 下載影片到暫存檔
        # -------------------------------------------------
        print(f"[AI 流程]：開始從 GCS 下載 {video_gcs_url}")
        # 建立一個暫存檔案 (影片)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            local_temp_video_path = temp_file.name
            # 使用 requests 下載
            with requests.get(video_gcs_url, stream=True) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=8192): 
                    temp_file.write(chunk)
        
        print(f"[AI 流程]：影片已下載到暫存路徑: {local_temp_video_path}")
        
        # -------------------------------------------------
        # 步驟 2: 執行 Gemini Client 和 MediaPipe
        # -------------------------------------------------
        
        # 移除 input()，使用函式參數
        video = local_temp_video_path
        # side = input("請輸入是左投還是右投(left or right) : ") # -> 已由參數 'side' 傳入

        # 使用傳入的 API Key
        client = genai.Client(api_key=gemini_api_key)
        chat = client.chats.create(model="gemini-3-pro-preview", config=types.GenerateContentConfig(
            system_instruction = f"""
            你是一位 MLB 頂級的投手運動科學專家。請根據提供的影片和 JSON 數據進行分析。

            【重要：輸出格式要求】
            1. 請務必使用 **Markdown** 格式輸出，以便在網頁上漂亮地顯示。
            2. 請使用 **H2 (##)** 標題來區分每個時期 (例如：## 第一時期：預備期)。
            3. 請使用 **H3 (###)** 標題來區分每個規則 (例如：### 1.1 抬腿高度)。
            4. 在提到關鍵動作時，請務必標註 **時間點** (例如：**在 00:02.45 時**)，而不只是幀數。數據中已有 "Time" 欄位。
            5. 使用 **列表 (-)** 和 **粗體 (**...**)** 來強調重點。
            6. 對於每個規則，請包含：
               - **標準：** ...
               - **你的表現：** (附上時間點) ...
               - **評分：** (合格/不合格，請用粗體)
               - **專家建議：** ...

            以下是詳細的分析規則：
            {RULES_PROMPT}
            """)
            )
        
        print("[AI 流程]：開始上傳影片到 Gemini...")
        gvideo = client.files.upload(file = video)
        
        print(f"Gemini Video is {gvideo.state.name}")
        while gvideo.state.name == "PROCESSING":
            print("still processing...")
            time.sleep(5)
            gvideo = client.files.get(name=gvideo.name)
        print(f"[AI 流程]：Gemini 影片上傳完成 {gvideo.state.name}")

        # 影片 mediapipe設定
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_full_path),
            running_mode=VisionRunningMode.VIDEO,
            min_pose_detection_confidence=0.6,
            min_pose_presence_confidence=0.6,
            min_tracking_confidence=0.6)

        # -------------------------------------------------
        # 步驟 3: MediaPipe 處理迴圈
        # -------------------------------------------------
        print(f"[AI 流程]：開始 MediaPipe 逐幀分析...")
        timestamp_ms = 0
        with PoseLandmarker.create_from_options(options) as landmarker1:
            cap1 = cv2.VideoCapture(video_gcs_url)
            fps = cap1.get(cv2.CAP_PROP_FPS)

            frame_index = 0
            temp_frame = []
            temp_angle = []
            knee_heights = []
            ankle_distances = []
            wrist_vs_head = []
            left_elbow_angle = []
            right_elbow_angle = []
            left_knee_angle = []
            right_knee_angle = []
            left_shoulder_abduction = []
            right_shoulder_abduction = []
            horizontal_abduction = []

            while cap1.isOpened():
                success1, frame1 = cap1.read()
                if not success1:
                    break
                frame_index += 1

                mp_image1 = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
                timestamp_ms = int(frame_index * 1000 / fps)
                
                pose_landmarker_result1 = landmarker1.detect_for_video(mp_image1, timestamp_ms)
                
                # 處理 MediaPipe 找不到姿勢的狀況
                if not pose_landmarker_result1.pose_landmarks:
                    print(f"Frame {frame_index}: No pose detected, skipping.")
                    continue
                
                landmarker_list1 = pose_landmarker_result1.pose_landmarks[0]

                if frame_index == 1:
                    h, w, _ = frame1.shape
                    body_h = pixel_distance(landmarker_list1, lm_1=0, lm_2=30, video_h=h, video_w=w)
                
                if side == "right":
                    knee_y = landmarker_list1[25].y
                    hip_y = landmarker_list1[23].y
                    knee_heights.append((frame_index, hip_y - knee_y))
                    wrist = landmarker_list1[16]
                    head = landmarker_list1[0]
                    if wrist.x >= head.x:
                        wrist_vs_head.append((frame_index, wrist.y - head.y))
                    if landmarker_list1[11].y > landmarker_list1[12].y and landmarker_list1[14].y < landmarker_list1[12].y and \
                            landmarker_list1[16].y < landmarker_list1[14].y:
                        temp_frame.append(frame_index)
                        temp_angle.append(angle_with_ground(landmarker_list1, side))
                else:
                    knee_y = landmarker_list1[26].y
                    hip_y = landmarker_list1[24].y
                    knee_heights.append((frame_index, hip_y - knee_y))
                    wrist = landmarker_list1[15]
                    head = landmarker_list1[0]
                    if wrist.x <= head.x:
                        wrist_vs_head.append((frame_index, wrist.y - head.y))
                    if landmarker_list1[11].y < landmarker_list1[12].y and landmarker_list1[13].y < landmarker_list1[12].y and \
                            landmarker_list1[15].y < landmarker_list1[13].y:
                        temp_frame.append(frame_index)
                        temp_angle.append(angle_with_ground(landmarker_list1, side))

                foot_dist = pixel_distance(landmarker_list1, lm_1=27, lm_2=28, video_h=h, video_w=w)
                ratio = (foot_dist / (body_h + 155)) * 100
                ankle_distances.append((frame_index, ratio))
                
                left_elbow_angle.append({"frame": frame_index, "angle": calculate_body_angle(landmarker_list1, 15, 13, 11)})
                right_elbow_angle.append({"frame": frame_index, "angle": calculate_body_angle(landmarker_list1, 16, 14, 12)})
                left_knee_angle.append({"frame": frame_index, "angle": calculate_body_angle(landmarker_list1, 23, 25, 27)})
                right_knee_angle.append({"frame": frame_index, "angle": calculate_body_angle(landmarker_list1, 24, 26, 28)})
                horizontal_abduction.append({"frame": frame_index, "angle": calculate_horizontal_abduction(landmarker_list1, side)})
                right_shoulder_abduction.append({"frame": frame_index, "angle": calculate_body_angle(landmarker_list1, 14, 12, 24)})
                left_shoulder_abduction.append({"frame": frame_index, "angle": calculate_body_angle(landmarker_list1, 13, 11, 23)})


            print(f"[AI 流程]：MediaPipe 逐幀分析完成。")
            print("\n" + "="*30)
            print("🤖 [AI 流程]：開始除錯 (Debug) 關鍵幀列表...")
            print(f"DEBUG: 'temp_frame' (Max ER 候選): {temp_frame}")
            print(f"DEBUG: 'temp_angle' (Max ER 候選角度): {temp_angle}")
            print(f"DEBUG: 'wrist_vs_head' (Ball Release 候選): {wrist_vs_head}")
            print(f"DEBUG: 'knee_heights' (Peak Leg 候選): {knee_heights}")
            print(f"DEBUG: 'ankle_distances' (Foot Plant 候選): {ankle_distances}")
            print("="*30 + "\n")
            # -------------------------------------------------
            # 步驟 4: 關鍵幀提取 (來自您的 main.py)
            # -------------------------------------------------

            # 檢查 Ball release
            if not wrist_vs_head:
                raise ValueError("MediaPipe 處理失敗：'wrist_vs_head' 列表為空，無法偵測到出手點。影片可能太短或無法辨識。")
            ball_release_tuple = min(wrist_vs_head, key=lambda x: x[1])
            print(f"關鍵幀 Ball release: {ball_release_tuple[0]}")

            # 檢查 Peak Leg
            search_window_pl = [x for x in knee_heights if x[0] < ball_release_tuple[0]]
            if not search_window_pl:
                raise ValueError("MediaPipe 處理失敗：'search_window_pl' 列表為空，無法偵測到抬腿高峰。")
            peak_leg_tuple = max(search_window_pl, key=lambda x: x[1])
            print(f"關鍵幀 Peak Leg: {peak_leg_tuple[0]}")

            # 檢查 Foot plant
            search_window_fp = [x for x in ankle_distances if x[0] > peak_leg_tuple[0] and x[0] < ball_release_tuple[0]]
            if not search_window_fp:
                raise ValueError("MediaPipe 處理失敗：'search_window_fp' 列表為空，無法偵測到跨步落地。")
            foot_plant_tuple = max(search_window_fp, key=lambda x: x[1])
            print(f"關鍵幀 Foot plant: {foot_plant_tuple[0]}")

            # 檢查 Max ER
            frame_stride2BallRelease = [i for i in temp_frame if foot_plant_tuple[0]<i<ball_release_tuple[0]]
            if not frame_stride2BallRelease:
                 raise ValueError("MediaPipe 處理失敗：'frame_stride2BallRelease' 列表為空，無法偵"
                                  "測到最大外旋區間。")
            
            index_frame =[temp_frame.index(i) for i in frame_stride2BallRelease]
            parallel = [np.abs(temp_angle[i]).tolist() for i in index_frame]
            near_parallel_num = [np.abs(temp_angle[i]-90).tolist() for i in index_frame]
            
            if not near_parallel_num:
                raise ValueError("MediaPipe 處理失敗：'near_parallel_num' 列表為空，無法計算最大外旋。")
                
            idx = near_parallel_num.index(min(near_parallel_num))
            maxER_tuple = (frame_stride2BallRelease[idx], parallel[idx]+90)
            print(f"關鍵幀 Max ER: {maxER_tuple[0]}")

            print(f"[AI 流程]：關鍵幀提取完成。")

            # -------------------------------------------------
            # 步驟 5: 產生 JSON (in memory)
            # -------------------------------------------------
            # 呼叫修改後的 convert2Json，它會回傳 dict
            json_dict = convert2Json(peak_leg_tuple, foot_plant_tuple, maxER_tuple, ball_release_tuple,
                       left_elbow_angle,right_elbow_angle,left_knee_angle,right_knee_angle,
                       right_shoulder_abduction,left_shoulder_abduction,horizontal_abduction,fps)
            
            # 直接從 dict 轉換為 string，不再讀取本地檔案
            json_data_as_string = json.dumps(json_dict, indent=2, ensure_ascii=False)

            # -------------------------------------------------
            # 步驟 6: 呼叫 Gemini API
            # -------------------------------------------------
            print(f"[AI 流程]：正在發送最終提示 (Prompt) 到 Gemini...")
            
            # ===== 重試邏輯 =====
            max_retries = 5  # 最多重試 5 次
            retry_count = 0
            final_response_text = None
            
            while retry_count < max_retries:
                try:
                    # 嘗試發送請求
                    response = chat.send_message([gvideo, f"這是一位{side}投手，以下是我給你的影片跟json檔階段的資訊 : {json_data_as_string}，我將關鍵幀都放在keyframe裡了"])
                    
                    # 如果成功，就存下結果並跳出迴圈
                    final_response_text = response.text
                    print(f"[AI 流程]：Gemini 回應已收到！")
                    break 
                    
                except Exception as e:
                    # 檢查是否為 503 (Overloaded) 或 429 (Rate Limit) 錯誤
                    error_str = str(e)
                    if "503" in error_str or "429" in error_str or "Overloaded" in error_str:
                        retry_count += 1
                        wait_time = 2 ** retry_count # 指數退避: 等待 2, 4, 8, 16, 32 秒
                        print(f"[AI 流程]：Gemini 伺服器忙碌 (503/429)，正在等待 {wait_time} 秒後重試 ({retry_count}/{max_retries})...")
                        time.sleep(wait_time)
                    else:
                        # 如果是其他錯誤 (例如程式碼寫錯)，就直接拋出，不重試
                        raise e
            
            if final_response_text is None:
                raise Exception("Gemini API 重試多次後仍然失敗 (503 Overloaded)。")
            
            # 回傳最終的文字
            return final_response_text
            
    except Exception as e:
        print(f"[AI 流程]：在主要 AI 流程中發生錯誤: {e}")
        return f"AI 分析過程中發生錯誤: {str(e)}"
    
    finally:
        # -------------------------------------------------
        # 步驟 7: 清理暫存檔案
        # -------------------------------------------------
        if local_temp_video_path and os.path.exists(local_temp_video_path):
            os.remove(local_temp_video_path)
            print(f"[AI 流程]：暫存影片 {local_temp_video_path} 已刪除。")