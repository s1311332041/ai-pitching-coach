import json

def convert2Json(peak_leg_tuple, foot_plant_tuple, maxER_tuple, ball_release_tuple,
           left_elbow_angle,right_elbow_angle,left_knee_angle,right_knee_angle,
           right_shoulder_abduction,left_shoulder_abduction,horizontal_abduction):
    peak_leg = {
        "Frame": peak_leg_tuple[0]
    }
    foot_plant = {
        "Frame": foot_plant_tuple[0],
        "stride_percentage": foot_plant_tuple[1]
    }
    max_ER = {
        "Frame": maxER_tuple[0],
        "ER(External Rotation)angle": maxER_tuple[1]
    }
    ball_release = {
        "Frame": ball_release_tuple[0]
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
    output_json_path = "json/Pitcher_pose_data.json"
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(frame_landmark, f, indent=4)
    print(f"成功將姿勢數據儲存至: {output_json_path}")