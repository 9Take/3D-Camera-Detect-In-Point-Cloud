import pandas as pd
import numpy as np
import cv2

# 1. โหลดไฟล์ CSV ที่คุณเซฟไว้
# แก้ไขพาธไฟล์ให้ตรงกับเครื่องของคุณครับ
csv_path = "./output/capture_log.csv"
df = pd.read_csv(csv_path)

print(f"Loaded {len(df)} points for optimization...")

# กล้อง (R/t) และ translation ของหุ่นยนต์ อ่านจาก npz — CSV ไม่ได้เก็บเมทริกซ์การหมุน
# npz = np.load("./output/hand_eye_result.npz")
# t_target2cam = [npz['t_target2cam'][i] for i in range(len(df))]
# t_gripper2base = [npz['t_gripper2base'][i] for i in range(len(df))]
# R_target2cam = [npz['R_target2cam'][i] for i in range(len(df))]

# ✅ โค้ดใหม่ (ดึงระยะหุ่นยนต์จาก df ใน CSV โดยตรง เพื่อให้ตรงรอบกับค่า A, B, C)
npz = np.load("./output/hand_eye_result.npz")
t_target2cam = [npz['t_target2cam'][i] for i in range(len(df))]
R_target2cam = [npz['R_target2cam'][i] for i in range(len(df))]

# ดึงพิกัด X, Y, Z ของหุ่นยนต์จากคอลัมน์ใน CSV แทน
# ⚠️ หมายเหตุ: ให้เปลี่ยนชื่อคอลัมน์ 'X', 'Y', 'Z' ให้ตรงกับที่คุณเซฟไว้ใน capture_log.csv 
# (เช่น ถ้าใน CSV หัวตารางชื่อ 'robot_x' ให้เปลี่ยนเป็น r['robot_x'] ครับ)
t_gripper2base = [np.array([[r['X']], [r['Y']], [r['Z']]]) for _, r in df.iterrows()]

# 2. ฟังก์ชันทดสอบการคำนวณเมทริกซ์หุ่นยนต์ในรูปแบบต่าง ๆ
def get_rotation_matrix(A, B, C, order, sign_A, sign_B, sign_C):
    a = np.radians(A * sign_A)
    b = np.radians(B * sign_B)
    c = np.radians(C * sign_C)

    R_z = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    R_y = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    R_x = np.array([[1, 0, 0], [0, np.cos(c), -np.sin(c)], [0, np.sin(c), np.cos(c)]])

    # ทดสอบลำดับการคูณเมทริกซ์
    if order == 0: return R_z @ R_y @ R_x  # Standard KUKA (Intrinsic Z-Y-X)
    if order == 1: return R_x @ R_y @ R_z  # Extrinsic Z-Y-X
    if order == 2: return R_y @ R_x @ R_z
    return R_z @ R_x @ R_y

# 2.5 ตรวจคู่ข้อมูล (desync / flip) ก่อน brute-force
# หลักการ: ระหว่างสองโพสที่ติดกัน "มุมการหมุน" ของกริปเปอร์กับของกล้องต้องเท่ากัน
# (เงื่อนไข AX=XB). ระยะ translation จะไม่เท่ากันเพราะมีระยะคานกล้อง-กริปเปอร์ จึงดูแค่ไว้อ้างอิง
def rot_angle_deg(R):
    """มุมการหมุนรวม (องศา) ของเมทริกซ์หมุน R."""
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))


def diagnose_pairs(df, R_target2cam, t_target2cam, t_gripper2base, ang_tol=8.0):
    # ใช้คอนเวนชัน KUKA มาตรฐาน (ZYX, เครื่องหมาย +) เป็นตัวอ้างอิงสำหรับการหมุนของหุ่นยนต์
    R_g2b = [get_rotation_matrix(r['A'], r['B'], r['C'], 0, 1, 1, 1) for _, r in df.iterrows()]
    print("\n--- Pair diagnostic (consecutive relative motions) ---")
    print(" pair |  rot_robot  rot_cam   d_rot |  mov_robot  mov_cam")
    bad = []
    for i in range(len(R_g2b) - 1):
        ar = rot_angle_deg(R_g2b[i].T @ R_g2b[i + 1])          # มุมหมุนสัมพัทธ์ของกริปเปอร์
        ac = rot_angle_deg(R_target2cam[i].T @ R_target2cam[i + 1])  # ของกล้อง
        mov_r = float(np.linalg.norm(t_gripper2base[i + 1] - t_gripper2base[i]))
        mov_c = float(np.linalg.norm(t_target2cam[i + 1] - t_target2cam[i]))
        d = abs(ar - ac)
        flag = "  <-- mismatch" if d > ang_tol else ""
        if d > ang_tol:
            bad.append((i, i + 1))
        print(f" {i:2d}-{i+1:<2d}| {ar:9.1f} {ac:8.1f} {d:7.1f} | {mov_r:9.1f} {mov_c:8.1f}{flag}")
    if bad:
        flagged = [p for pair in bad for p in pair]
        worst = max(set(flagged), key=flagged.count)
        print(f"\n⚠️ {len(bad)} suspicious pair(s): gripper vs camera rotation disagree > {ang_tol} deg.")
        print(f"   Pose #{worst} appears most in flagged pairs — likely flipped or desynced; drop/recapture it.")
    else:
        print(f"\n✅ All consecutive pairs agree within {ang_tol} deg — pairing looks consistent.")
    return bad


diagnose_pairs(df, R_target2cam, t_target2cam, t_gripper2base)

# 3. ลูปค้นหาคำตอบที่ดีที่สุด (Brute-force)
# ต้องไล่ทุกวิธี: TSAI ล้มเหลวเมื่อมุมหมุนระหว่างโพสใหญ่ — PARK/ANDREFF เสถียรกว่า
best_rms = float('inf')
best_config = None

orders = [0, 1, 2, 3]
signs = [1, -1]
methods = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}

print("Optimizing rotation conventions x methods... Please wait.")

for order in orders:
    for sA in signs:
        for sB in signs:
            for sC in signs:
                # สร้างชุดเมทริกซ์หุ่นยนต์ตามเงื่อนไขรอบนี้
                R_gripper2base = []
                for _, r in df.iterrows():
                    R_g2b = get_rotation_matrix(r['A'], r['B'], r['C'], order, sA, sB, sC)
                    R_gripper2base.append(R_g2b)

                for mname, mflag in methods.items():
                    try:
                        R_c2g, t_c2g = cv2.calibrateHandEye(
                            R_gripper2base, t_gripper2base, R_target2cam, t_target2cam,
                            method=mflag
                        )

                        # คำนวณค่า Residual Spread
                        pts_base = []
                        for Rg, tg, t_t2c in zip(R_gripper2base, t_gripper2base, t_target2cam):
                            p_grip = R_c2g @ t_t2c + t_c2g
                            p_base = Rg @ p_grip + tg
                            pts_base.append(p_base.ravel())

                        pts_base = np.array(pts_base)
                        mean_pt = pts_base.mean(axis=0)
                        resid = np.linalg.norm(pts_base - mean_pt, axis=1)
                        rms = np.sqrt((resid**2).mean())

                        # บันทึกค่าที่ดีที่สุด
                        if rms < best_rms:
                            best_rms = rms
                            best_config = (order, sA, sB, sC, mname)
                    except:
                        continue

print("\n=======================================================")
print(" OPTIMIZATION BREAKTHROUGH RESULT ")
if best_rms < 50.0:
    order_names = ["R_z @ R_y @ R_x", "R_x @ R_y @ R_z", "R_y @ R_x @ R_z", "R_z @ R_x @ R_y"]
    o, sa, sb, sc, mname = best_config
    print(f"✅ Found working convention with RMS Residual: {best_rms:.2f} mm")
    print(f"   Hand-Eye Method             : {mname}")
    print(f"   Matrix Multiplication Order : {order_names[o]}")
    print(f"   Sign Multipliers (A, B, C)  : ({sa}, {sb}, {sc})")
    print("\n👉 นำค่าลอจิกนี้ไปอัปเดตในฟังก์ชัน `rotation_matrix_from_abc` ในโค้ดหลักได้เลย!")
else:
    print(f"❌ ค้นหาไม่สำเร็จ ค่า RMS ต่ำสุดที่ทำได้คือ {best_rms:.2f} mm ซึ่งยังสูงเกินไป")
print("=======================================================")
