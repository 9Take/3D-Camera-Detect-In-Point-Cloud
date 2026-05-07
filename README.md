# 🤖 Project 3D Guidance Robot: Heat Exchanger Vision-to-PLC System
**Project Name:** 3D-Camera-Detect-In-Point-Cloud (Vision-in-a-Box)
**Last Updated:** 2024-05-24 (Update V1.1)
**Hardware:** Intel RealSense D435i, KUKA/Industrial Robot, Mitsubishi PLC (MC Protocol Type3E)

## 📌 1. Project Overview
ระบบ Vision สำหรับตรวจหาตำแหน่งจุดเชื่อมบน Heat Exchanger โดยใช้กล้อง 3D ระบุพิกัด 6-DOF (X, Y, Z, Roll, Pitch, Yaw) ส่งข้อมูลแบบหลายจุด (สูงสุด 5 ชุด) ไปยัง PLC ผ่าน Protocolสื่อสารแบบ Binary (MC Protocol) เพื่อควบคุมแขนกล

## 📂 2. Project Structure
```text
project_root/
├── main.py                   # ลูปหลัก: จัดการ Multi-point Logic, JSON Save, และ PLC Handshake
├── config.yaml               # โครงสร้างใหม่ (Heartbeat D1000, Trigger M1000, Data D1001++)
├── core/ 
│       ├── detector.py       # SIFT Matching & Confidence Score
│       └── transformer.py    # [Update] คำนวณ 6-DOF โดยคืนค่า Orientation เป็น Radian
├── communication/ 
│        ├── realsense.py     # RealSense Wrapper
│        └── plc_comm.py      # PLC Communicator (Handle 16-bit Integer Scaling)
└── data/ 
        ├── logs/                 
        |    ├──  position_mem.json # [Update] บันทึก Snapshot ทุกเป้าหมายที่พบก่อนส่ง PLC
        |    └──  current_detect.json 
        └── templates/

```