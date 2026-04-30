import numpy as np
import pyrealsense2 as rs
import cv2
import open3d as o3d
import os
import argparse
from realsense_depth import DepthCamera
from utils import createPointCloudO3D

# --- Configuration ---
resolution_width, resolution_height = (848, 480)

def get_config():
    """รับค่า configuration จาก command-line arguments หรือ user input"""
    parser = argparse.ArgumentParser(
        description='Capture Multiple Views and Create Point Cloud',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python capture_multiview.py --save-dir realsensepy/my_data --view-names top front side
  python capture_multiview.py -sd realsensepy/my_data -vn top front side
  python capture_multiview.py  (ใช้ค่า default)
        '''
    )
    
    parser.add_argument(
        '-sd', '--save-dir',
        type=str,
        default="realsensepy/capture_multiview",
        help='โฟลเดอร์สำหรับบันทึกข้อมูล (default: realsensepy/capture_multiview)'
    )
    parser.add_argument(
        '-vn', '--view-names',
        type=str,
        nargs='+',
        default=["view_1", "view_2", "view_3"],
        help='ชื่อมุมมอง เช่น top front side (default: view_1 view_2 view_3)'
    )
    parser.add_argument(
        '-n', '--num-views',
        type=int,
        default=3,
        help='จำนวนมุมมองที่ต้องการบันทึก (default: 3)'
    )
    
    args = parser.parse_args()
    
    save_dir = args.save_dir
    view_names = args.view_names[:args.num_views]
    
    # สร้าง folder ถ้ายังไม่มี
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"\n{'='*60}")
    print(f"[CONFIG] Multi-View Point Cloud Capture System")
    print(f"{'='*60}")
    print(f"Save Directory: {os.path.abspath(save_dir)}")
    print(f"View Names: {', '.join(view_names)}")
    print(f"Number of Views: {len(view_names)}")
    print(f"{'='*60}\n")
    
    return save_dir, view_names

SAVE_DIR, VIEW_NAMES = get_config()

def main():
    print("\n[INIT] Initializing RealSense Camera...")
    Realsensed435Cam = DepthCamera(resolution_width, resolution_height)
    if Realsensed435Cam is None:
        print("[ERROR] Failed to initialize camera. Exiting...")
        return
    else:
        print("[INIT] Camera initialized successfully!")
    
    print("[INIT] Creating display window...")
    cv2.namedWindow("Multi-View Capture", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Multi-View Capture", 848, 480)
    print("[INIT] Window created!")
    
    print("--- MULTI-VIEW CAPTURE MODE ---")
    print("Instructions:")
    print("  SPACE: บันทึกภาพของมุมมองปัจจุบัน")
    print("  'r': รีเซ็ตและเริ่มใหม่\n")
    
    captured_data = {
        'color_frames': [],
        'depth_frames': [],
        'view_names': []
    }
    
    current_view_idx = 0
    frame_count = 0
    no_frame_count = 0
    
    while current_view_idx < len(VIEW_NAMES):
        ret, depth_raw_frame, color_raw_frame = Realsensed435Cam.get_raw_frame()
        if not ret:
            no_frame_count += 1
            print(f".", end="", flush=True)
            if no_frame_count > 500:  # ถ้าเกิน 500 frame ไม่ได้
                print(f"\n[ERROR] Cannot get frames from camera. Exiting...")
                break
            continue
        
        no_frame_count = 0
        frame_count += 1
        color_frame = np.asanyarray(color_raw_frame.get_data())
        display_frame = color_frame.copy()
        
        # แสดงข้อมูลสถานะ
        cv2.putText(display_frame, f"View {current_view_idx + 1}/{len(VIEW_NAMES)}: {VIEW_NAMES[current_view_idx]}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        cv2.putText(display_frame, "SPACE: Capture | 'r': Reset | 'q': Skip", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(display_frame, f"Frame: {frame_count}", 
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # แสดงรายการมุมมองที่บันทึกแล้ว
        for i, view_name in enumerate(VIEW_NAMES):
            status = "✓" if i < current_view_idx else "○"
            color = (0, 255, 0) if i < current_view_idx else (255, 255, 255)
            cv2.putText(display_frame, f"{status} {view_name}", 
                        (10, 130 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        try:
            cv2.imshow("Multi-View Capture", display_frame)
        except Exception as e:
            print(f"\n[ERROR] Failed to display frame: {e}")
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # SPACE
            print(f"\n[CAPTURE] Saving '{VIEW_NAMES[current_view_idx]}'...")
            
            # บันทึกภาพ color
            color_path = os.path.join(SAVE_DIR, f"{VIEW_NAMES[current_view_idx]}_color.png")
            cv2.imwrite(color_path, color_frame)
            
            # เก็บข้อมูลสำหรับ point cloud
            captured_data['color_frames'].append(color_raw_frame)
            captured_data['depth_frames'].append(depth_raw_frame)
            captured_data['view_names'].append(VIEW_NAMES[current_view_idx])
            
            print(f"  ✓ Color saved: {color_path}")
            
            current_view_idx += 1
            frame_count = 0
            
            # เมื่อเก็บครบทุกมุม ให้สร้าง combined point cloud ทันที
            if current_view_idx == len(VIEW_NAMES):
                print(f"\n[AUTO COMPLETE] All {len(VIEW_NAMES)} views captured!")
                break
        
        elif key == ord('r'):  # RESET
            print("\n[RESET] Clearing all captured data...")
            captured_data = {
                'color_frames': [],
                'depth_frames': [],
                'view_names': []
            }
            current_view_idx = 0
            frame_count = 0
        
        elif key == ord('q'):  # SKIP current view
            print(f"\n[SKIP] Skipping view '{VIEW_NAMES[current_view_idx]}'")
            current_view_idx += 1
            frame_count = 0
            
            if current_view_idx == len(VIEW_NAMES):
                print(f"\n[AUTO COMPLETE] All {len(VIEW_NAMES)} views captured!")
                break
    
    Realsensed435Cam.release()
    cv2.destroyAllWindows()
    
    # --- เมื่อเก็บครบทุกมุม ให้สร้าง Combined Point Cloud ---
    if len(captured_data['color_frames']) > 0:
        print(f"\n{'='*60}")
        print(f"[PROCESSING] Creating Mapped Point Clouds...")
        print(f"[PROCESSING] Total captured views: {len(captured_data['color_frames'])}")
        print(f"{'='*60}\n")
        
        all_pcds = []
        all_markers = []
        
        try:
            # สร้าง Point Cloud จากแต่ละมุมมอง
            for i, (color_frame, depth_frame, view_name) in enumerate(zip(
                captured_data['color_frames'], 
                captured_data['depth_frames'],
                captured_data['view_names']
            )):
                print(f"[{i+1}/{len(captured_data['view_names'])}] Creating PCD for view: {view_name}...", end="", flush=True)
                
                # สร้าง point cloud
                pcd = createPointCloudO3D(color_frame, depth_frame)
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
                
                # บันทึก point cloud แต่ละมุม
                pcd_path = os.path.join(SAVE_DIR, f"{view_name}_pointcloud.ply")
                o3d.io.write_point_cloud(pcd_path, pcd)
                print(f" ✓ ({len(pcd.points)} points)")
                
                all_pcds.append(pcd)
                
                # สร้างทรงกลมเพื่อบ่งชี้มุมมอง
                colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                color_idx = i % len(colors)
                
                marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                marker.paint_uniform_color([colors[color_idx][2]/255.0, colors[color_idx][1]/255.0, colors[color_idx][0]/255.0])
                all_markers.append((marker, view_name, colors[color_idx]))
            
            # Register and align Point Clouds
            print(f"\n[REGISTRATION] Aligning {len(all_pcds)} point clouds using ICP...")
            registered_pcds = [all_pcds[0]]
            
            for i in range(1, len(all_pcds)):
                print(f"  Aligning view {i+1}...", end="", flush=True)
                source = all_pcds[i]
                target = registered_pcds[0]
                
                # Downsampling for faster registration
                source_down = source.voxel_down_sample(voxel_size=0.01)
                target_down = target.voxel_down_sample(voxel_size=0.01)
                
                # ICP Registration
                trans_init = np.eye(4)
                reg_p2p = o3d.pipelines.registration.registration_icp(
                    source_down, target_down, max_correspondence_distance=0.1,
                    init=trans_init,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
                )
                
                # Transform original point cloud
                source.transform(reg_p2p.transformation)
                registered_pcds.append(source)
                print(f" ✓ (fitness: {reg_p2p.fitness:.4f})")
            
            # รวม Point Clouds
            print(f"\n[MERGING] Combining all {len(registered_pcds)} registered point clouds...", end="", flush=True)
            combined_pcd = registered_pcds[0]
            for pcd in registered_pcds[1:]:
                combined_pcd += pcd
            print(f" ✓")
            
            combined_path = os.path.join(SAVE_DIR, "combined_pointcloud.ply")
            o3d.io.write_point_cloud(combined_path, combined_pcd)
            print(f"✓ Saved combined: {combined_path}")
            print(f"  Total points: {len(combined_pcd.points)}")
            
            # บันทึกสถิติ
            stats_path = os.path.join(SAVE_DIR, "capture_stats.txt")
            with open(stats_path, "w") as f:
                f.write(f"Number of Views: {len(captured_data['view_names'])}\n")
                f.write(f"View Names: {', '.join(captured_data['view_names'])}\n")
                f.write(f"Resolution: {resolution_width}x{resolution_height}\n")
                f.write(f"Total Points in Combined Cloud: {len(combined_pcd.points)}\n")
                for j, view_name in enumerate(captured_data['view_names']):
                    f.write(f"  View {j+1} ({view_name}): {len(all_pcds[j].points)} points\n")
            print(f"✓ Saved stats: {stats_path}")
            
            print(f"\n{'='*60}")
            print(f"[SUCCESS] All data saved to: {os.path.abspath(SAVE_DIR)}")
            print(f"{'='*60}")
            
            # แสดง visualization
            print(f"\n[VISUALIZATION] Opening 3D Mapped View...", end="", flush=True)
            mat_unlit = o3d.visualization.rendering.MaterialRecord()
            mat_unlit.shader = "defaultUnlit"
            
            geometries = [{"name": "combined_pcd", "geometry": combined_pcd, "material": mat_unlit}]
            
            # เพิ่ม markers สำหรับแต่ละมุมมอง
            for i, (marker, view_name, color) in enumerate(all_markers):
                # วาง marker ที่ตำแหน่งต่างๆ เพื่อแสดง origin ของแต่ละ view
                offset = np.array([0.1 * i, 0, 0])
                marker.translate(offset)
                geometries.append({"name": f"marker_{view_name}", "geometry": marker, "material": mat_unlit})
            
            print(" ✓")
            o3d.visualization.draw(geometries, title="Mapped Multi-View Point Clouds")
            
        except Exception as e:
            print(f"\n[ERROR] Failed to create point clouds: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

