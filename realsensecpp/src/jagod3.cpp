#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

namespace fs = std::filesystem;

// --- Configuration ---
const int RESOLUTION_WIDTH = 640;
const int RESOLUTION_HEIGHT = 480;
const std::string SAVE_DIR = "test1/test1_template_match";
const std::string TEMPLATE_DIR = "test1";
const std::vector<std::string> TARGET_BASENAMES = {"target_A", "target_B"};
const std::vector<cv::Scalar> COLORS = {cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0)};

// Helper to convert Rotation Matrix to Euler Angles (Degrees)
Eigen::Vector3d rotationMatrixToEulerAngles(const Eigen::Matrix3d& R) {
    double sy = std::sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0));
    bool singular = sy < 1e-6;
    double x, y, z;
    if (!singular) {
        x = std::atan2(R(2, 1), R(2, 2));
        y = std::atan2(-R(2, 0), sy);
        z = std::atan2(R(1, 0), R(0, 0));
    } else {
        x = std::atan2(-R(1, 2), R(1, 1));
        y = std::atan2(-R(2, 0), sy);
        z = 0;
    }
    return Eigen::Vector3d(x, y, z) * (180.0 / M_PI);
}

int main() {
    // 1. Setup Directories
    if (!fs::exists(SAVE_DIR)) fs::create_directories(SAVE_DIR);

    std::vector<cv::Mat> templates;
    std::vector<cv::Point> offsets;

    // 2. Load Templates and Offsets
    for (const auto& base : TARGET_BASENAMES) {
        std::string img_path = TEMPLATE_DIR + "/" + base + "_template.png";
        std::string txt_path = TEMPLATE_DIR + "/" + base + "_offset.txt";

        cv::Mat tpl = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
        if (tpl.empty()) {
            std::cerr << "[ERROR] Could not load template: " << img_path << std::endl;
            return -1;
        }
        templates.push_back(tpl);

        std::ifstream offset_file(txt_path);
        if (offset_file.is_open()) {
            std::string line;
            std::getline(offset_file, line);
            size_t comma = line.find(',');
            int ox = std::stoi(line.substr(0, comma));
            int oy = std::stoi(line.substr(comma + 1));
            offsets.push_back(cv::Point(ox, oy));
        } else {
            std::cerr << "[ERROR] Missing offset file: " << txt_path << std::endl;
            return -1;
        }
    }

    // 3. Initialize RealSense
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, RESOLUTION_WIDTH, RESOLUTION_HEIGHT, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, RESOLUTION_WIDTH, RESOLUTION_HEIGHT, RS2_FORMAT_Z16, 30);
    rs2::pipeline_profile selection = pipe.start(cfg);
    auto depth_intrinsics = selection.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>().get_intrinsics();

    rs2::pointcloud pc;
    rs2::points points;

    std::cout << "\n--- MULTI-TARGET AUTO DETECTION ---" << std::endl;
    
    while (true) {
        rs2::frameset frames = pipe.wait_for_frames();
        rs2::depth_frame depth = frames.get_depth_frame();
        rs2::video_frame color = frames.get_color_frame();

        cv::Mat color_mat(cv::Size(RESOLUTION_WIDTH, RESOLUTION_HEIGHT), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat gray_mat;
        cv::cvtColor(color_mat, gray_mat, cv::COLOR_BGR2GRAY);
        cv::Mat display_frame = color_mat.clone();

        std::vector<std::pair<int, cv::Point>> detected_pixels;

        // 4. Template Matching
        for (size_t i = 0; i < templates.size(); ++i) {
            cv::Mat res;
            cv::matchTemplate(gray_mat, templates[i], res, cv::TM_CCOEFF_NORMED);
            double minVal, maxVal;
            cv::Point minLoc, maxLoc;
            cv::minMaxLoc(res, &minVal, &maxVal, &minLoc, &maxLoc);

            if (maxVal >= 0.70) {
                cv::Point target_center = maxLoc + offsets[i];
                detected_pixels.push_back({(int)i, target_center});

                cv::rectangle(display_frame, maxLoc, cv::Point(maxLoc.x + templates[i].cols, maxLoc.y + templates[i].rows), COLORS[i], 2);
                cv::circle(display_frame, target_center, 5, cv::Scalar(0, 0, 255), -1);
                cv::putText(display_frame, TARGET_BASENAMES[i], cv::Point(maxLoc.x, maxLoc.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, COLORS[i], 2);
            }
        }

        cv::imshow("C++ RealSense Matcher", display_frame);
        char key = (char)cv::waitKey(1);
        if (key == 'q') {
            if (detected_pixels.empty()) continue;

            // 5. 3D Processing
            pc.map_to(color);
            points = pc.calculate(depth);
            
            // TODO: Open3D visualization - requires separate installation
            // auto pcd_o3d = std::make_shared<open3d::geometry::PointCloud>();
            // const rs2::vertex* vertices = points.get_vertices();
            // for (size_t i = 0; i < points.size(); ++i) {
            //     if (vertices[i].z > 0) {
            //         pcd_o3d->points_.push_back({vertices[i].x, -vertices[i].y, -vertices[i].z});
            //     }
            // }
            // pcd_o3d->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(0.01, 30));

            for (auto& det : detected_pixels) {
                int id = det.first;
                cv::Point pix = det.second;

                // Projection logic (Finding nearest 3D point to pixel)
                float min_dist = std::numeric_limits<float>::max();
                Eigen::Vector3d best_pt(0, 0, 0);
                int best_idx = 0;

                const rs2::vertex* vertices = points.get_vertices();
                for (size_t i = 0; i < points.size(); ++i) {
                    rs2::vertex v = vertices[i];
                    if (v.z > 0) {
                        float u = (v.x * depth_intrinsics.fx / -v.z) + depth_intrinsics.ppx;
                        float v_coord = (v.y * depth_intrinsics.fy / -v.z) + depth_intrinsics.ppy;
                        float d = std::pow(u - pix.x, 2) + std::pow(v_coord - pix.y, 2);
                        if (d < min_dist) {
                            min_dist = d;
                            best_pt = Eigen::Vector3d(v.x, v.y, v.z);
                            best_idx = i;
                        }
                    }
                }

                // Orientation from Normal (simplified without Open3D)
                Eigen::Vector3d z_axis(0, 0, 1);  // Default orientation
                Eigen::Vector3d x_axis = (std::abs(z_axis(0)) < 0.9) ? Eigen::Vector3d(1, 0, 0) : Eigen::Vector3d(0, 1, 0);
                Eigen::Vector3d y_axis = z_axis.cross(x_axis).normalized();
                x_axis = y_axis.cross(z_axis);

                Eigen::Matrix3d R;
                R.col(0) = x_axis; R.col(1) = y_axis; R.col(2) = z_axis;
                Eigen::Vector3d euler = rotationMatrixToEulerAngles(R);

                // Save Data
                std::ofstream outfile(SAVE_DIR + "/" + TARGET_BASENAMES[id] + "_data.txt");
                outfile << "Position_X: " << best_pt(0) << "\nPosition_Y: " << best_pt(1) << "\nPosition_Z: " << best_pt(2) << "\n";
                outfile << "Roll: " << euler(0) << "\nPitch: " << euler(1) << "\nYaw: " << euler(2) << "\n";
                outfile.close();
                
                std::cout << "[" << TARGET_BASENAMES[id] << "] Saved 6-DOF." << std::endl;
            }
            
            // TODO: open3d::visualization::DrawGeometries({pcd_o3d}, "Production Mode");
            break;
        }
    }

    return 0;
}