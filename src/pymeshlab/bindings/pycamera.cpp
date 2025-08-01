/****************************************************************************
* PyMeshLab                                                         o o     *
*                                                                _   O  _    *
* Copyright(C) 2019-2021                                           \/)\/     *
* Visual Computing Lab                                            /\/|       *
* ISTI - Italian National Research Council                           |       *
*                                                                    \       *
* All rights reserved.                                                       *
*                                                                            *
* This program is free software; you can redistribute it and/or modify      *
* it under the terms of the GNU General Public License as published by      *
* the Free Software Foundation; either version 2 of the License, or         *
* (at your option) any later version.                                       *
*                                                                            *
* This program is distributed in the hope that it will be useful,           *
* but WITHOUT ANY WARRANTY; without even the implied warranty of            *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             *
* GNU General Public License (http://www.gnu.org/licenses/gpl.txt)          *
* for more details.                                                          *
****************************************************************************/

#include "pycamera.h"
#include <vcg/math/shot.h>
#include <vcg/complex/algorithms/shot.h>
#include <vcg/complex/algorithms/camera_utils.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <fstream>
#include <sstream>

namespace pymeshlab {

void pymeshlab::initCamera(pybind11::module& m)
{
    py::class_<vcg::Shotf>(m, "Camera")
        .def(py::init<>())
        .def(py::init<const vcg::Shotf&>())

        // Intrinsic parameters
        .def("set_intrinsics", &vcg::Shotf::SetIntrinsics,
             "Set camera intrinsic parameters",
             py::arg("focal_x"), py::arg("focal_y"),
             py::arg("principal_x"), py::arg("principal_y"),
             py::arg("viewport_width"), py::arg("viewport_height"))

        .def("get_focal_x", [](const vcg::Shotf& shot) {
            return shot.Intrinsics.FocalMm.X();
        })
        .def("get_focal_y", [](const vcg::Shotf& shot) {
            return shot.Intrinsics.FocalMm.Y();
        })
        .def("get_principal_point", [](const vcg::Shotf& shot) {
            return std::make_pair(shot.Intrinsics.CenterPx.X(), shot.Intrinsics.CenterPx.Y());
        })
        .def("get_viewport_size", [](const vcg::Shotf& shot) {
            return std::make_pair(shot.Intrinsics.ViewportPx.X(), shot.Intrinsics.ViewportPx.Y());
        })

        // Extrinsic parameters
        .def("set_extrinsics", [](vcg::Shotf& shot, const vcg::Point3f& translation,
                                  const vcg::Matrix33f& rotation) {
            shot.Extrinsics.SetTra(translation);
            shot.Extrinsics.SetRot(rotation);
        }, "Set camera extrinsic parameters", py::arg("translation"), py::arg("rotation"))

        .def("get_translation", [](const vcg::Shotf& shot) {
            return shot.Extrinsics.Tra();
        })
        .def("get_rotation_matrix", [](const vcg::Shotf& shot) {
            return shot.Extrinsics.Rot();
        })

        // Camera position and orientation
        .def("set_view_point", [](vcg::Shotf& shot, const vcg::Point3f& viewpoint) {
            shot.SetViewPoint(viewpoint);
        }, "Set camera viewpoint", py::arg("viewpoint"))

        .def("get_view_point", [](const vcg::Shotf& shot) {
            return shot.GetViewPoint();
        })

        .def("set_view_direction", [](vcg::Shotf& shot, const vcg::Point3f& direction) {
            shot.SetViewDirection(direction);
        }, "Set camera view direction", py::arg("direction"))

        .def("get_view_direction", [](const vcg::Shotf& shot) {
            return shot.GetViewDirection();
        })

        // Distortion parameters
        .def("set_lens_distortion", [](vcg::Shotf& shot, float k1, float k2, float p1, float p2, float k3 = 0.0f) {
            shot.Intrinsics.DistorCenterPx = shot.Intrinsics.CenterPx;
            shot.Intrinsics.k[0] = k1;
            shot.Intrinsics.k[1] = k2;
            shot.Intrinsics.k[2] = k3;
            shot.Intrinsics.k[3] = p1;
            shot.Intrinsics.k[4] = p2;
        }, "Set lens distortion parameters",
           py::arg("k1"), py::arg("k2"), py::arg("p1"), py::arg("p2"), py::arg("k3") = 0.0f)

        .def("get_distortion_coefficients", [](const vcg::Shotf& shot) {
            return std::vector<float>(shot.Intrinsics.k, shot.Intrinsics.k + 5);
        })

        // Utility methods
        .def("project", [](const vcg::Shotf& shot, const vcg::Point3f& point) {
            vcg::Point2f projected;
            shot.Project(point, projected);
            return projected;
        }, "Project 3D point to 2D image coordinates", py::arg("point"))

        .def("unproject", [](const vcg::Shotf& shot, const vcg::Point2f& point, float depth) {
            vcg::Point3f unprojected;
            shot.UnProject(point, depth, unprojected);
            return unprojected;
        }, "Unproject 2D point with depth to 3D coordinates", py::arg("point"), py::arg("depth"))

        .def("depth", [](const vcg::Shotf& shot, const vcg::Point3f& point) {
            return shot.Depth(point);
        }, "Get depth of 3D point from camera", py::arg("point"))

        .def("is_valid", [](const vcg::Shotf& shot) {
            return shot.IsValid();
        }, "Check if camera parameters are valid")

        // File I/O
        .def("load_shotf", [](vcg::Shotf& shot, const std::string& filename) {
            return vcg::ReadShotFromFile<vcg::Shotf>(filename, shot);
        }, "Load camera from SHOTF file", py::arg("filename"))

        .def("save_shotf", [](const vcg::Shotf& shot, const std::string& filename) {
            // This is the correct call with the append parameter set to false
            return vcg::WriteShotToFile<vcg::Shotf>(shot, filename, false);
        }, "Save camera to SHOTF file", py::arg("filename"))

        // Transformation matrices
        .def("get_projection_matrix", [](const vcg::Shotf& shot) {
            vcg::Matrix44f proj;
            shot.GetProjectionMatrix(proj);
            return proj;
        }, "Get OpenGL-style projection matrix")

        .def("get_modelview_matrix", [](const vcg::Shotf& shot) {
            vcg::Matrix44f modelview;
            shot.GetModelViewMatrix(modelview);
            return modelview;
        }, "Get OpenGL-style modelview matrix")

        // Camera calibration matrix (intrinsic matrix)
        .def("get_calibration_matrix", [](const vcg::Shotf& shot) {
            vcg::Matrix33f K;
            shot.GetIntrinsicsMatrix(K);
            return K;
        }, "Get camera calibration (intrinsic) matrix")

        // Similarity transformation
        .def("apply_similarity", [](vcg::Shotf& shot, const vcg::Similarityf& sim) {
            shot.ApplySimilarity(sim);
        }, "Apply similarity transformation to camera", py::arg("similarity"))

        // Camera frustum
        .def("get_frustum", [](const vcg::Shotf& shot, float near_plane, float far_plane) {
            std::vector<vcg::Point3f> frustum(8);
            shot.GetFrustum(frustum, near_plane, far_plane);
            return frustum;
        }, "Get camera frustum vertices", py::arg("near_plane"), py::arg("far_plane"))

        // String representation
        .def("__repr__", [](const vcg::Shotf& shot) {
            std::ostringstream oss;
            oss << "Camera(";
            oss << "focal=(" << shot.Intrinsics.FocalMm.X() << ", " << shot.Intrinsics.FocalMm.Y() << "), ";
            oss << "center=(" << shot.Intrinsics.CenterPx.X() << ", " << shot.Intrinsics.CenterPx.Y() << "), ";
            oss << "viewport=(" << shot.Intrinsics.ViewportPx.X() << ", " << shot.Intrinsics.ViewportPx.Y() << "), ";
            vcg::Point3f viewpoint = shot.GetViewPoint();
            oss << "viewpoint=(" << viewpoint.X() << ", " << viewpoint.Y() << ", " << viewpoint.Z() << ")";
            oss << ")";
            return oss.str();
        });

    // Camera list operations for MeshSet
    py::class_<std::vector<vcg::Shotf>>(m, "CameraList")
        .def(py::init<>())
        .def("__len__", [](const std::vector<vcg::Shotf>& v) { return v.size(); })
        .def("__getitem__", [](const std::vector<vcg::Shotf>& v, size_t i) {
            if (i >= v.size()) throw py::index_error();
            return v[i];
        })
        .def("__setitem__", [](std::vector<vcg::Shotf>& v, size_t i, const vcg::Shotf& shot) {
            if (i >= v.size()) throw py::index_error();
            v[i] = shot;
        })
        .def("append", [](std::vector<vcg::Shotf>& v, const vcg::Shotf& shot) {
            v.push_back(shot);
        })
        .def("clear", &std::vector<vcg::Shotf>::clear);

    // Utility functions for SHOTF format
    m.def("load_shotf_file", [](const std::string& filename) {
        vcg::Shotf shot;
        if (vcg::ReadShotFromFile<vcg::Shotf>(filename, shot)) {
            return shot;
        }
        throw std::runtime_error("Failed to load SHOTF file: " + filename);
    }, "Load a single camera from SHOTF file", py::arg("filename"));

    m.def("save_shotf_file", [](const vcg::Shotf& shot, const std::string& filename) {
        if (!vcg::WriteShotToFile<vcg::Shotf>(shot, filename)) {
            throw std::runtime_error("Failed to save SHOTF file: " + filename);
        }
    }, "Save a single camera to SHOTF file", py::arg("camera"), py::arg("filename"));

    m.def("load_multiple_shotf", [](const std::vector<std::string>& filenames) {
        std::vector<vcg::Shotf> cameras;
        for (const auto& filename : filenames) {
            vcg::Shotf shot;
            if (vcg::ReadShotFromFile<vcg::Shotf>(filename, shot)) {
                cameras.push_back(shot);
            } else {
                throw std::runtime_error("Failed to load SHOTF file: " + filename);
            }
        }
        return cameras;
    }, "Load multiple cameras from SHOTF files", py::arg("filenames"));

    } // namespace pymeshlab

// Implementation of CameraUtils methods
vcg::Shotf CameraUtils::createCamera(
    float focal_length,
    int image_width,
    int image_height,
    float sensor_width)
{
    vcg::Shotf shot;

    // Calculate focal length in pixels
    float focal_x = (focal_length / sensor_width) * image_width;
    float focal_y = focal_x; // Assume square pixels

    // Principal point at center of image
    float principal_x = image_width / 2.0f;
    float principal_y = image_height / 2.0f;

    // Set intrinsics
    shot.SetIntrinsics(focal_x, focal_y, principal_x, principal_y,
                      image_width, image_height);

    return shot;
}

vcg::Shotf CameraUtils::createCameraFromMatrix(
    const vcg::Matrix33f& K,
    int image_width,
    int image_height)
{
    vcg::Shotf shot;

    // Extract parameters from calibration matrix
    float focal_x = K[0][0];
    float focal_y = K[1][1];
    float principal_x = K[0][2];
    float principal_y = K[1][2];

    // Set intrinsics
    shot.SetIntrinsics(focal_x, focal_y, principal_x, principal_y,
                      image_width, image_height);

    return shot;
}

bool CameraUtils::estimatePose(
    const std::vector<vcg::Point3f>& world_points,
    const std::vector<vcg::Point2f>& image_points,
    vcg::Shotf& camera)
{
    if (world_points.size() != image_points.size() || world_points.empty()) {
        return false;
    }

    // Make copies of the points (VCG functions might modify them)
    std::vector<vcg::Point3f> wp = world_points;
    std::vector<vcg::Point2f> ip = image_points;

    // Estimate pose
    return vcg::PoseFromCorrespondences(wp, ip, camera);
}

vcg::Shotf CameraUtils::convertCoordinateSystem(
    const vcg::Shotf& shot,
    const std::string& from_convention,
    const std::string& to_convention)
{
    vcg::Shotf converted = shot;

    // MeshLab uses a right-handed coordinate system: X right, Y up, Z backward
    // OpenCV/Colmap use: X right, Y down, Z forward

    if (from_convention == to_convention) {
        return shot; // No conversion needed
    }

    // Create transformation matrices for coordinate system conversion
    vcg::Matrix33f flipY;
    flipY.SetIdentity();
    flipY[1][1] = -1.0f;  // Flip Y axis

    vcg::Matrix33f flipZ;
    flipZ.SetIdentity();
    flipZ[2][2] = -1.0f;  // Flip Z axis

    // Combined transformation (Y and Z flip)
    vcg::Matrix33f flipYZ = flipY * flipZ;

    if ((from_convention == "opencv" || from_convention == "colmap") &&
        to_convention == "meshlab") {
        // OpenCV/Colmap to MeshLab: need to flip Y and Z
        converted.Extrinsics.SetRot(flipYZ * shot.Extrinsics.Rot());
        converted.Extrinsics.SetTra(flipYZ * shot.Extrinsics.Tra());
    }
    else if ((from_convention == "meshlab") &&
             (to_convention == "opencv" || to_convention == "colmap")) {
        // MeshLab to OpenCV/Colmap: need to flip Y and Z
        converted.Extrinsics.SetRot(flipYZ * shot.Extrinsics.Rot());
        converted.Extrinsics.SetTra(flipYZ * shot.Extrinsics.Tra());
    }
    else {
        // For other conventions, you might need additional transformations
        // This is a simplified implementation
        throw std::runtime_error("Unsupported coordinate system conversion: " +
                                from_convention + " to " + to_convention);
    }

    return converted;
}

std::string CameraUtils::validateCamera(const vcg::Shotf& shot) {
    if (!shot.IsValid()) {
        return "Camera parameters are invalid";
    }

    if (shot.Intrinsics.FocalMm.X() <= 0 || shot.Intrinsics.FocalMm.Y() <= 0) {
        return "Focal length must be positive";
    }

    if (shot.Intrinsics.ViewportPx.X() <= 0 || shot.Intrinsics.ViewportPx.Y() <= 0) {
        return "Viewport dimensions must be positive";
    }

    return "";
}

// Implementation of ShotfIO methods
std::map<std::string, std::string> ShotfIO::readShotfHeader(const std::string& filename) {
    std::map<std::string, std::string> header;
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        // SHOTF header lines start with #
        if (line.empty() || line[0] != '#') {
            break; // End of header
        }

        // Remove the # character
        std::string content = line.substr(1);

        // Find the first colon to separate key and value
        size_t colonPos = content.find(':');
        if (colonPos != std::string::npos) {
            std::string key = content.substr(0, colonPos);
            std::string value = content.substr(colonPos + 1);

            // Trim whitespace
            auto trim = [](std::string& s) {
                s.erase(0, s.find_first_not_of(" \t"));
                s.erase(s.find_last_not_of(" \t") + 1);
            };

            trim(key);
            trim(value);

            header[key] = value;
        }
    }

    return header;
}

bool ShotfIO::writeShotfWithMetadata(
    const vcg::Shotf& shot,
    const std::string& filename,
    const std::map<std::string, std::string>& metadata)
{
    // First, write the SHOTF data to a temporary file
    std::string tempFilename = filename + ".tmp";

    // Use VCG's function to write the actual SHOTF data
    if (!vcg::WriteShotToFile<vcg::Shotf>(shot, tempFilename)) {
        return false;
    }

    // Read the temporary file content
    std::ifstream tempFile(tempFilename, std::ios::binary);
    if (!tempFile.is_open()) {
        return false;
    }

    std::vector<char> buffer((std::istreambuf_iterator<char>(tempFile)),
                             std::istreambuf_iterator<char>());
    tempFile.close();

    // Delete the temporary file
    std::remove(tempFilename.c_str());

    // Write the final file with metadata header
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile.is_open()) {
        return false;
    }

    // Write standard SHOTF header
    outFile << "# SHOTF 1.0" << std::endl;
    outFile << "# Created by PyMeshLab" << std::endl;

    // Write custom metadata
    for (const auto& pair : metadata) {
        outFile << "# " << pair.first << ": " << pair.second << std::endl;
    }

    // Write a blank line to separate header from data
    outFile << std::endl;

    // Write the binary SHOTF data
    if (!buffer.empty()) {
        outFile.write(&buffer[0], buffer.size());
    }

    return true;
}

bool ShotfIO::isShotfFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    std::string line;
    if (std::getline(file, line)) {
        // SHOTF files typically start with a comment line
        return line.find("#") == 0;
    }

    return false;
}

std::string ShotfIO::getShotfVersion(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return "";
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.find("# SHOTF") == 0) {
            // Extract version number
            size_t pos = line.find("SHOTF");
            if (pos != std::string::npos) {
                std::string versionStr = line.substr(pos + 5);
                // Trim whitespace
                auto trim = [](std::string& s) {
                    s.erase(0, s.find_first_not_of(" \t"));
                    s.erase(s.find_last_not_of(" \t") + 1);
                };
                trim(versionStr);
                return versionStr;
            }
        }
        else if (line.empty() || line[0] != '#') {
            break; // End of header
        }
    }

    return "1.0"; // Default version if not specified
}
}//namespace pymeshlab
