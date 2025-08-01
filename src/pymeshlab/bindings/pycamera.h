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

#ifndef PYMESHLAB_PYCAMERA_H
#define PYMESHLAB_PYCAMERA_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

// VCG Library includes
#include <vcg/math/shot.h>
#include <vcg/math/similarity.h>
#include <vcg/space/point3.h>
#include <vcg/space/point2.h>
#include <vcg/math/matrix44.h>
#include <vcg/math/matrix33.h>

// Standard library includes
#include <vector>
#include <string>
#include <sstream>

namespace pymeshlab {

/**
 * @brief Initialize camera-related Python bindings for PyMeshLab
 *
 * This function sets up the Python bindings for the SHOTF camera format
 * support in PyMeshLab. It exposes the vcg::Shotf class and related
 * functionality to Python.
 *
 * @param m The pybind11 module to add the bindings to
 */
void initCamera(pybind11::module& m);

/**
 * @brief Utility class for camera operations
 *
 * This class provides additional utility functions for working with
 * cameras in the context of PyMeshLab operations.
 */
class CameraUtils {
public:
    /**
     * @brief Create a camera from basic parameters
     *
     * @param focal_length Focal length in mm
     * @param image_width Image width in pixels
     * @param image_height Image height in pixels
     * @param sensor_width Sensor width in mm (default 36mm for full frame)
     * @return vcg::Shotf The created camera
     */
    static vcg::Shotf createCamera(
        float focal_length,
        int image_width,
        int image_height,
        float sensor_width = 36.0f
    );

    /**
     * @brief Create a camera from calibration matrix
     *
     * @param K 3x3 calibration matrix
     * @param image_width Image width in pixels
     * @param image_height Image height in pixels
     * @return vcg::Shotf The created camera
     */
    static vcg::Shotf createCameraFromMatrix(
        const vcg::Matrix33f& K,
        int image_width,
        int image_height
    );

    /**
     * @brief Estimate camera pose from point correspondences
     *
     * @param world_points 3D world points
     * @param image_points Corresponding 2D image points
     * @param camera Camera with known intrinsics
     * @return bool True if pose estimation succeeded
     */
    static bool estimatePose(
        const std::vector<vcg::Point3f>& world_points,
        const std::vector<vcg::Point2f>& image_points,
        vcg::Shotf& camera
    );

    /**
     * @brief Convert between different camera coordinate systems
     *
     * @param shot Input camera
     * @param from_convention Source convention (e.g., "opencv", "meshlab", "colmap")
     * @param to_convention Target convention
     * @return vcg::Shotf Converted camera
     */
    static vcg::Shotf convertCoordinateSystem(
        const vcg::Shotf& shot,
        const std::string& from_convention,
        const std::string& to_convention
    );

    /**
     * @brief Validate camera parameters
     *
     * @param shot Camera to validate
     * @return std::string Error message, empty if valid
     */
    static std::string validateCamera(const vcg::Shotf& shot);
};

/**
 * @brief SHOTF file format operations
 *
 * This class provides specialized operations for the SHOTF file format
 * used by MeshLab for storing camera information.
 */
class ShotfIO {
public:
    /**
     * @brief Read SHOTF file header information
     *
     * @param filename Path to SHOTF file
     * @return std::map<std::string, std::string> Header information
     */
    static std::map<std::string, std::string> readShotfHeader(const std::string& filename);

    /**
     * @brief Write SHOTF file with custom header
     *
     * @param shot Camera to save
     * @param filename Output filename
     * @param metadata Additional metadata to include
     * @return bool True if successful
     */
    static bool writeShotfWithMetadata(
        const vcg::Shotf& shot,
        const std::string& filename,
        const std::map<std::string, std::string>& metadata
    );

    /**
     * @brief Check if file is a valid SHOTF file
     *
     * @param filename Path to file
     * @return bool True if valid SHOTF file
     */
    static bool isShotfFile(const std::string& filename);

    /**
     * @brief Get SHOTF file version
     *
     * @param filename Path to SHOTF file
     * @return std::string Version string
     */
    static std::string getShotfVersion(const std::string& filename);
};

} // namespace pymeshlab

#endif // PYMESHLAB_PYCAMERA_H