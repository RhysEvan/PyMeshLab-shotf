/*****************************************************************************
 * PyMeshLab                                                         o o     *
 * A versatile mesh processing toolbox                             o     o   *
 *                                                                _   O  _   *
 * Copyright(C) 2005-2021                                           \/)\/    *
 * Visual Computing Lab                                            /\/|      *
 * ISTI - Italian National Research Council                           |      *
 *                                                                    \      *
 * All rights reserved.                                                      *
 *                                                                           *
 * This program is free software; you can redistribute it and/or modify      *
 * it under the terms of the GNU General Public License as published by      *
 * the Free Software Foundation; either version 2 of the License, or         *
 * (at your option) any later version.                                       *
 *                                                                           *
 * This program is distributed in the hope that it will be useful,           *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of            *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             *
 * GNU General Public License (http://www.gnu.org/licenses/gpl.txt)          *
 * for more details.                                                         *
 *                                                                           *
 ****************************************************************************/
#include "pymeshset.h"
#include "pycamera.h"

#include "docs/pymeshset_doc.h"
#include "pymeshlab/helpers/common.h"
#include "pymeshlab/meshset.h"
#include <pybind11/eval.h>
#include <vcg/../wrap/io_trimesh/import_obj.h>
#include <vcg/complex/algorithms/shot.h>

namespace py = pybind11;

void pymeshlab::initMeshSet(pybind11::module& m)
{
	py::class_<MeshSet> meshSetClass(m, "MeshSet");

	// constructor
	meshSetClass.def(py::init<bool>(), doc::PYMS_INIT_VERB, py::arg("verbose") = false);

	meshSetClass.def("__len__", &MeshSet::meshNumber, doc::PYMS_NUMBER_MESHES_DOC);
	meshSetClass.def("__str__", &MeshSet::printStatus, doc::PYMS_PRINT_STATUS);
	meshSetClass.def(
		"__getitem__",
		&MeshSet::mesh,
		doc::PYMS_MESH,
		py::arg("id"),
		py::return_value_policy::reference);
	meshSetClass.def(
		"__iter__",
		[](const MeshSet& s) { return py::make_iterator(s.meshBegin(), s.meshEnd()); },
		py::keep_alive<0, 1>() /* Essential: keep object alive while iterator exists*/);

	meshSetClass.def(
		"set_verbosity", &MeshSet::setVerbosity, doc::PYMS_SET_VERBOSITY_DOC, py::arg("verbosity"));
	meshSetClass.def(
		"set_parameter_verbosity",
		&MeshSet::setParameterVerbosity,
		doc::PYMS_SET_PAR_VERBOSITY_DOC,
		py::arg("verbosity"));
	meshSetClass.def("mesh_number", &MeshSet::meshNumber, doc::PYMS_MESH_NUMBER_DOC);
	meshSetClass.def("raster_number", &MeshSet::rasterNumber, doc::PYMS_RASTER_NUMBER_DOC);
	// todo - deprecated functions - to remove
	meshSetClass.def("number_meshes", &MeshSet::numberMeshes, doc::PYMS_NUMBER_MESHES_DOC);
	meshSetClass.def("number_rasters", &MeshSet::numberRasters, doc::PYMS_NUMBER_RASTERS_DOC);
	meshSetClass.def(
		"set_current_mesh",
		&MeshSet::setCurrentMesh,
		doc::PYMS_SET_CURRENT_MESH,
		py::arg("new_curr_id"));
	meshSetClass.def(
		"current_mesh",
		&MeshSet::currentMesh,
		doc::PYMS_CURRENT_MESH,
		py::return_value_policy::reference);
	meshSetClass.def("current_mesh_id", &MeshSet::currentMeshId, doc::PYMS_CURRENT_MESH_ID);
	meshSetClass.def(
		"mesh_id_exists", &MeshSet::meshIdExists, doc::PYMS_MESH_ID_EXISTS, py::arg("id"));
	meshSetClass.def(
		"mesh", &MeshSet::mesh, doc::PYMS_MESH, py::arg("id"), py::return_value_policy::reference);
	meshSetClass.def(
		"set_current_mesh_visibility",
		&MeshSet::setCurrentMeshVisibility,
		doc::PYMS_SET_CURRENT_MESH_VISIBILITY,
		py::arg("visibility"));
	meshSetClass.def(
		"set_mesh_visibility",
		&MeshSet::setMeshVisibility,
		doc::PYMS_SET_MESH_VISIBILITY,
		py::arg("id"),
		py::arg("visibility"));
	meshSetClass.def(
		"is_current_mesh_visible",
		&MeshSet::isCurrentMeshVisible,
		doc::PYMS_IS_CURRENT_MESH_VISIBLE);
	meshSetClass.def(
		"is_mesh_visible", &MeshSet::isMeshVisible, doc::PYMS_IS_MESH_VISIBLE, py::arg("id"));
	meshSetClass.def(
		"load_new_mesh", &MeshSet::loadNewMesh, doc::PYMS_LOAD_NEW_MESH, py::arg("file_name"));
	meshSetClass.def(
		"save_current_mesh",
		&MeshSet::saveCurrentMesh,
		doc::PYMS_SAVE_CURRENT_MESH,
		py::arg("file_name"),
		py::arg("save_textures")   = true,
		py::arg("texture_quality") = -1);
	meshSetClass.def(
		"load_new_raster",
		&MeshSet::loadNewRaster,
		doc::PYMS_LOAD_NEW_RASTER,
		py::arg("file_name"));
	meshSetClass.def(
		"add_mesh",
		&MeshSet::addMesh,
		doc::PYMS_ADD_MESH,
		py::arg("mesh"),
		py::arg("mesh_name")      = "",
		py::arg("set_as_current") = true);
	meshSetClass.def("clear", &MeshSet::clear, doc::PYMS_CLEAR);
	meshSetClass.def(
		"load_project", &MeshSet::loadProject, doc::PYMS_LOAD_PROJECT, py::arg("file_name"));
	meshSetClass.def(
		"save_project", &MeshSet::saveProject, doc::PYMS_SAVE_PROJECT, py::arg("file_name"));
	meshSetClass.def(
		"apply_filter", &MeshSet::applyFilter, doc::PYMS_APPLY_FILTER, py::arg("filter_name"));
	meshSetClass.def(
		"load_filter_script",
		&MeshSet::loadFilterScript,
		doc::PYMS_LOAD_FILTER_SCRIPT,
		py::arg("filter_script_name"));
	meshSetClass.def(
		"save_filter_script",
		&MeshSet::saveFilterScript,
		doc::PYMS_SAVE_FILTER_SCRIPT,
		py::arg("filter_script_name"));
	meshSetClass.def(
		"clear_filter_script", &MeshSet::clearFilterScript, doc::PYMS_CLEAR_FILTER_SCRIPT);
	meshSetClass.def(
		"apply_filter_script", &MeshSet::applyFilterScript, doc::PYMS_APPLY_FILTER_SCRIPT);
	meshSetClass.def("print_status", &MeshSet::printStatus, doc::PYMS_PRINT_STATUS);
	meshSetClass.def(
		"filter_parameter_values",
		&MeshSet::filterParameterValues,
		doc::PYMS_FILTER_PARAMETER_VALUES);
	meshSetClass.def(
		"print_filter_script", &MeshSet::printFilterScript, doc::PYMS_PRINT_FILTER_SCIRPT);

	.def("add_camera", [](pymeshlab::MeshSet& ms, const vcg::Shotf& camera) {
            ms.cm.shot = camera;
        }, "Add a camera to the current mesh", py::arg("camera"));

    .def("get_camera", [](pymeshlab::MeshSet& ms) -> vcg::Shotf {
            return ms.cm.shot;
        }, "Get the camera associated with the current mesh");

    .def("has_camera", [](pymeshlab::MeshSet& ms) {
            return ms.cm.IsCameraPresent();
        }, "Check if the current mesh has a camera");

    .def("clear_camera", [](pymeshlab::MeshSet& ms) {
            ms.cm.ClearShot();
        }, "Remove the camera from the current mesh");

    .def("add_raster_camera", [](pymeshlab::MeshSet& ms,
                                    const std::string& shot_filename,
                                    const std::string& image_filename) {
            // Add the camera to the raster layer system
            ms.addRasterCamera(shot_filename, image_filename);
        }, "Add a raster camera to the mesh set",
           py::arg("shot_filename"),
           py::arg("image_filename"));

    .def("compute_texmapping_from_camera", [](pymeshlab::MeshSet& ms,
                                                bool per_camera_uvs = true,
                                                float border = 2.0f) {
            // Compute texture mapping using the camera
            ms.computeTexMappingFromCamera(per_camera_uvs, border);
        }, "Compute texture mapping from camera parameters",
           py::arg("per_camera_uvs") = true,
           py::arg("border") = 2.0f);
}
