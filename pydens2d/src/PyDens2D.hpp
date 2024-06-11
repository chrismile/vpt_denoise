/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2023, Christoph Neuhauser
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef PYCORIANDER_PYCORIANDER_HPP
#define PYCORIANDER_PYCORIANDER_HPP

#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

void updateVisibilityField(
        py::EigenDRef<Eigen::MatrixXf> densityField, py::EigenDRef<Eigen::MatrixXf> visibilityField,
        py::EigenDRef<Eigen::VectorXf> aabb,
        int camRes, Eigen::Ref<Eigen::VectorXf> camPos, float theta, float fov);

float computeEnergy(
        int N, float gamma,
        py::EigenDRef<Eigen::MatrixXi> nonEmptyVoxelPosField,
        py::array_t<int> obsFreqField,
        py::array_t<int> angularObsFreqField);

void updateObservationFrequencyFields(
        py::EigenDRef<Eigen::MatrixXf> densityField,
        py::array_t<int> obsFreqField,
        py::array_t<int> angularObsFreqField,
        py::EigenDRef<Eigen::VectorXf> aabb,
        int camRes, Eigen::Ref<Eigen::VectorXf> camPos, float theta, float fov);

#endif //PYCORIANDER_PYCORIANDER_HPP
