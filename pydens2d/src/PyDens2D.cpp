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

#include <algorithm>
#include <iostream>
#include <cmath>

#include "PyDens2D.hpp"

PYBIND11_MODULE(pydens2d, m) {
    m.def("update_visibility_field", updateVisibilityField,
          "Updates the visibility field.",
          py::arg("density_field"), py::arg("visibility_field"), py::arg("aabb"),
          py::arg("cam_res"), py::arg("cam_pos"), py::arg("theta"), py::arg("fov"));
    m.def("compute_energy", computeEnergy,
          "Calculates the camera view combination energy term field.",
          py::arg("num_iterations"), py::arg("gamma"), py::arg("non_empty_voxel_pos_field"),
          py::arg("obs_freq_field"), py::arg("angular_obs_freq_field"));
    m.def("update_observation_frequency_fields", updateObservationFrequencyFields,
          "Updates the observation frequency fields.",
          py::arg("density_field"), py::arg("obs_freq_field"), py::arg("angular_obs_freq_field"), py::arg("aabb"),
          py::arg("cam_res"), py::arg("cam_pos"), py::arg("theta"), py::arg("fov"));
}

bool rayBoxIntersect(Eigen::VectorXf& bMin, Eigen::VectorXf& bMax, Eigen::VectorXf& P, Eigen::VectorXf& D, float& tMin, float& tMax) {
    D(0) = std::abs(D(0)) <= 0.000001f ? 0.000001f : D(0);
    D(2) = std::abs(D(2)) <= 0.000001f ? 0.000001f : D(2);
    //auto C_Min = (bMin - P) / D;
    //auto C_Max = (bMax - P) / D;
    Eigen::MatrixXf C_Min = bMin - P;
    Eigen::MatrixXf C_Max = bMax - P;
    C_Min(0) = C_Min(0) / D(0);
    C_Min(1) = C_Min(1) / D(1);
    C_Max(0) = C_Max(0) / D(0);
    C_Max(1) = C_Max(1) / D(1);
    tMin = std::max(std::min(C_Min(0), C_Max(0)), std::min(C_Min(1), C_Max(1)));
    tMin = std::max(0.0f, tMin);
    tMax = std::min(std::max(C_Min(0), C_Max(0)), std::max(C_Min(1), C_Max(1)));
    if (tMax <= tMin or tMax <= 0) {
        return false;
    }
    return true;
}

void updateVisibilityField(
        py::EigenDRef<Eigen::MatrixXf> densityField, py::EigenDRef<Eigen::MatrixXf> visibilityField,
        py::EigenDRef<Eigen::VectorXf> aabb,
        int camRes, Eigen::Ref<Eigen::VectorXf> camPos, float theta, float fov) {
    int res = densityField.cols();
    Eigen::VectorXf bMin(2); bMin << aabb(0), aabb(2);
    Eigen::VectorXf bMax(2); bMax << aabb(1), aabb(3);
    float cosTheta = std::cos(theta);
    float sinTheta = std::sin(theta);
    //Eigen::MatrixXf front = { cosTheta, sinTheta };
    //Eigen::MatrixXf up = { -sinTheta, cosTheta };
    Eigen::VectorXf front(2); front << cosTheta, sinTheta;
    Eigen::VectorXf up(2); up << (-sinTheta), cosTheta;
    float dist_up = 2.0f * std::tan(fov * 0.5f);
    Eigen::VectorXf pt0 = front - dist_up * up;
    Eigen::VectorXf pt1 = front + dist_up * up;
    float step_size = 0.0001f;
    #pragma omp parallel for
    for (int i = 0; i < camRes; i++) {
        float t = (float(i) + 0.5f) / float(camRes);
        auto pt = (1.0f - t) * pt0 + t * pt1;
        auto dir = pt.normalized();
        Eigen::VectorXf p = camPos;
        float tMin, tMax;
        bool intersects = rayBoxIntersect(bMin, bMax, p, dir, tMin, tMax);
        if (intersects) {
            p = p + (tMin + 1e-7f) * dir;
            while (true) {
                // Check if in AABB:
                //bool is_in_aabb = aabb(0) <= p(0) <= aabb(1) and aabb(2) <= p(1) <= aabb(3);
                bool is_in_aabb = aabb(0) <= p(0) && p(0) <= aabb(1) && aabb(2) <= p(1) && p(1) <= aabb(3);
                if (!is_in_aabb) {
                    break;
                }
                auto xi = int((p(0) - aabb(0)) / (aabb(1) - aabb(0)) * float(res - 1));
                auto yi = int((p(1) - aabb(2)) / (aabb(3) - aabb(2)) * float(res - 1));
                visibilityField(yi, xi) = 1.0f;
                if (densityField(yi, xi) > 0.0f) {
                    break;
                }
                p += step_size * dir;
            }
        }
    }
}

float computeEnergy(
        int N, float gamma,
        py::EigenDRef<Eigen::MatrixXi> nonEmptyVoxelPosField,
        py::array_t<int> obsFreqField,
        py::array_t<int> angularObsFreqField) {
    auto obsFreqFieldAccessor = obsFreqField.unchecked<2>();
    auto angularObsFreqFieldAccessor = angularObsFreqField.unchecked<3>();
    int numNonEmptyVoxels = nonEmptyVoxelPosField.rows(); // TODO: Correct???
    int angularBinSize = angularObsFreqField.shape(2);
    float fN = 1.0f / float(N);
    float fB = 1.0f / float(angularBinSize);

    float energyObsFreq = 0.0f, energyAngularObsFreq = 0.0f;
    for (int voxelIdx = 0; voxelIdx < numNonEmptyVoxels; voxelIdx++) {
        int xi = nonEmptyVoxelPosField(voxelIdx, 0);
        int yi = nonEmptyVoxelPosField(voxelIdx, 1);
        energyObsFreq += std::pow(float(obsFreqFieldAccessor(yi, xi)) * fN, gamma);
        // Use total variation distance.
        float TV = 0.0f;
        for (int bi = 0; bi < angularBinSize; bi++) {
            TV += std::abs(float(angularObsFreqFieldAccessor(yi, xi, bi)) * fN - fB);
        }
        TV *= 0.5f;
        energyAngularObsFreq += 1.0f - TV;
    }
    return energyObsFreq + energyAngularObsFreq;
}

void updateObservationFrequencyFields(
        py::EigenDRef<Eigen::MatrixXf> densityField,
        py::array_t<int> obsFreqField,
        py::array_t<int> angularObsFreqField,
        py::EigenDRef<Eigen::VectorXf> aabb,
        int camRes, Eigen::Ref<Eigen::VectorXf> camPos, float theta, float fov) {
    auto obsFreqFieldAccessor = obsFreqField.mutable_unchecked<2>();
    auto angularObsFreqFieldAccessor = angularObsFreqField.mutable_unchecked<3>();
    int res = densityField.cols();
    std::vector<float> transmittanceField(res * res, 0.0f);
    Eigen::VectorXf bMin(2); bMin << aabb(0), aabb(2);
    Eigen::VectorXf bMax(2); bMax << aabb(1), aabb(3);
    float cosTheta = std::cos(theta);
    float sinTheta = std::sin(theta);
    //Eigen::MatrixXf front = { cosTheta, sinTheta };
    //Eigen::MatrixXf up = { -sinTheta, cosTheta };
    Eigen::VectorXf front(2); front << cosTheta, sinTheta;
    Eigen::VectorXf up(2); up << (-sinTheta), cosTheta;
    float dist_up = 2.0f * std::tan(fov * 0.5f);
    Eigen::VectorXf pt0 = front - dist_up * up;
    Eigen::VectorXf pt1 = front + dist_up * up;
    float step_size = 0.0001f;
    #pragma omp parallel for
    for (int i = 0; i < camRes; i++) {
        float t = (float(i) + 0.5f) / float(camRes);
        auto pt = (1.0f - t) * pt0 + t * pt1;
        auto dir = pt.normalized();
        Eigen::VectorXf p = camPos;
        float tMin, tMax;
        bool intersects = rayBoxIntersect(bMin, bMax, p, dir, tMin, tMax);
        if (intersects) {
            p = p + (tMin + 1e-7f) * dir;
            while (true) {
                // Check if in AABB:
                //bool is_in_aabb = aabb(0) <= p(0) <= aabb(1) and aabb(2) <= p(1) <= aabb(3);
                bool is_in_aabb = aabb(0) <= p(0) && p(0) <= aabb(1) && aabb(2) <= p(1) && p(1) <= aabb(3);
                if (!is_in_aabb) {
                    break;
                }
                auto xi = int((p(0) - aabb(0)) / (aabb(1) - aabb(0)) * float(res - 1));
                auto yi = int((p(1) - aabb(2)) / (aabb(3) - aabb(2)) * float(res - 1));
                transmittanceField.at(yi * res + xi) = 1.0f;
                if (densityField(yi, xi) > 0.0f) {
                    break;
                }
                p += step_size * dir;
            }
        }
    }

    const float TWO_PI = 2.0f * M_PI;
    int angularBinSize = angularObsFreqFieldAccessor.shape(2);
    for (int yi = 0; yi < res; yi++) {
        for (int xi = 0; xi < res; xi++) {
            if (transmittanceField.at(yi * res + xi) > 1e-5f) {
                float px = (float(xi) + 0.5f) / float(res - 1) * (aabb(1) - aabb(0)) + aabb(0);
                float py = (float(xi) + 0.5f) / float(res - 1) * (aabb(3) - aabb(2)) + aabb(2);
                float dx = px - camPos(0);
                float dy = py - camPos(1);
                float fract = std::fmod(std::atan2(dy, dx) + TWO_PI, TWO_PI) / TWO_PI;
                int bi = int(fract * angularBinSize);
                obsFreqFieldAccessor(yi, xi) += 1;
                angularObsFreqFieldAccessor(yi, xi, bi) += 1;
           }
        }
    }
}
