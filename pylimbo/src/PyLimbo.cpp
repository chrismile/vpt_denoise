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

#define USE_NLOPT
#include <limbo/bayes_opt/boptimizer.hpp>
#include <utility>
#include <pybind11/functional.h>

#include "PyLimbo.hpp"

PYBIND11_MODULE(pylimbo, m) {
    py::class_<BayOptSettings>(m, "BayOptSettings")
            .def(py::init<>())
            .def_readwrite("pbounds", &BayOptSettings::pbounds)
            .def_readwrite("num_iterations", &BayOptSettings::num_iterations)
            .def_readwrite("num_optimizer_iterations", &BayOptSettings::num_optimizer_iterations)
            .def_readwrite("ucb_kappa", &BayOptSettings::ucb_kappa);
    m.def("maximize", maximize,
          "Applies Bayesian optimization.",
          py::arg("settings"), py::arg("init_points"), py::arg("callback"));
}

namespace BayOpt {

using namespace limbo;

struct Params {
    struct bayes_opt_boptimizer : public defaults::bayes_opt_boptimizer {
    };
#ifdef USE_NLOPT
    struct opt_nloptnograd : public defaults::opt_nloptnograd {
        BO_DYN_PARAM(int, iterations);
    };
#elif defined(USE_LIBCMAES)
    struct opt_cmaes : public defaults::opt_cmaes {
    };
#else
    struct opt_gridsearch : public defaults::opt_gridsearch {
    };
#endif
    struct bayes_opt_bobase : public defaults::bayes_opt_bobase {
        BO_PARAM(int, stats_enabled, false);
    };
    struct kernel : public defaults::kernel {
        BO_PARAM(double, noise, 1e-10);
    };
    struct kernel_maternfivehalves : public defaults::kernel_maternfivehalves {
    };
    struct init_randomsampling {
        BO_DYN_PARAM(int, samples);
    };
    struct stop_maxiterations {
        BO_DYN_PARAM(int, iterations);
    };
    struct acqui_ucb : public defaults::acqui_ucb {
        BO_DYN_PARAM(double, alpha);
    };
};

struct Eval {
    const OptBounds pbounds;
    const std::function<float(const OptParams&)> f; //< Function to be optimized.
    mutable float bestValue = std::numeric_limits<float>::lowest();
    mutable OptParams bestParams;
    BO_DYN_PARAM(size_t, dim_in);
    BO_PARAM(size_t, dim_out, 1);

    Eigen::VectorXd operator()(const Eigen::VectorXd& v) const {
        OptParams params;
        int i = 0;
        for (const auto& bound : pbounds) {
            const auto& range = bound.second;
            params.insert(std::make_pair(bound.first, range.first + v[i] * (range.second - range.first)));
            i++;
        }
        float value = f(params);
        if (std::isnan(value)) {
            std::cerr << "Error: NaN sample detected." << std::endl;
            value = 0.0f;
        }
        if (value > bestValue) {
            bestValue = value;
            bestParams = params;
        }
        return tools::make_vector(value);
    }

    Eigen::VectorXd evalDirect(const OptParams& params) const {
        float value = f(params);
        if (std::isnan(value)) {
            std::cerr << "Error: NaN sample detected." << std::endl;
            value = 0.0f;
        }
        if (value > bestValue) {
            bestValue = value;
            bestParams = params;
        }
        return tools::make_vector(value);
    }

    Eigen::VectorXd convertSample(const OptParams& params) const {
        Eigen::VectorXd paramsVec(params.size());
        int i = 0;
        auto pboundsIt = pbounds.begin();
        auto paramsIt = params.begin();
        for (int i = 0; i < params.size(); i++) {
            const auto& range = pboundsIt->second;
            const auto& param = paramsIt->second;
            paramsVec[i] = (param - range.first) / (range.second - range.first);
            pboundsIt++;
            paramsIt++;
        }
        return paramsVec;
    }
};

BO_DECLARE_DYN_PARAM(int, BayOpt::Params::stop_maxiterations, iterations);
BO_DECLARE_DYN_PARAM(int, BayOpt::Params::init_randomsampling, samples);
BO_DECLARE_DYN_PARAM(int, BayOpt::Params::opt_nloptnograd, iterations);
BO_DECLARE_DYN_PARAM(double, BayOpt::Params::acqui_ucb, alpha);
BO_DECLARE_DYN_PARAM(size_t, BayOpt::Eval, dim_in);

}

std::pair<float, OptParams> maximize(
        BayOptSettings settings, const std::vector<OptParams>& initPoints,
        std::function<float(const OptParams&)> callback) {
    limbo::bayes_opt::BOptimizer<BayOpt::Params> optimizer;

    auto eval = BayOpt::Eval{settings.pbounds, callback};
    for (const OptParams& initParams : initPoints) {
        optimizer.add_new_sample(eval.convertSample(initParams), eval.evalDirect(initParams));
    }

    BayOpt::Params::stop_maxiterations::set_iterations(settings.num_iterations);
    BayOpt::Params::init_randomsampling::set_samples(0);
    BayOpt::Params::opt_nloptnograd::set_iterations(settings.num_optimizer_iterations);
    BayOpt::Params::acqui_ucb::set_alpha(settings.ucb_kappa);
    BayOpt::Eval::set_dim_in(settings.pbounds.size());
    optimizer.optimize(eval);
    return std::make_pair(eval.bestValue, eval.bestParams);
}
