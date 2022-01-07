#pragma once

#include "ceres/ceres.h"
#include "rust/cxx.h"

struct DecisionVariable;
struct ParameterBlock;
struct Result;

class CostManager {
public:
  void add_cost(rust::Fn<bool(const rust::Vec<DecisionVariable> &constants,
                              double const *const *parameters,
                              double *residuals, double **jacobians)>
                    fn,
                rust::Vec<ParameterBlock> parameter_blocks, int32_t num_residuals);
  ceres::Problem ceres_problem;
};

class Problem {
public:
  Problem();

  void add_cost_terms(
      rust::Fn<void(CostManager&manager, const rust::Vec<DecisionVariable> &descision_variables)>
          fn) noexcept;

  CostManager costs;
  rust::Vec<DecisionVariable> descision_variables;
};

std::unique_ptr<Problem> new_problem_from_variables(
    rust::Vec<DecisionVariable> decision_variables) noexcept;

std::unique_ptr<Problem> new_problem() noexcept;

class SolverOptions {
public:
  SolverOptions();

  ceres::Solver::Options solver_options;
};

std::unique_ptr<SolverOptions> new_solver_options() noexcept;

class SolverSummary {
public:
  SolverSummary() {}

  void print() const noexcept;

  ceres::Solver::Summary solver_summary;
};

std::unique_ptr<Result>
solve(const std::unique_ptr<SolverOptions> &options,
      std::unique_ptr<Problem> &problem) noexcept;
