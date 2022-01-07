// src/blobstore.cc

#include "metal-carl/include/ceres_wrapper.h"
#include "metal-carl/src/main.rs.h"

Problem::Problem() {}

std::unique_ptr<Problem> new_problem() noexcept {
  return std::unique_ptr<Problem>(new Problem());
}

// namespace {

class WrapperCost : public ceres::CostFunction {
public:
  WrapperCost(rust::Vec<DecisionVariable> constants,
              rust::Fn<bool(const rust::Vec<DecisionVariable> &constants,
                            double const *const *parameters, double *residuals,
                            double **jacobians)>
                  fn,
              rust::Vec<ParameterBlock> blocks, int num_residuals)
      : constants(std::move(constants)), fn(fn) {
    this->set_num_residuals(num_residuals);
    for (const ParameterBlock &block : blocks) {
      this->mutable_parameter_block_sizes()->push_back(block.size);
    }
  }

  virtual ~WrapperCost() {}

  virtual bool Evaluate(double const *const *parameters, double *residuals,
                        double **jacobians) const {

    return fn(constants, parameters, residuals, jacobians);
  }

  const rust::Vec<DecisionVariable> constants;

  rust::Fn<bool(const rust::Vec<DecisionVariable> &constants,
                double const *const *parameters, double *residuals,
                double **jacobians)>
      fn;
};

// } // namespace

void CostManager::add_cost(
    rust::Fn<bool(const rust::Vec<DecisionVariable> &constants,
                  double const *const *parameters, double *residuals,
                  double **jacobians)>
        fn,

    rust::Vec<ParameterBlock> parameter_blocks, int num_residuals) {

  std::vector<double *> parameter_block_ptrs;
  parameter_block_ptrs.reserve(parameter_blocks.size());
  for (const auto &p : parameter_blocks) {
    parameter_block_ptrs.push_back(const_cast<double *>(p.ptr));
  }
  ceres::CostFunction *cost_function =
      new WrapperCost({}, fn, parameter_blocks, num_residuals);

  this->ceres_problem.AddResidualBlock(cost_function, NULL,
                                       parameter_block_ptrs);
}

std::unique_ptr<Problem> new_problem_from_variables(
    rust::Vec<DecisionVariable> decision_variables) noexcept {

  // std::unique_ptr<Problem> new_problem_from_vec_of_vecs(
  //     rust::Vec<rust::Vec<double>>&& decision_variable_vectors) noexcept {
  std::unique_ptr<Problem> problem = new_problem();
  problem->descision_variables = std::move(decision_variables);
  return problem;
}

void Problem::add_cost_terms(
    rust::Fn<void(CostManager &manager,
                  const rust::Vec<DecisionVariable> &descision_variables)>
        fn) noexcept {
  fn(this->costs, this->descision_variables);
}

SolverOptions::SolverOptions() {
  solver_options.minimizer_progress_to_stdout = true;
}

void SolverSummary::print() const noexcept {
  std::cout << solver_summary.FullReport() << "\n";
}

std::unique_ptr<SolverOptions> new_solver_options() noexcept {
  return std::unique_ptr<SolverOptions>(new SolverOptions());
}

std::unique_ptr<Result> solve(const std::unique_ptr<SolverOptions> &options,
                              std::unique_ptr<Problem> &problem) noexcept {
  std::unique_ptr<SolverSummary> summary =
      std::unique_ptr<SolverSummary>(new SolverSummary());
  ceres::Solve(options->solver_options, &problem->costs.ceres_problem,
               &summary->solver_summary);
  std::unique_ptr<Result> result
      (new Result);
  result->summary = std::move(summary);
  result->result = std::move(problem->descision_variables);
  return result;
}