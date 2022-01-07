use ffi::DecisionVariable;

// src/main.rs

#[cxx::bridge]
mod ffi {
    struct DecisionVariable {
        v: Vec<f64>,
    }

    struct ParameterBlock {
        size: i32,
        ptr: *const f64,
    }

    unsafe extern "C++" {
        include!("metal-carl/include/ceres_wrapper.h");

        type Problem;
        fn new_problem_from_variables(
            decision_variables: Vec<DecisionVariable>,
        ) -> UniquePtr<Problem>;
        fn add_cost_terms(
            self: Pin<&mut Problem>,
            decision_variables: fn(Pin<&mut CostManager>, &Vec<DecisionVariable>),
        );

        type SolverOptions;
        fn new_solver_options() -> UniquePtr<SolverOptions>;

        fn print(self: &SolverSummary);

        type SolverSummary;

        fn solve(
            options: &UniquePtr<SolverOptions>,
            problem: &mut UniquePtr<Problem>,
        ) -> UniquePtr<SolverSummary>;

        type CostManager;

        fn add_cost(
            self: Pin<&mut CostManager>,
            f: unsafe fn(
                constants: &Vec<DecisionVariable>,
                *const *const f64,
                *mut f64,
                *mut *mut f64,
            ) -> bool,
            parameter_blocks: Vec<ParameterBlock>,
            num_residuals: i32,
        );

    }
}

fn main() {
    let options = ffi::new_solver_options();

    let mut problem = ffi::new_problem_from_variables(vec![ffi::DecisionVariable { v: vec![0.5] }]);

    problem.pin_mut().add_cost_terms(
        |manager: core::pin::Pin<&mut ffi::CostManager>, vars: &Vec<DecisionVariable>| {
            let parameter_blocks: Vec<ffi::ParameterBlock> = vec!(ffi::ParameterBlock {
                size: vars[0].v.len() as i32,
                ptr: vars[0].v.as_ptr(),
            });

            manager.add_cost(
                |_constants, parameters, residuals, jacobians| unsafe {
                    *residuals.offset(0) = 10.0 - *(*parameters.offset(0)).offset(0);
                    if !jacobians.is_null() &&  !(*jacobians.offset(0)).is_null() {
                        *(*jacobians.offset(0)).offset(0) = -1.0;
                    }
                    true
                },
                parameter_blocks,
                1,
            );
        },
    );

    let summary = ffi::solve(&options, &mut problem);

    summary.print();
}
