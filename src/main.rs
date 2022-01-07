// src/main.rs

#[cxx::bridge]
mod ffi {

    #[derive(Debug)]
    struct DecisionVariable {
        v: Vec<f64>,
    }

    struct ParameterBlock {
        size: i32,
        ptr: *const f64,
    }

    struct Result {
        summary: UniquePtr<SolverSummary>,
        result: Vec<DecisionVariable>,
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
        ) -> UniquePtr<Result>;

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

impl ffi::ParameterBlock {
    fn from_var(var: &ffi::DecisionVariable) -> Self {
        Self {
            size: var.v.len() as i32,
            ptr: var.v.as_ptr(),
        }
    }
}

fn main() {
    let options = ffi::new_solver_options();

    let mut problem = ffi::new_problem_from_variables(vec![ffi::DecisionVariable { v: vec![0.5] }]);

    problem.pin_mut().add_cost_terms(
        |manager: core::pin::Pin<&mut ffi::CostManager>, vars: &Vec<ffi::DecisionVariable>| {
            let parameter_blocks: Vec<ffi::ParameterBlock> =
                vec![ffi::ParameterBlock::from_var(&vars[0])];

            manager.add_cost(
                |_constants, parameters, residuals, jacobians| unsafe {
                    let x = *(*parameters.offset(0)).offset(0);

                    fn f<T: num_traits::Float>(x: T) -> T {
                        T::from(10.0).unwrap() - x
                    }

                    let fx = f(autodiff::F1::var(x));
                    *residuals.offset(0) = f(x);

                    if !jacobians.is_null() && !(*jacobians.offset(0)).is_null() {
                        *(*jacobians.offset(0)).offset(0) = fx.deriv();
                    }
                    true
                },
                parameter_blocks,
                1,
            );
        },
    );

    let result = ffi::solve(&options, &mut problem);

    result.summary.print();
    println!("{:?}", result.result);
}
