use pyo3::prelude::*;

pub enum Operator {
    Add,
    Sub,
    Mul,
    Div,
}

#[pyclass]
pub struct ArithmaticEnv {
    op: Operator,
    x: i32,
    y: i32,
}

#[pymethods]
impl ArithmaticEnv {
    #[new]
    fn new() -> Self {
        ArithmaticEnv {
            op: Operator::Add,
            x: 0,
            y: 0,
        }
    }

    fn reset(&self) -> PyResult<String> {
        Ok(String::from("Reset Message"))
    }
}
