use pyo3::prelude::*;
use rand::prelude::*;
use crate::env::EnvInstance;

struct WordleSettings {}

struct WordleInstance {

}

impl EnvInstance for WordleInstance {
    type Settings = WordleSettings;

    fn new(seed: u64, settings: &Self::Settings) -> Self {
        WordleInstance {  }
    }

    fn reset(&mut self) -> String {
        String::new()
    }

    fn step(&mut self, action: &str) -> (String, f32, bool) {
        (String::new(), 0.0, false)
    }
}

#[pyclass]
pub struct WordleEnv {

}

#[pymethods]
impl WordleEnv {
    #[new]
    fn new() -> Self {
        WordleEnv {  }
    }
}
