use std::borrow::Borrow;

use itertools::Itertools;
use rand::prelude::*;

pub trait EnvInstance {
    type Settings;

    fn new(seed: u64, settings: &Self::Settings) -> Self;
    fn reset(&mut self) -> String;
    fn step(&mut self, action: &str) -> (String, f32, bool);
}

pub struct Envs<E> {
    envs: Vec<E>
}

impl<E: EnvInstance> Envs<E> {
    pub fn new(seed: u64, num: usize, settings: &E::Settings) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);

        let envs = (0..num)
            .map(|_| E::new(rng.next_u64(), settings))
            .collect();

        Self { envs }
    }

    pub fn reset<'a>(&mut self, indices: impl IntoIterator<Item=&'a i32>) -> Vec<String> {
        indices
            .into_iter()
            .map(|&i| self.envs.get_mut(i as usize).unwrap().reset())
            .collect()
    }

    pub fn step<I, A>(&mut self, indices: I, actions: A) -> (Vec<String>, Vec<f32>, Vec<bool>)
    where
        I: IntoIterator,
        I::Item: Borrow<i32>,
        A: IntoIterator,
        A::Item: AsRef<str>,
    {
        indices.into_iter().zip(actions).map(|(index, action)| {
            let index = *index.borrow();
            let action = action.as_ref();

            self.envs
                    .get_mut(index as usize)
                    .unwrap()
                    .step(action)
        }).multiunzip()
    }
}


#[macro_export]
macro_rules! create_env_wrapper {
    ($py_name:ident, $rust_env:ty, $settings:expr, $instr:literal) => {
        #[pyclass]
        pub struct $py_name {
            envs: Envs<$rust_env>,
        }

        #[pymethods]
        impl $py_name {
            #[new]
            fn new(num_agents: usize) -> Self {
                Self { 
                    envs: Envs::new(0, num_agents, &$settings) 
                }
            }

            fn reset<'py>(&mut self, batch_indices: PyReadonlyArray1<'py, i32>) -> PyResult<Vec<String>> {
                let indices = batch_indices.as_array();
                Ok(self.envs.reset(indices))
            }

            fn step<'py>(
                &mut self,
                py: Python<'py>,
                batch_indices: PyReadonlyArray1<'py, i32>,
                actions: Vec<String>,
            ) -> PyResult<(Vec<String>, Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<bool>>)> {
                let indices = batch_indices.as_array();

                let (obs, rewards, dones) = self.envs.step(&indices, &actions);

                let rewards = rewards.into_pyarray(py);
                let dones = dones.into_pyarray(py);

                Ok((obs, rewards, dones))
            }

            fn instructions(&self) -> PyResult<&'static str> {
                Ok($instr)
            }
        }
    };
}