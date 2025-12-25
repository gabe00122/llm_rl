extern crate rand;

use itertools::MultiUnzip;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rand::{distr::Uniform, prelude::*};
use regex::Regex;

#[derive(Clone, Copy)]
pub enum Operator {
    Add,
    Sub,
    Mul,
    Div,
}

struct ArithmeticEnvInstance {
    number_re: Regex,
    rng: SmallRng,
    op: Operator,
    x: f32,
    y: f32,
    result: f32,
}

#[pyclass]
pub struct ArithmeticEnv {
    envs: Vec<ArithmeticEnvInstance>,
}

fn sample_op(rng: &mut impl Rng) -> Operator {
    let id: u8 = rng.sample(Uniform::new(0, 2).unwrap());
    match id {
        0 => Operator::Add,
        1 => Operator::Sub,
        2 => Operator::Mul,
        _ => Operator::Div,
    }
}

fn op_str(op: Operator) -> &'static str {
    match op {
        Operator::Add => "+",
        Operator::Sub => "-",
        Operator::Mul => "*",
        Operator::Div => "/",
    }
}

fn calc(op: Operator, x: f32, y: f32) -> f32 {
    match op {
        Operator::Add => x + y,
        Operator::Sub => x - y,
        Operator::Mul => x * y,
        Operator::Div => x / y,
    }
}

fn parse_response(re: &Regex, text: &str) -> Option<f32> {
    if let Some(last_match) = re.captures_iter(text).last() {
        let clean_match = last_match[0].replace(",", "");
        clean_match.parse().ok()
    } else {
        None
    }
}

impl ArithmeticEnvInstance {
    fn new(seed: u64) -> Self {
        ArithmeticEnvInstance {
            number_re: Regex::new(r"[\d,]+(?:\.\d*)?").unwrap(),
            rng: SmallRng::seed_from_u64(seed),
            op: Operator::Add,
            x: 0.0,
            y: 0.0,
            result: 0.0,
        }
    }

    fn reset(&mut self) -> String {
        let dist = Uniform::new(0.0, 10000.0).unwrap();
        let x: f32 = self.rng.sample::<f32, _>(dist).round();
        let y: f32 = self.rng.sample::<f32, _>(dist).round();
        let op = sample_op(&mut self.rng);

        self.x = x;
        self.y = y;
        self.op = op;
        self.result = calc(op, x, y);

        let op_prompt = op_str(op);
        let prompt = format!("{x} {op_prompt} {y} = ...");

        prompt
    }

    fn step(&mut self, action: &str) -> (String, f32, bool) {
        let parsed = parse_response(&self.number_re, action);

        let corrected = if let Some(p) = parsed {
            (p - self.result).abs() < 0.001
        } else {
            false
        };
        let reward = if corrected { 1.0 } else { 0.0 };
        let done = true;

        (self.reset(), reward, done)
    }
}

#[pymethods]
impl ArithmeticEnv {
    #[new]
    fn new(num_agents: i32) -> Self {
        let mut rng = SmallRng::seed_from_u64(1);

        let envs = (0..num_agents)
            .map(|_| ArithmeticEnvInstance::new(rng.next_u64()))
            .collect();

        ArithmeticEnv {
            envs,
        }
    }

    fn reset<'py>(&mut self, batch_indices: PyReadonlyArray1<'py, i32>) -> PyResult<Vec<String>> {
        let indices = batch_indices.as_array();
        
        let result = indices
            .iter()
            .map(|&i| self.envs.get_mut(i as usize).unwrap().reset())
            .collect();

        Ok(result)
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        batch_indices: PyReadonlyArray1<'py, i32>,
        actions: Vec<String>,
    ) -> PyResult<(Vec<String>, Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<bool>>)> {
        let indices = batch_indices.as_array();

        let (obs, rewards, dones): (Vec<String>, Vec<f32>, Vec<bool>) = indices
            .iter()
            .zip(actions.iter())
            .map(|(&i, action)| {
                self.envs
                    .get_mut(i as usize)
                    .unwrap()
                    .step(action)
            })
            .multiunzip();
        
        let rewards: Bound<'py, PyArray1<f32>> = rewards.into_pyarray(py);
        let dones: Bound<'py, PyArray1<bool>> = dones.into_pyarray(py);

        Ok((obs, rewards, dones))
    }

    fn instructions(&self) -> PyResult<&'static str> {
        Ok(
            "Solve the arithmetic expression using +, -, * or /. Show your work if needed, but end with only the numeric result on its own line. Always output with decimals such as 123.456",
        )
    }
}
