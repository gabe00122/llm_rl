extern crate rand;

use std::iter::zip;

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
    op: Operator,
    x: f32,
    y: f32,
    result: f32,
}

#[pyclass]
pub struct ArithmeticEnv {
    number_re: Regex,
    rng: SmallRng,
    envs: Vec<ArithmeticEnvInstance>
}

fn sample_op(rng: &mut impl Rng) -> Operator {
    let id: u8 = rng.sample(Uniform::new(0, 4).unwrap());
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
    fn new() -> Self {
        ArithmeticEnvInstance {
            op: Operator::Add,
            x: 0.0,
            y: 0.0,
            result: 0.0,
        }
    }

    fn reset(&mut self, rng: &mut impl Rng) -> String {
        let dist = Uniform::new(0.0, 10000.0).unwrap();
        let x: f32 = rng.sample::<f32, _>(dist).round();
        let y: f32 = rng.sample::<f32, _>(dist).round();
        let op = sample_op(rng);

        self.x = x;
        self.y = y;
        self.op = op;
        self.result = calc(op, x, y);

        let op_prompt = op_str(op);
        let prompt = format!("{x} {op_prompt} {y} = ...");

        prompt
    }

    fn step(&self, action: &str, number_regex: &Regex) -> (String, f32) {
        let parsed = parse_response(number_regex, action);

        let corrected = if let Some(p) = parsed {
            (p - self.result).abs() < 0.001
        } else {
            false
        };
        let reward = if corrected { 1.0 } else { 0.0 };

        (String::new(), reward)
    }
}

#[pymethods]
impl ArithmeticEnv {
    #[new]
    fn new(num_agents: i32) -> Self {
        let envs = (0..num_agents)
            .map(|_| ArithmeticEnvInstance::new())
            .collect();

        ArithmeticEnv {
            number_re: Regex::new(r"[\d,]+(?:\.\d*)?").unwrap(),
            rng: SmallRng::seed_from_u64(1),
            envs,
        }
    }

    fn reset(&mut self) -> PyResult<Vec<String>> {
        let result = self.envs
            .iter_mut()
            .map(|env| env.reset(&mut self.rng))
            .collect();

        Ok(result)
    }

    fn step(&self, actions: Vec<String>) -> PyResult<(Vec<String>, Vec<f32>)> {
        let result = zip(&self.envs, &actions)
            .map(|(env, action)| env.step(action, &self.number_re))
            .unzip();

        Ok(result)
    }

    fn instructions(&self) -> PyResult<&'static str> {
        Ok(
            "Solve the arithmetic expression using +, -, * or /. Show your work if needed, but end with only the numeric result on its own line. Always output with decimals such as 123.456",
        )
    }
}
