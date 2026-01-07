use numpy::{PyReadonlyArray2, PyReadwriteArray2, ndarray::Zip};
use pyo3::prelude::*;

#[pyfunction]
pub fn lambda_returns<'py>(
    rewards: PyReadonlyArray2<'py, f32>,
    values: PyReadonlyArray2<'py, f32>,
    discount: f32,
    lambda: f32,
    mut targets: PyReadwriteArray2<'py, f32>,
) {
    let rewards = rewards.as_array();
    let values = values.as_array();
    let mut targets = targets.as_array_mut();

    Zip::from(rewards.rows())
        .and(values.rows())
        .and(targets.rows_mut())
        .for_each(|reward_row, value_row, mut target_row| {
            let mut acc = *value_row.last().unwrap();
            for i in (0..reward_row.len()).rev() {
                acc = reward_row[i] + discount * ((1.0 - lambda) * value_row[i] + lambda * acc);
                target_row[i] = acc;
            }
        });
}
