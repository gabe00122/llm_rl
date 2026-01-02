use crate::create_env_wrapper;
use crate::env::{EnvInstance, EnvShared, Envs};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rand::prelude::*;
use std::sync::Arc;

#[derive(Clone, FromPyObject)]
pub struct WordleSettings {
    max_guesses: usize,
    words: Vec<String>,
}

pub struct WordleShared {
    settings: WordleSettings,
}

impl EnvShared for WordleShared {
    type Settings = WordleSettings;

    fn new(settings: Self::Settings) -> Self {
        Self { settings }
    }
}

struct WordleInstance {
    shared: Arc<WordleShared>,
    rng: SmallRng,
    secret_word_index: usize,
    guesses: usize,
}

impl EnvInstance for WordleInstance {
    type Shared = WordleShared;

    fn new(seed: u64, shared: Arc<Self::Shared>) -> Self {
        let rng = SmallRng::seed_from_u64(seed);

        WordleInstance {
            shared,
            rng,
            secret_word_index: 0,
            guesses: 0,
        }
    }

    fn reset(&mut self) -> String {
        String::new()
    }

    fn step(&mut self, action: &str) -> (String, f32, bool) {
        let obs = String::new();
        let reward = 0.0;
        let done = false;

        (obs, reward, done)
    }
}

// Create the wrapper for Wordle
create_env_wrapper!(
    WordleEnv,
    WordleInstance,
    WordleSettings,
    "Wordle: Guess the 5-letter word in 6 tries. Feedback: G=Green (Correct), Y=Yellow (Wrong spot), -=Grey (Not in word)."
);
