use crate::create_env_wrapper;
use crate::env::{EnvInstance, EnvShared, Envs};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rand::prelude::*;
use regex::Regex;
use std::sync::Arc;

#[derive(Clone, FromPyObject)]
pub struct WordleSettings {
    max_guesses: usize,
    words: Vec<String>,
}

pub struct WordleShared {
    settings: WordleSettings,
    guess_re: Regex,
}

impl EnvShared for WordleShared {
    type Settings = WordleSettings;

    fn new(settings: Self::Settings) -> Self {
        Self {
            settings,
            guess_re: Regex::new(r"[a-zA-Z]{5}").unwrap(),
        }
    }
}

struct WordleInstance {
    shared: Arc<WordleShared>,
    rng: SmallRng,
    secret_word_index: usize,
    guesses: usize,
}

impl WordleInstance {
    fn generate_feedback(&self, guess: &str) -> String {
        let secret_word: &str = &self.shared.settings.words[self.secret_word_index];

        let mut feedback = ['-'; 5];

        for (i, (g, s)) in guess.chars().zip(secret_word.chars()).enumerate() {
            if g == s {
                feedback[i] = 'G';
            } else if secret_word.contains(g) {
                feedback[i] = 'Y';
            }
        }

        feedback.iter().collect()
    }
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
        self.guesses = 0;
        self.secret_word_index = self.rng.random_range(0..self.shared.settings.words.len());

        "Make your first guess now".to_string()
    }

    fn step(&mut self, action: &str) -> (String, f32, bool) {
        let secret_word: &str = &self.shared.settings.words[self.secret_word_index];
        let guess = self
            .shared
            .guess_re
            .find_iter(action)
            .last()
            .map(|m| m.as_str().to_lowercase());

        let (obs, reward, word_found) = if let Some(guess) = guess {
            if guess == secret_word {
                ("You guessed the word!".to_string(), 1.0, true)
            } else {
                (self.generate_feedback(&guess), 0.0, false)
            }
        } else {
            ("No guess was found".to_string(), 0.0, false)
        };

        self.guesses += 1;
        if word_found || self.guesses >= self.shared.settings.max_guesses {
            (self.reset(), reward, true)
        } else {
            (obs, reward, false)
        }
    }
}

// Create the wrapper for Wordle
create_env_wrapper!(
    WordleEnv,
    WordleInstance,
    WordleSettings,
    "Wordle: Guess the 5-letter word in 6 tries. Feedback: G=Green (Correct), Y=Yellow (Wrong spot), -=Grey (Not in word)."
);
