use crate::create_env_wrapper;
use crate::env::{EnvInstance, EnvShared, Envs};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rand::prelude::*;
use regex::Regex;
use std::fs;
use std::sync::Arc;

#[derive(Clone, FromPyObject)]
pub struct WordleSettings {
    max_guesses: usize,
}

pub struct WordleShared {
    settings: WordleSettings,
    guess_re: Regex,
    words: Vec<String>,
}

impl EnvShared for WordleShared {
    type Settings = WordleSettings;

    fn new(settings: Self::Settings) -> Self {
        let words: Vec<String> = fs::read_to_string("./env_assets/wordle_words.txt")
            .expect("Failed to read wordle_words.txt")
            .split_whitespace()
            .map(|word| word.to_string())
            .collect();

        Self {
            settings,
            guess_re: Regex::new(r"[A-Z]{5}").unwrap(),
            words,
        }
    }
}

struct WordleInstance {
    shared: Arc<WordleShared>,
    rng: SmallRng,
    secret_word_index: usize,
    guesses: usize,
    correct: [bool; 5],
}

impl WordleInstance {
    fn generate_feedback(&mut self, guess: &str) -> (String, f32) {
        let secret_word: &str = &self.shared.words[self.secret_word_index];

        let mut reward = 0.0;
        let mut feedback = ['-'; 5];

        for (i, (g, s)) in guess.chars().zip(secret_word.chars()).enumerate() {
            if g == s {
                feedback[i] = 'G';
                if !self.correct[i] {
                    reward += 1.0 / 5.0;
                    self.correct[i] = true;
                }
            } else if secret_word.contains(g) {
                feedback[i] = 'Y';
            }
        }

        (
            format!(
                "You guessed \"{}\" - your feedback is \"{}\"",
                guess,
                feedback.iter().collect::<String>()
            ),
            reward,
        )
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
            correct: [false; 5],
        }
    }

    fn reset(&mut self) -> String {
        self.guesses = 0;
        self.secret_word_index = self.rng.random_range(0..self.shared.words.len());
        self.correct = [false; 5];

        "Make your first guess now".to_string()
    }

    fn step(&mut self, action: &str) -> (String, f32, bool) {
        let guess = self
            .shared
            .guess_re
            .find_iter(action)
            .last()
            .map(|m| m.as_str().to_uppercase());

        let (obs, reward, word_found) = if let Some(guess) = guess {
            let (feedback, reward) = self.generate_feedback(&guess);
            let word_found = guess == self.shared.words[self.secret_word_index];

            (feedback, reward, word_found)
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
    "Your job is to play wordle: Guess the 5-letter word in 6 tries. Feedback: G=Green (Correct), Y=Yellow (Wrong spot), -=Grey (Not in word). Your guess must be in all caps and the last word in your response."
);
