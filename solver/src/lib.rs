use proconio::{input, source::Source};
use std::io::BufRead;

#[derive(Clone, Debug)]
pub struct Input {
    pub n: usize,
}

impl Input {
    pub fn from_source<R: BufRead, S: Source<R>>(mut source: &mut S) -> Self {
        input! {
            from &mut source,
            n: usize,
        }
        Self { n }
    }
}

pub struct Output {
    pub output: String,
    pub score: usize,
}

pub fn solve(input: &Input) -> Output {
    Output {
        output: format!("{} output examples output example", input.n),
        score: input.n + 1,
    }
}
