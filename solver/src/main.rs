#[cfg(feature = "local")]
use solver::*;

use proconio::source::line::LineSource;
use std::io::BufReader;

fn main() {
    let stdin = std::io::stdin();
    let mut source = LineSource::new(BufReader::new(stdin));
    let input = Input::from_source(&mut source);
    let solution = solve(&input);
    println!("{}", solution.score);
}
