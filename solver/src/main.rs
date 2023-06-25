#[cfg(feature = "local")]
use solver::*;

use proconio::source::line::LineSource;
use std::io::BufReader;

#[cfg(feature = "local")]
fn main() {
    let stdin = std::io::stdin();
    let mut source = LineSource::new(BufReader::new(stdin));
    let input = Input::from_source(&mut source);
    let solution = solve(&input);
    println!("{}", solution.output);
    println!("{}", solution.score);
}

#[cfg(not(feature = "local"))]
fn main() {
    let stdin = std::io::stdin();
    let mut source = LineSource::new(BufReader::new(stdin));
    let input = Input::from_source(&mut source);
    let solution = solve(&input);
    println!("{}", solution.output);
}
