use proconio::{input, marker::Usize1, source::Source};
use std::io::BufRead;

#[derive(Clone, Debug)]
pub struct Input {
    pub n: usize,
    pub m: usize,
    pub k: usize,
    pub x: Vec<i64>,
    pub y: Vec<i64>,
    pub u: Vec<usize>,
    pub v: Vec<usize>,
    pub w: Vec<usize>,
    pub a: Vec<i64>,
    pub b: Vec<i64>,
}

impl Input {
    pub fn from_source<R: BufRead, S: Source<R>>(mut source: &mut S) -> Self {
        input! {
            from &mut source,
            n: usize,
            m: usize,
            k: usize,
            xy: [(i64, i64); n],
            uvw: [(Usize1, Usize1, usize); m],
            ab: [(i64, i64); k],
        }
        let mut x = vec![0; n];
        let mut y = vec![0; n];
        for i in 0..n {
            let (xi, yi) = xy[i];
            x[i] = xi;
            y[i] = yi;
        }

        let mut u = vec![0; m];
        let mut v = vec![0; m];
        let mut w = vec![0; m];
        for i in 0..m {
            let (ui, vi, wi) = uvw[i];
            u[i] = ui;
            v[i] = vi;
            w[i] = wi;
        }
        let mut a = vec![0; k];
        let mut b = vec![0; k];
        for i in 0..k {
            let (ai, bi) = ab[i];
            a[i] = ai;
            b[i] = bi;
        }

        Self {
            n,
            m,
            k,
            x,
            y,
            u,
            v,
            w,
            a,
            b,
        }
    }
}

pub struct Solution {
    pub p: Vec<usize>,
    pub b: Vec<usize>,
}

impl std::fmt::Display for Solution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ps = self
            .p
            .iter()
            .map(usize::to_string)
            .collect::<Vec<String>>()
            .join(" ");
        let bs = self
            .b
            .iter()
            .map(usize::to_string)
            .collect::<Vec<String>>()
            .join(" ");
        writeln!(f, "{}", ps)?;
        writeln!(f, "{}", bs)
    }
}

pub struct Output {
    pub output: String,
    pub score: usize,
}

pub fn solve(input: &Input) -> Output {
    let p = vec![5000; input.n];
    let b = vec![1; input.m];
    let sol = Solution { p, b };
    Output {
        output: format!("{}", sol),
        score: input.n + 1,
    }
}
