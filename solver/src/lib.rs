use proconio::{input, marker::Usize1, source::Source};
use std::{
    collections::{BinaryHeap, HashMap},
    io::BufRead,
};
use union_find::UnionFind;

pub mod union_find {

    #[derive(Debug, Clone)]
    pub struct UnionFind {
        par: Vec<usize>,
        size: Vec<usize>,
    }

    impl UnionFind {
        pub fn new(n: usize) -> Self {
            Self {
                par: vec![0; n],
                size: vec![1; n],
            }
        }

        pub fn find_root(&mut self, a: usize) -> usize {
            if self.size[a] > 0 {
                return a;
            }
            self.par[a] = self.find_root(self.par[a]);
            self.par[a]
        }

        pub fn union(&mut self, a: usize, b: usize) -> usize {
            let mut x = self.find_root(a);
            let mut y = self.find_root(b);
            if x == y {
                return x;
            }
            if self.size[x] < self.size[y] {
                std::mem::swap(&mut x, &mut y);
            }
            self.size[x] += self.size[y];
            self.size[y] = 0;
            self.par[y] = x;
            x
        }

        pub fn in_same_set(&mut self, a: usize, b: usize) -> bool {
            self.find_root(a) == self.find_root(b)
        }

        pub fn group_size(&mut self, a: usize) -> usize {
            let x = self.find_root(a);
            self.size[x]
        }
    }
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

impl Point {
    pub fn dist(&self, other: &Self) -> i32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }
}

#[derive(Clone, Debug)]
pub struct Input {
    pub n: usize,
    pub m: usize,
    pub k: usize,
    pub stations: Vec<Point>,
    pub u: Vec<usize>,
    pub v: Vec<usize>,
    pub w: Vec<usize>,
    pub residents: Vec<Point>,
}

impl Input {
    pub fn from_source<R: BufRead, S: Source<R>>(mut source: &mut S) -> Self {
        input! {
            from &mut source,
            n: usize,
            m: usize,
            k: usize,
            xy: [(i32, i32); n],
            uvw: [(Usize1, Usize1, usize); m],
            ab: [(i32, i32); k],
        }
        let mut x = vec![0; n];
        let mut y = vec![0; n];
        for i in 0..n {
            let (xi, yi) = xy[i];
            x[i] = xi;
            y[i] = yi;
        }
        let stations = xy.iter().map(|&(x, y)| Point { x, y }).collect();

        let mut u = vec![0; m];
        let mut v = vec![0; m];
        let mut w = vec![0; m];
        for i in 0..m {
            let (ui, vi, wi) = uvw[i];
            u[i] = ui;
            v[i] = vi;
            w[i] = wi;
        }

        let residents = ab.iter().map(|&(x, y)| Point { x, y }).collect();

        Self {
            n,
            m,
            k,
            stations,
            u,
            v,
            w,
            residents,
        }
    }

    pub fn edge_hash(&self) -> HashMap<(usize, usize), usize> {
        let mut hs = HashMap::new();

        for i in 0..self.m {
            let u = self.u[i];
            let v = self.v[i];
            hs.insert((u, v), i);
            hs.insert((v, u), i);
        }
        hs
    }

    pub fn graph(&self) -> Vec<Vec<usize>> {
        let n = self.n;
        let mut cost = vec![vec![std::usize::MAX / 4; n]; n];
        // 使う辺を覚えておきたい

        for i in 0..self.m {
            let u = self.u[i];
            let v = self.v[i];
            let w = self.w[i];
            cost[u][v] = w;
            cost[v][u] = w;
        }
        cost
    }
}

pub struct Solution {
    pub powers: Vec<usize>,
    pub edges: Vec<usize>,
}

impl Solution {
    // stationが繋がっているかどうかを判定する
    fn connected_nodes(&self, input: &Input) -> Vec<bool> {
        let mut uf = UnionFind::new(input.n);
        for i in 0..input.m {
            if self.edges[i] == 1 {
                uf.union(input.u[i], input.v[i]);
            }
        }
        let mut connected = vec![false; input.k];
        for i in 0..input.k {
            for j in 0..input.n {
                if !uf.in_same_set(0, j) {
                    // 繋がっていないのでスキップ
                    continue;
                }
                let pow = self.powers[j];
                if input.residents[i].dist(&input.stations[j]) <= (pow * pow) as i32 {
                    connected[i] = true;
                    break;
                }
            }
        }
        connected
    }

    /// スコアを計算する
    fn evaluate_score(&self, input: &Input) -> usize {
        let connected = self.connected_nodes(input);
        let count = connected.iter().filter(|&x| *x).count();
        if count < input.k {
            (1e6 * ((count + 1) as f64 / input.k as f64)) as usize
        } else {
            let psq = self.powers.iter().map(|x| x * x).sum::<usize>();
            let wsum = input
                .w
                .iter()
                .zip(&self.edges)
                .map(|(w, b)| w * b)
                .sum::<usize>();
            // println!("{} {}", psq, wsum);
            let s = psq + wsum;
            (1e6 * (1.0 + 1e8 / (s as f64 + 1e7))) as usize
        }
    }
}

impl std::fmt::Display for Solution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ps = self
            .powers
            .iter()
            .map(usize::to_string)
            .collect::<Vec<String>>()
            .join(" ");
        let bs = self
            .edges
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

// プリム法で最小全域木を求める
pub fn prim(n: usize, cost: &Vec<Vec<usize>>) -> Vec<(usize, usize)> {
    let mut used = vec![false; n];
    let mut min_cost = vec![std::usize::MAX; n]; // コスト
    let mut min_edge = vec![std::usize::MAX; n]; // 一番近いノード

    min_cost[0] = 0;
    let mut prev_v = 0;

    let mut edge_nodes = Vec::new();
    loop {
        let mut v = std::usize::MAX;
        // 属さない辺のうち、コスト最小になるものを探す
        for i in 0..n {
            if !used[i] && (v == std::usize::MAX || min_cost[i] < min_cost[v]) {
                v = i;
            }
        }

        if v == std::usize::MAX {
            // 見つからなかった
            break;
        }

        // 使った辺であることを記録
        if prev_v != v {
            edge_nodes.push((v, min_edge[v]));
        }

        prev_v = v;
        used[v] = true;
        for j in 0..n {
            if cost[v][j] < min_cost[j] {
                min_cost[j] = cost[v][j];
                min_edge[j] = v;
            }
        }
    }
    edge_nodes
}

pub fn broadcast_from_nearest_station(input: &Input) -> Vec<usize> {
    // 最も近い放送局から電波を流してもらう
    // 最初に、各住民に対し、最も近い放送局と、放送局までの距離を調べる
    // その後、最も放送局から遠い場所にいる住民から放送を開始する

    let mut min_edge = vec![0; input.k];
    let mut dist_pair = vec![vec![0; input.n]; input.k];
    let mut heap = BinaryHeap::new();

    for i in 0..input.k {
        let mut min_dist = std::i32::MAX;
        for j in 0..input.n {
            let dist = input.residents[i].dist(&input.stations[j]);
            dist_pair[i][j] = dist; // 距離を覚えておく
            if dist < min_dist {
                min_dist = dist;
                min_edge[i] = j;
            }
        }
        heap.push((min_dist, i));
    }

    let mut powers = vec![0; input.n];
    let mut used = vec![false; input.k];
    // 距離が遠いものから解決していく
    while let Some((dist, i)) = heap.pop() {
        if used[i] {
            continue;
        }
        // i から出力してもらう
        let p = (dist as f64).sqrt().ceil() as usize;
        let edge = min_edge[i];
        powers[edge] = powers[edge].max(p);
        // 既に放送ずみの住民を取り除く
        // パフォーマンスはどうか？
        for u in 0..input.k {
            if dist_pair[u][edge] <= dist {
                used[u] = true;
            }
        }
    }

    powers
}

// ワーシャルフロイド法。経路復元したい
pub fn floyd_warshall(graph: &Vec<Vec<usize>>) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let n = graph.len();

    let mut dist = graph.clone();
    let mut next = vec![vec![0; n]; n];
    for i in 0..n {
        // 自分自身の距離は0
        dist[i][i] = 0;
    }
    for i in 0..n {
        for j in 0..n {
            next[i][j] = j;
        }
    }

    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                let d = dist[i][k] + dist[k][j];
                if d < dist[i][j] {
                    dist[i][j] = d;
                    next[i][j] = next[i][k];
                }
            }
        }
    }

    (dist, next)
}

pub fn solve(input: &Input) -> Output {
    // グラフ
    let graph = input.graph();

    // ワーシャルフロイド法で二点間最短経路を求める
    let (w_dist, w_next) = floyd_warshall(&graph);

    // 辺と片の番号の対応
    let edge_hash = input.edge_hash();

    // 一番近い放送局に中継してもらう
    let powers = broadcast_from_nearest_station(input);

    // 使っているノードを集める
    // 0は使うことにする
    let used_nodes = std::iter::once(0)
        .chain(
            powers
                .iter()
                .enumerate()
                .filter(|&(i, v)| i != 0 && *v > 0) // i == 0 は必ず必要なので排除
                .map(|(i, _)| i),
        )
        .collect::<Vec<_>>();

    // 使っているノードだけからグラフを構成する
    let new_n = used_nodes.len();
    let mut new_graph = vec![vec![0; new_n]; new_n];
    for i in 0..new_n {
        for j in 0..new_n {
            let ii = used_nodes[i];
            let jj = used_nodes[j];
            new_graph[i][j] = w_dist[ii][jj];
            new_graph[j][i] = w_dist[ii][jj];
        }
    }

    // prim 法で最小全域木を求める
    let edge_nodes = prim(new_n, &new_graph);

    // 経路復元をする
    let mut edges = vec![0; input.m];
    for (u, v) in edge_nodes {
        // 再構成した辺を、元の辺に戻す
        let uu = used_nodes[u];
        let vv = used_nodes[v];

        let mut c = uu;
        while c != vv {
            // 次は？
            let next_c = w_next[c][vv];
            let idx = edge_hash.get(&(c, next_c)).unwrap();
            edges[*idx] = 1;
            c = next_c;
        }
    }

    let sol = Solution { powers, edges };
    Output {
        output: format!("{}", sol),
        score: sol.evaluate_score(input),
    }
}
