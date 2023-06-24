# HeuristicSolverTemplate

```shell
cargo lambda build --release --arm64
```

```shell
cargo lambda deploy lambda-template
```

## メモ
 - 正の得点を出す
    seed 0: 1030664,
    提出: 309165227

 - 全ての辺を利用しているので、最小全域木を使うことにする
   プリム法で使っている辺を復元するにはどうしたらいい？→各ノードで一番近いedgeを覚えておけばいい。
   seed 0: 1038630,
   提出: 311598105

   あまり効果が見られない。次は出力に変化を加える。