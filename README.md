# HeuristicSolverTemplate

```shell
cargo lambda build --release --arm64
```

```shell
# cargo lambda deploy ahc
cargo lambda deploy ahc -l /target/lambda/ -r ap-northeast-1 
```

```shell
cargo run --bin solver < ./tools/in/0002.txt
```

## メモ
 - 正の得点を出す
    - seed 0: 1030664,
    - 提出: 309,165,227

 - 全ての辺を利用しているので、最小全域木を使うことにする
   プリム法で使っている辺を復元するにはどうしたらいい？→各ノードで一番近いedgeを覚えておけばいい。
   - seed 0: 1038630,
   - 提出: 311,598,105

   あまり効果が見られない。次は出力に変化を加える。

 - 一番近い放送局から電波を放送することにして、電力を抑える。
   - seed 0: 1348506,
   - 提出: 411,516,112
 
 - 既に他の放送局でカバーされているのであれば、出力しない
   - seed 0: 1422158
   - 提出: 431,347,949
   - 100件: 143,540,711

 - 無駄な放送局は不要であれば、接続しないようにする。
   接続しなくても良いだけで、経由して接続した方が距離が短くなるのであればそちらの方がいい
   - seed 0: 1462381
   - 提出: 442,663,036
   - 100件: 147,315,097

 - 既にほかの放送局に覆われている場合はスキップする。
　　ほぼ変わらず。
   - seed 0: 1462381
   - 提出: 442,690,698
   - 100件: 147,326,748

 - 放送局の力の大きさをコントロールする
   他の放送局の大きさを少し大きくして消せる放送局は消していく
   辺については考えていない
   - seed 0: 1495669
   - 提出: 456,839,197
   - 100件: 151,948,921

 - 上は一周しかしていないので何度も繰り返す
   - seed 0: 1499533
   - 提出: 458,154,144
   - 100件: 152,421,956

 - 消せる放送局をチェックするときに、辺の重みも再計算。遅くはなったけど、点数は向上。
   - seed 0: 1544914
   - 提出: 474,039,364
   - 100件: 157,270,870

---
 - 複数の放送局を一つに統合する(辺が節約できる)