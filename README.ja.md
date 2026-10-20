# Looped-DiT for ComfyUI（日本語概要）

[Looped-DiT](https://github.com/OpenSenseNova/Looped-DiT) を ComfyUI で使うための非公式ノードです。

- **loop depth（ループ深度）はサンプリングのステップ数ではありません。** すべて公式の Euler 100 ステップ・CFG 6（モデル計算 200 回）で生成しています。ループ深度 N は、1 回の計算の中で共有ブロックを何回繰り返すかだけを変えます（B/16 では 6 + 5×N + 6 ブロック）。
- **公式コードと同じ画素:** B/16 の EMA 重みで比較した 27 枚すべてについて、出力が公式実装とビット単位で一致しました（8bit 画像と float32 の結果）。トークン ID、FLAN-T5 の埋め込み、初期ノイズも一致しています。内訳は、公式 README の例 prompt の loops 1/2/4 と、4 prompts × 2 seeds × loops 1/2/4 です。ComfyUI のキュー経由、RTX 5090・Windows で確認しました。
- **Loop Sweep ノード:** 同じ prompt・seed・初期ノイズで loops 1/2/4 などを順番に生成します。それぞれ独立した完全なサンプリングです。比較グリッドと JSON レポートも返します。公式の `sample.py --loops 1 2 3 4` にも同じ比較機能があります。
- **追加のパッケージは不要です。** ComfyUI のプロセス内で動き、公式の model / diffusion は改変せずに、config は学習用の YAML 読み込み関数 2 つだけを無効にして同梱しています（looped_dit_comfy/vendor/UPSTREAM.md）。実行時の subprocess やダウンロードはありません。
- **tokenizer の差に対応:** transformers 5.x は末尾の空白を 4.x と違う形で分割します。そこで tokenizers ライブラリを直接使い、テスト用の 317 prompts すべてで 4.x と同じトークン ID になることを確認しました。

導入方法: `custom_nodes` に clone し、ComfyUI の Python で `tools/download_models.py --models-dir models` を実行します（取得するリビジョンを固定し、SHA-256 も照合します）。

制限: CUDA 専用です（Mac の MPS/MLX は未対応・未検証）。batch 1、512×512 のみです。メモリは Unload ノードで解放してください。ビット一致は上記の 1 つの環境で確認したもので、他の GPU や `--fast` などのオプションでは変わる可能性があります。

画像の評価は作者が目で見た所感にとどまり、ベンチマークは測っていません。上流 README のスコアは著者による測定値です。ループ深度を上げれば必ず良くなるわけではありません（例: loops 2 で提灯が 2 つ描かれるケースがありました）。

ライセンス: MIT（同梱した上流ファイルも MIT のまま）。FLAN-T5-Large は Apache-2.0 です。詳しくは NOTICE を参照してください。
