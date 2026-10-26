Implementation of the original transformer in Attention Is All You Need paper

<!-- BENCHMARK_START -->
## Model Benchmark Results

| Model                       | Positional Emb            | Attention Type     | Norm    |   Parameters |   Train Loss |   Best Val Loss |   Final Val Loss |   Final PPL |   Steps/sec | Completion                                                                                                                                                                                              |
|:----------------------------|:--------------------------|:-------------------|:--------|-------------:|-------------:|----------------:|-----------------:|------------:|------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| BabyTransformerPositional   | AbsolutePositionalEncoder | Single-Head Causal | RMSNorm |    9,095,744 |       4.6855 |          4.8987 |           4.9112 |       135.8 |       35.74 | Thy king is here Thy king is hereby Abbey . " The bishop can be the first abbreviation for this is usually the bishopric of a diocese of Lincoln of bishopric , bishopric chapter to 1209 in the bishop |
| Baseline                    | None                      | Single-Head Causal | RMSNorm |    9,030,208 |       4.3668 |          5.1657 |           5.3426 |       209.1 |       36.03 | Thy king is here Thy king is here , when he can be seen , but the ritual may be seen . Nddropsdropareshddempsilon , is known as to the ritual in the sacred                                             |
| BabyTransformerPositionalV2 | PositionalEncoder         | Single-Head Causal | RMSNorm |    9,030,208 |       4.7304 |          4.9022 |           4.9116 |       135.9 |       35.75 | Thy king is here Thy king is here by an ovalier , a powerful snake . " He was one of whom he can be a result of a great deal which is really effective . He is much @-@ to play in my career .          |
<!-- BENCHMARK_END -->
