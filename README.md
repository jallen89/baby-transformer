Implementation of the original transformer in Attention Is All You Need paper

<!-- BENCHMARK_START -->
## Model Benchmark Results

| Model                       | Positional Emb            | Attention Type     | Norm    |   Parameters |   Train Loss |   Best Val Loss |   Final Val Loss |   Final PPL |   Steps/sec | Completion                                                                                                                                                                                              |
|:----------------------------|:--------------------------|:-------------------|:--------|-------------:|-------------:|----------------:|-----------------:|------------:|------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| BabyTransformerPositional   | AbsolutePositionalEncoder | Single-Head Causal | RMSNorm |    2,319,744 |       4.8986 |          5.0737 |           5.0832 |       161.3 |       76.04 | Thy king is here Thy king is here by Lithmn , and the kit , and that have a few years is likely , and he has since in Bðmun Temple of the first king of Djedkare Ishmunn                                |
| Baseline                    | None                      | Single-Head Causal | RMSNorm |    2,286,976 |       4.6268 |          5.1181 |           5.1314 |       169.3 |       78.65 | Thy king is here Thy king is here by the king 's name . This seems to have been that have a royal officials is likely , and he has since in royal prerative of the monarch are often used . In the same |
| BabyTransformerPositionalV2 | PositionalEncoder         | Single-Head Causal | RMSNorm |    2,286,976 |       4.9415 |          5.0551 |           5.0602 |       157.6 |       76.01 | Thy king is here Thy king is here by the king 's name . A. Pavadan and P. Rain , and T.g.2 A.56 , G. B.g. W.                                                                                            |
<!-- BENCHMARK_END -->
