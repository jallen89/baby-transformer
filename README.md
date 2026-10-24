Implementation of the original transformer in Attention Is All You Need paper

<!-- BENCHMARK_START -->
## Model Benchmark Results

| Model                     | Positional Emb            | Attention Type     | Norm    |   Parameters |   Train Loss |   Best Val Loss |   Final Val Loss |   Final PPL |   Steps/sec | Completion                                                                                                                                                                                                |
|:--------------------------|:--------------------------|:-------------------|:--------|-------------:|-------------:|----------------:|-----------------:|------------:|------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Baseline                  | None                      | Single-Head Causal | RMSNorm |    1,466,632 |       4.0353 |           6.077 |           7.2267 |      1375.7 |      137.02 | Thy king is here Thy king is hereach ? 'da Killer Ko and Amrina , Krown and , Priyaan " . " he labelled and Murder 2 alongside bass , Wood . The duo                                                      |
| BabyTransformerPositional | AbsolutePositionalEncoder | Single-Head Causal | RMSNorm |    1,474,824 |       4.6055 |           6.339 |           6.6496 |       772.5 |      129.13 | Thy king is here Thy king is herees are often call for the original full establishment of the tomb are broken but that a " , is known as " . " he attribute that the true story , as one , and other than |
<!-- BENCHMARK_END -->
