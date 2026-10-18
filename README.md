Implementation of the original transformer in Attention Is All You Need paper

<!-- BENCHMARK_START -->
## Model Benchmark Results

| Model                     | Positional Emb            | Attention Type     | Norm    |   Parameters |   Train Loss |   Best Val Loss |   Final Val Loss |   Final PPL |   Steps/sec | Completion                                     |
|:--------------------------|:--------------------------|:-------------------|:--------|-------------:|-------------:|----------------:|-----------------:|------------:|------------:|:-----------------------------------------------|
| Baseline                  | None                      | Single-Head Causal | RMSNorm |    1,466,632 |       6.31   |          6.0852 |           6.0852 |       439.3 |      239.37 | Thy king is here Thy king is here the dead me? |
|                           |                           |                    |         |              |              |                 |                  |             |             | To's:                                          |
|                           |                           |                    |         |              |              |                 |                  |             |             | But me and to the common?                      |
|                           |                           |                    |         |              |              |                 |                  |             |             | SICINIUS we, or:                               |
|                           |                           |                    |         |              |              |                 |                  |             |             | That;                                          |
|                           |                           |                    |         |              |              |                 |                  |             |             | Then and that with'sting him:                  |
|                           |                           |                    |         |              |              |                 |                  |             |             | Cing our he                                    |
| BabyTransformerPositional | AbsolutePositionalEncoder | Single-Head Causal | RMSNorm |    1,474,824 |       6.4486 |          6.3156 |           6.3156 |       553.2 |      228.04 | Thy king is here Thy king is here the, and?    |
|                           |                           |                    |         |              |              |                 |                  |             |             | Toed:                                          |
|                           |                           |                    |         |              |              |                 |                  |             |             | But me and to the!                             |
|                           |                           |                    |         |              |              |                 |                  |             |             |                                                |
|                           |                           |                    |         |              |              |                 |                  |             |             | He we,                                         |
|                           |                           |                    |         |              |              |                 |                  |             |             |                                                |
|                           |                           |                    |         |              |              |                 |                  |             |             |                                                |
|                           |                           |                    |         |              |              |                 |                  |             |             | That in the thy and to,                        |
|                           |                           |                    |         |              |              |                 |                  |             |             | B the'll I the,                                |
|                           |                           |                    |         |              |              |                 |                  |             |             | What the                                       |
<!-- BENCHMARK_END -->
