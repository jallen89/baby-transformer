Implementation of the original transformer in Attention Is All You Need paper

<!-- BENCHMARK_START -->
## Model Benchmark Results

| Model                     | Positional Emb            | Attention Type     | Norm    |   Parameters |   Final Train Loss |   Best Val Loss |   Final Val Loss |   Final PPL |   Steps/sec | Example Output                                                     |
|:--------------------------|:--------------------------|:-------------------|:--------|-------------:|-------------------:|----------------:|-----------------:|------------:|------------:|:-------------------------------------------------------------------|
| Baseline                  | None                      | Single-Head Causal | RMSNorm |    1,209,632 |             4.6942 |          5.0478 |           5.0748 |       159.9 |       75.18 | The king shall set you free' our foes me now                       |
|                           |                           |                    |         |              |                    |                 |                  |             |             | That he is the king, as in his princely the king and not in a good |
|                           |                           |                    |         |              |                    |                 |                  |             |             |                                                                    |
|                           |                           |                    |         |              |                    |                 |                  |             |             | For to the gods:                                                   |
|                           |                           |                    |         |              |                    |                 |                  |             |             | I had the state                                                    |
|                           |                           |                    |         |              |                    |                 |                  |             |             | Wess, and we' the lill'd for                                       |
|                           |                           |                    |         |              |                    |                 |                  |             |             | O great                                                            |
| BabyTransformerPositional | AbsolutePositionalEncoder | Single-Head Causal | RMSNorm |    1,217,824 |             4.9867 |          5.1662 |           5.1723 |       176.3 |       72.28 | The king shall set you free' our foes me now                       |
|                           |                           |                    |         |              |                    |                 |                  |             |             | That he is the king, as in his princely the king and not in a good |
|                           |                           |                    |         |              |                    |                 |                  |             |             |                                                                    |
|                           |                           |                    |         |              |                    |                 |                  |             |             | For to the gods:                                                   |
|                           |                           |                    |         |              |                    |                 |                  |             |             | I had the state                                                    |
|                           |                           |                    |         |              |                    |                 |                  |             |             | Wess, and we' the lill'd for                                       |
|                           |                           |                    |         |              |                    |                 |                  |             |             | O great                                                            |
<!-- BENCHMARK_END -->
