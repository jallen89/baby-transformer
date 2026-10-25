# Attention

Attention contextualizes the input embeddings based on the prior tokens in the input sequence.

Example sequence:

> A house has cheese

## Scaled dot-product attention

$$
\operatorname{Attention}(Q, K, V)
=
\operatorname{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

For the example in the notes:

$$
B = 1, \qquad S = 4, \qquad d_h = 3
$$

where:

- $B$ is the batch size.
- $S$ is the sequence length.
- $d_m$ is the model/embedding dimension.
- $d_h$ is the head dimension.

The input tensor has shape:

$$
X \in \mathbb{R}^{B \times S \times d_m}
$$

Projecting the input into the value space gives:

$$
V = XW_V,
\qquad
(B \times S \times d_m)(d_m \times d_h)
\rightarrow
B \times S \times d_h
$$

For one batch item, $V$ can be viewed as one row per token and one column per head dimension:

$$
V =
\begin{array}{c|ccc}
 & d_0 & d_1 & d_2 \\
\hline
\text{A}      & A_0      & A_1      & A_2 \\
\text{house}  & house_0  & house_1  & house_2 \\
\text{has}    & has_0    & has_1    & has_2 \\
\text{cheese} & cheese_0 & cheese_1 & cheese_2
\end{array}
$$

## Query-key scores

The product $QK^T$ has shape $S \times S$. Each row is a query token and each column is a key token:

$$
QK^T =
\begin{array}{c|cccc}
 & \text{A}_k & \text{house}_k & \text{has}_k & \text{cheese}_k \\
\hline
\text{A}_q      & A_q \cdot A_k      & A_q \cdot house_k      & A_q \cdot has_k      & A_q \cdot cheese_k \\
\text{house}_q  & house_q \cdot A_k  & house_q \cdot house_k  & house_q \cdot has_k  & house_q \cdot cheese_k \\
\text{has}_q    & has_q \cdot A_k    & has_q \cdot house_k    & has_q \cdot has_k    & has_q \cdot cheese_k \\
\text{cheese}_q & cheese_q \cdot A_k & cheese_q \cdot house_k & cheese_q \cdot has_k & cheese_q \cdot cheese_k
\end{array}
$$

## Causal mask

A causal mask prevents a token from attending to tokens that occur after it. Add $-\infty$ above the main diagonal of $QK^T$:

$$
QK^T_{\text{masked}} =
\begin{array}{c|cccc}
 & \text{A}_k & \text{house}_k & \text{has}_k & \text{cheese}_k \\
\hline
\text{A}_q
  & A_q \cdot A_k
  & -\infty
  & -\infty
  & -\infty \\
\text{house}_q
  & house_q \cdot A_k
  & house_q \cdot house_k
  & -\infty
  & -\infty \\
\text{has}_q
  & has_q \cdot A_k
  & has_q \cdot house_k
  & has_q \cdot has_k
  & -\infty \\
\text{cheese}_q
  & cheese_q \cdot A_k
  & cheese_q \cdot house_k
  & cheese_q \cdot has_k
  & cheese_q \cdot cheese_k
\end{array}
$$

This is a lower-triangular causal-attention pattern:

- `A` can attend only to `A`.
- `house` can attend to `A` and `house`.
- `has` can attend to `A`, `house`, and `has`.
- `cheese` can attend to every token because no token comes after it.

Scale the masked scores and apply softmax:

$$
A = \operatorname{softmax}\!\left(
\frac{QK^T_{\text{masked}}}{\sqrt{d_k}}
\right),
\qquad
A \in \mathbb{R}^{S \times S}
$$

The masked entries become zero after softmax. Finally, multiply the attention weights by the value vectors:

$$
AV:
\quad
(S \times S)(S \times d_h)
\rightarrow
S \times d_h
$$

Thus, each output row is a weighted combination of the value vectors for the current token and the tokens before it. For the running example, $A \in \mathbb{R}^{4 \times 4}$ and $V \in \mathbb{R}^{4 \times 3}$, so:

$$
AV \in \mathbb{R}^{4 \times 3}
$$
