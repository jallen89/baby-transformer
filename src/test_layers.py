from .layers import FeedForwardLayer, PositionalEncoding, MultiHeadAttention
import torch.nn.functional as F
import pytest
import torch

def test_feedforward_layer():
    batch_size = 32
    sequence_length = 8
    embedding_size = 512
    model_size = 2048
    
    x = torch.rand(batch_size, sequence_length, embedding_size)
    feed_forward_layer = FeedForwardLayer(embedding_size, model_size)
    o = feed_forward_layer(x)
    # Verify correct output shape
    assert o.shape == (batch_size, sequence_length, model_size)


def test_positional_encoder():
    sequence_length = 5
    d_model = 4
    positional_encoder = PositionalEncoding(d_model, sequence_length=sequence_length)

    batch_size = 1
    x = torch.rand((batch_size, sequence_length, d_model))
    #print("x.shape: ", x.shape)
    o = positional_encoder(x)
    module_pe_tensor = positional_encoder.pe.squeeze(0)
    ground_truth_tensor = torch.tensor([
        [ 0.0000,  1.0000,  0.0000,  1.0000],
        [ 0.8415,  0.5403,  0.0100,  0.9999],
        [ 0.9093, -0.4161,  0.0200,  0.9998],
        [ 0.1411, -0.9900,  0.0300,  0.9996],
        [-0.7568, -0.6536,  0.0400,  0.9992]
    ])


    assert torch.allclose(ground_truth_tensor, module_pe_tensor, atol=1e-4)


def test_multihead_attention_self_attention():
    '''
    This tests encoder self attention layer

    In this test, the L_q == L_k == L_v
    '''

    head_cnt = 8
    d_model = 64
    seq_len = 5
    batch_size = 4
    q_len = 6

    multihead_attention = MultiHeadAttention(head_cnt, d_model)
    x = torch.rand((batch_size, seq_len, d_model))
    k = v = q = x
    
    lengths = torch.arange(1, batch_size + 1).unsqueeze(1)
    positions = torch.arange(seq_len).unsqueeze(0)
    mask_2d = positions >= lengths
    attention_mask = mask_2d.unsqueeze(1).unsqueeze(1)
    q_mask = mask_2d.unsqueeze(-1)
    o_proj, scores = multihead_attention(q, k, v,  attention_mask, q_mask)
    attention_weights = F.softmax(scores, dim=-1)

    mask_to_check = attention_mask.expand_as(attention_weights)

    # Check 1 if the sum of all locations where the mask exist sum to 0 after the softmax. 
    assert torch.allclose(attention_weights[mask_to_check], torch.tensor(0.0), atol=1e-6)

    # Check 2: All the non-masked values should sum to 1
    unmask_check = ((~mask_to_check).float())
    attention_weights_sum = (attention_weights * unmask_check).sum(dim=-1)
    assert torch.allclose(attention_weights_sum, torch.tensor(1.0), atol=1e-6)

    # Check 3: The final projected output (o_proj) must be zero for padding query positions.
    output_mask = mask_2d.unsqueeze(-1).expand_as(o_proj) # (B, Q, D_MODEL)
    assert torch.allclose(o_proj[output_mask], torch.tensor(0.0), atol=1e-6)


def test_multihead_attention_cross_attention():
    '''
        Test cross-attention layer. 

        The query vector's length has different lenght than k and v.
    '''
    head_cnt = 8
    d_model = 16
    k_seq_len = 5
    batch_size = 4
    q_seq_len = 10

    multihead_attention = MultiHeadAttention(head_cnt, d_model)
    v = torch.rand((batch_size, k_seq_len, d_model))
    k = v 
    q = torch.rand((batch_size, q_seq_len, d_model))

    # ## Step 1. Create the attention mask.
    # The lengths are created like this for testing purposes. We assume each 
    # sample in the batch has a size +1 of the prior sample. 
    k_lengths = torch.arange(1, batch_size + 1).unsqueeze(1)
    k_positions = torch.arange(k_seq_len).unsqueeze(0)
    k_mask_2d = k_positions >= k_lengths
    attention_mask = k_mask_2d.unsqueeze(1).unsqueeze(1)
    # Attention mask shape should be B x 1 x 1 x K so we can broadcast down the head and query dimensions.
    assert attention_mask.shape == (batch_size, 1, 1, k_seq_len), "attention mask shape is incorrect."

    # ## Step 2. Create the q_mask 
    q_lengths = torch.arange(1, batch_size + 1).unsqueeze(1)
    q_positions = torch.arange(q_seq_len).unsqueeze(0)
    q_mask_2d = q_positions >= q_lengths 
    q_mask_3d = q_mask_2d.unsqueeze(-1)
    assert q_mask_3d.shape == (batch_size, q_seq_len, 1), "q_mask shape is correct."

    # ## Step 3. Do cross-attention operation. 
    o_proj, scores = multihead_attention(q, k, v, attention_mask, q_mask_3d)
    attention_weights = F.softmax(scores, dim=-1)

    mask_to_check = attention_mask.expand_as(attention_weights)

    # Check #1 (attention_mask): There should be no attention on padded tokens. 
    assert torch.allclose(attention_weights[mask_to_check], torch.tensor(0.0), atol=1e-6), \
          "There should be no attention on padded tokens."
    
    # Check #2 (attention_mask): The sum of attention on non-padded tokens should equal 1
    sum_results = (~mask_to_check).float() * attention_weights
    assert torch.allclose(sum_results.sum(dim=-1), torch.tensor(1.0), atol=1e-6), \
        "The sum of attention on non-padded tokens should equal"

    # Check #3 (q_mask): The sum of of the embeddings for padded tokens should be zero. 
    q_mask_to_check = q_mask_3d.expand_as(o_proj)
    assert torch.allclose(o_proj[q_mask_to_check].sum(), torch.tensor(0.0), atol=1e-6), \
        "The sum of the embeddings for padded tokens should be zero."