
# x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=torch.float)

# seq_len, d_model = x.shape[0], x.shape[1]


# # Notes: 
# # m - This value controls how much we rotate the embedding and it is depedent on the token's
# #     location in the sequence. Token's later in the sequency are rotated furhter. 
# # theta - This value controls the frequency at which we rotate. This value is predetermined, 
# #         and is depedent only on d_model and the RoPE base frequency (10k). Theta is calculated 
# #         for d_model/2 pairs and pairs that are near the beginning of the embedding rotate 
# #         have 


# i = torch.arange(0, d_model, 2, dtype=torch.float)
# theta = torch.pow(10000, -i/d_model)
# m = torch.arange(0, seq_len, dtype=torch.float)
# rope_angles = m.unsqueeze(-1) * theta
# print(f"rope_angles={rope_angles.shape}")


# cos_raw = torch.cos(rope_angles)
# sin_raw = torch.sin(rope_angles)
# print(f"cos_raw={cos_raw.shape}")
# print(f"sin_raw={sin_raw.shape}")

# cos_expaned = torch.repeat_interleave(cos_raw, dim=1)
# sin_expaned = torch.repeat_interleave(sin_raw, dim=1)
# cos_expaned = torch.repeat_interleave(cos_raw, repeats=2, dim=1)
# sin_expaned = torch.repeat_interleave(sin_raw, repeats=2, dim=1)

# print(f"cos_expaned={cos_expaned.shape}")
# print(f"sin_expaned={sin_expaned.shape}")


# x_rotated = torch.rotate_half(x)

# # R = torch.stack(
# #     [
# #         torch.stack([
# #             torch.cos(rope_angles), -torch.sin(rope_angles)
# #         ], dim=-1),
# #         torch.stack([
# #             torch.sin(rope_angles), torch.cos(rope_angles)
# #         ], dim=-1)
# #     ], dim=-1
# # )


# # print(rope_angles.shape)

# # # print(R)
# # print(R.shape)


# # # cos = torch.cos(rope_angles)
# # # sin = torch.sin(rope_angles)
# # x_grouped = x.view(4, 2, 2).unsqueeze(-1)



# # # print(R.shape)
# # # print(x_grouped.shape)
# # x_rotated = (R @ x_grouped)

# # print(x_rotated.shape)

# # print(x_rotated)

# # print(x_rotated.shape)

# # We have input of x of size 4 tokens with dim = 4

# # We need to rotate each token's embedding 








