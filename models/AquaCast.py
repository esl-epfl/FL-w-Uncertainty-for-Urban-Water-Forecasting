__all__ = ['AquaCast']

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        # Load config values
        context_window = configs.seq_len
        target_window = configs.pred_len
        nvars = configs.enc_in
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        kernel_size = configs.kernel_size
        stride = configs.stride

        self.use_exo_future = configs.exo_future
        self.n_pred_var = configs.enc_in - (configs.exo)

        # CNN embeddings for main input
        self.embedding1 = nn.Conv1d(in_channels=nvars, out_channels=d_model // 4,
                                    kernel_size=kernel_size, stride=stride, bias=True)
        self.embedding2 = nn.Conv1d(in_channels=d_model // 4, out_channels=d_model,
                                    kernel_size=kernel_size, stride=stride, bias=True)

        # Compute length after two Conv1D layers
        conv_out_len = self.calculate_output_length_conv1d(context_window, kernel_size, stride)
        conv_out_len = self.calculate_output_length_conv1d(conv_out_len, kernel_size, stride)
        self.conv_out_len = conv_out_len

        # Optional embedding for exogenous future time series
        if self.use_exo_future:
            # #option1: CNN1D + AvgPool1D
            # self.exo_embedder = nn.Sequential(
            #     # nn.Conv1d(1, d_model, kernel_size=kernel_size, padding=1),
            #     # nn.AdaptiveAvgPool1d(1)
            # )
            
            #option2: Linear layer
            # self.exo_embedder = nn.Linear(configs.pred_len, d_model)  # [B, Exo_len, 1] -> [B, 1, d_model]
            self.exo_embedder = nn.Linear(36, d_model)  # [B, Exo_len, 1] -> [B, 1, d_model]
            
            # #option3: Conv1D + linear
            # conv_exo_out_len = self.calculate_output_length_conv1d(target_window, kernel_size, stride)
            # self.exo_embedder = nn.Sequential(
            #     nn.Conv1d(1, d_model, kernel_size=kernel_size, stride=stride, bias=True),
            #     nn.Flatten(start_dim=-2),
            #     nn.Linear(d_model * conv_exo_out_len, d_model)
            # ) # [B, Exo_len, 1] -> [B, d_model]

        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                                        dim_feedforward=d_ff, dropout=dropout,
                                                        batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)

        self.flatten = nn.Flatten(start_dim=-2)
        total_len = conv_out_len + (1 if self.use_exo_future else 0)
        self.decoder = nn.Linear(d_model * total_len, target_window * self.n_pred_var)

    def calculate_output_length_conv1d(self, length_in, kernel_size, stride=1, padding=0, dilation=1):
        return (length_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    def forward(self, x, exo_future=None):
        """
        x:           [Batch, Input_len, nvars]
        exo_future:  [Batch, Exo_len, 1] or None
        """
        batch_size = x.size(0)
        # CNN Embedding of main input
        x = x.permute(0, 2, 1)               # [B, nvars, Input_len]
        x = self.embedding1(x)              # [B, d_model//4, L']
        x = self.embedding2(x)              # [B, d_model, L'']
        x = x.permute(0, 2, 1)              # [B, L'', d_model]

        # Optional embedding of exo forecast (exo_future)
        if self.use_exo_future:
            exo_token = self.exo_embedder(exo_future.permute(0, 2, 1))#.unsqueeze(1)  # new: [B, 1, Exo_len]
            # exo_token = exo_token.squeeze(-1).unsqueeze(1)          # [B, 1, d_model]
            x = torch.cat([exo_token, x], dim=1) 

        # Transformer and decoding
        x = self.transformer_encoder(x)           # [B, total_len, d_model]
        x = self.flatten(x)                       # [B, total_len * d_model]
        x = self.decoder(x)                       # [B, target_window * n_pred_var]
        x = x.view(batch_size, -1, self.n_pred_var)  # [B, target_window, n_pred_var]

        return x

if __name__ == "__main__":
    # --- Mock Config Object ---
    class Configs:
        def __init__(self):
            self.seq_len = 96           # Input sequence length
            self.pred_len = 48          # Output prediction length
            self.enc_in = 5             # Number of input variables (including exo features)
            self.e_layers = 2           # Transformer encoder layers
            self.n_heads = 4            # Attention heads
            self.d_model = 64           # Embedding/Transformer dimension
            self.d_ff = 128             # Feedforward dimension in Transformer
            self.dropout = 0.1
            self.kernel_size = 3
            self.stride = 2
            self.exo = 1                # If you're excluding one exo var from prediction (e.g. f0)
            self.exo_future = 1         # Indicates rain forecast will be provided

    # --- Initialize Model ---
    configs = Configs()
    model = Model(configs)

    # --- Dummy Input ---
    batch_size = 8
    input_len = configs.seq_len
    nvars = configs.enc_in
    exo_len = configs.pred_len
    exo_future = torch.randn(batch_size, exo_len, 1)  # [B, pred_len, 1]

    x = torch.randn(batch_size, input_len, nvars)         # [B, T, V]

    # --- Forward Pass ---
    output = model(x, exo_future)                         # [B, pred_len, n_pred_var]

    # --- Print Output Info ---
    print("Output shape:", output.shape)
    print("Expected shape:", (batch_size, configs.pred_len, configs.enc_in - (configs.exo + configs.exo_future)))
