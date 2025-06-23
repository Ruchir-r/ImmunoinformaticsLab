import torch
import torch.nn as nn
import torch.nn.functional as F

#only CNN_CDR123_global_max_normalization has been changed for the new embedding dims

class CNN_CDR123_global_max_normalization(nn.Module):
    def __init__(
        self,
        dropout_rate: float = 0.6,
        embed_dim: int = 20,
        filters: int = 16,
        kernel_sizes: list = [1, 3, 5, 7, 9],
        linear_dim: int = 64,
    ):
        super(CNN_CDR123_global_max_normalization, self).__init__()

        # CNN parameters
        self.embed_dim = embed_dim
        self.kernel_sizes = kernel_sizes
        self.filters = filters
        self.dropout_rate = dropout_rate

        self.pep_conv_layers = nn.ModuleList(
            [self._get_conv1d_layer(kernel_size, "pep") for kernel_size in kernel_sizes]
        )
        self.a1_conv_layers = nn.ModuleList(
            [self._get_conv1d_layer(kernel_size, "cdr") for kernel_size in kernel_sizes]
        )
        self.a2_conv_layers = nn.ModuleList(
            [self._get_conv1d_layer(kernel_size, "cdr") for kernel_size in kernel_sizes]
        )
        self.a3_conv_layers = nn.ModuleList(
            [self._get_conv1d_layer(kernel_size, "cdr") for kernel_size in kernel_sizes]
        )
        self.b1_conv_layers = nn.ModuleList(
            [self._get_conv1d_layer(kernel_size, "cdr") for kernel_size in kernel_sizes]
        )
        self.b2_conv_layers = nn.ModuleList(
            [self._get_conv1d_layer(kernel_size, "cdr") for kernel_size in kernel_sizes]
        )
        self.b3_conv_layers = nn.ModuleList(
            [self._get_conv1d_layer(kernel_size, "cdr") for kernel_size in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.linear_1 = nn.Linear(7 * len(kernel_sizes) * self.filters, linear_dim)
        self.linear_2 = nn.Linear(linear_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def _get_conv1d_layer(self, kernel_size: int, seq_type):
        if seq_type == "cdr":
            return nn.Sequential(
                nn.Conv1d(
                    self.embed_dim, self.filters, kernel_size, padding="same", bias=False
                ),
                nn.BatchNorm1d(self.filters),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),
            )
        elif seq_type == "pep": #for BLOSUM encodings
            return nn.Sequential(
                nn.Conv1d(
                    20, self.filters, kernel_size, padding="same", bias=False
                ),
                nn.BatchNorm1d(self.filters),
                nn.ReLU(), 
                nn.AdaptiveMaxPool1d(1),
            )

    def forward(self, pep, a1, a2, a3, b1, b2, b3):
        pep = torch.nn.functional.normalize(pep, p=2, dim=2)   
        a1 = torch.nn.functional.normalize(a1, p=2, dim=2)   #(64, 480, 7)
        a2 = torch.nn.functional.normalize(a2, p=2, dim=2)
        a3 = torch.nn.functional.normalize(a3, p=2, dim=2)
        b1 = torch.nn.functional.normalize(b1, p=2, dim=2)
        b2 = torch.nn.functional.normalize(b2, p=2, dim=2)
        b3 = torch.nn.functional.normalize(b3, p=2, dim=2)

        pep = torch.cat([conv(pep) for conv in self.pep_conv_layers], dim=1)
        a1 = torch.cat([conv(a1) for conv in self.a1_conv_layers], dim=1)
        a2 = torch.cat([conv(a2) for conv in self.a2_conv_layers], dim=1)
        a3 = torch.cat([conv(a3) for conv in self.a3_conv_layers], dim=1)
        b1 = torch.cat([conv(b1) for conv in self.b1_conv_layers], dim=1)
        b2 = torch.cat([conv(b2) for conv in self.b2_conv_layers], dim=1)
        b3 = torch.cat([conv(b3) for conv in self.b3_conv_layers], dim=1)
        cat = torch.cat([pep, a1, a2, a3, b1, b2, b3], dim=1)
        cat = cat.permute(0, 2, 1)
        print(cat.shape)
        cat = self.dropout(cat)
        cat = torch.sigmoid(self.linear_1(cat))
        return self.linear_2(cat).view(-1)

class FFN(nn.Module):
    def __init__(
            self,
            dropout_rate: float = 0.6,
            embed_dim: int = 20,
            filters: int = 16,
            kernel_sizes: list = [1, 3, 5, 7, 9],
            linear_dim: int = 64,
        ):        
        super(FFN, self).__init__()
        
        hidden_dims = [1024, 512, 128]
        input_dim = 480
        
        self.linear_1 = nn.Linear(input_dim, linear_dim)
        self.linear_2 = nn.Linear(linear_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.linear_layers = nn.Sequential(
            nn.Linear(85 * 480, 4096),  # First projection
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(1024, 256),
            nn.ReLU(),

            nn.Linear(256, 1),  # Final output layer
            nn.Sigmoid()
        )

    def forward(self, pep, a1, a2, a3, b1, b2, b3):
        pep = torch.nn.functional.normalize(pep, p=2, dim=1)   
        a1 = torch.nn.functional.normalize(a1, p=2, dim=1)   #(64, 480, 7)
        a2 = torch.nn.functional.normalize(a2, p=2, dim=1)
        a3 = torch.nn.functional.normalize(a3, p=2, dim=1)
        b1 = torch.nn.functional.normalize(b1, p=2, dim=1)
        b2 = torch.nn.functional.normalize(b2, p=2, dim=1)
        b3 = torch.nn.functional.normalize(b3, p=2, dim=1)
        
        # Pad pep from (64, 20, seq_len) to (64, 480, seq_len)
        pad_size = 480 - pep.shape[1]
        pep_padded = F.pad(pep, (0, 0, 0, pad_size))  # Pad feature dim (dim=1)

        # Sanity check
        assert pep_padded.shape[1] == 480

        # Now concatenate along sequence dimension (dim=2)
        cat = torch.cat([pep_padded, a1, a2, a3, b1, b2, b3], dim=2)
        # cat = torch.cat([pep, a1, a2, a3, b1, b2, b3], dim=1)
        
        cat = cat.permute(0, 2, 1)
        # (64, 85, 480)
        cat = self.dropout(cat)
        cat = cat.reshape(cat.size(0), -1)  # Flatten to (64, 40800)
        out = self.linear_layers(cat)    # Shape: (64, 1)
        return out.view(-1)