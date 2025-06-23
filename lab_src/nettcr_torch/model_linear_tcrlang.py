import torch
import torch.nn as nn

#only CNN_CDR123_global_max_normalization has been changed for the new embedding dims

class CNN_CDR123_global_max(nn.Module):
    def __init__(
        self,
        dropout_rate: float = 0.6,
        embed_dim: int = 20,
        filters: int = 16,
        kernel_sizes: list = [1, 3, 5, 7, 9],
        linear_dim: int = 64,
    ):
        super(CNN_CDR123_global_max, self).__init__()

        # CNN parameters
        self.embed_dim = embed_dim
        self.kernel_sizes = kernel_sizes
        self.filters = filters
        self.dropout_rate = dropout_rate

        self.pep_conv_layers = nn.ModuleList(
            [self._get_conv1d_layer(kernel_size) for kernel_size in kernel_sizes]
        )
        self.a1_conv_layers = nn.ModuleList(
            [self._get_conv1d_layer(kernel_size) for kernel_size in kernel_sizes]
        )
        self.a2_conv_layers = nn.ModuleList(
            [self._get_conv1d_layer(kernel_size) for kernel_size in kernel_sizes]
        )
        self.a3_conv_layers = nn.ModuleList(
            [self._get_conv1d_layer(kernel_size) for kernel_size in kernel_sizes]
        )
        self.b1_conv_layers = nn.ModuleList(
            [self._get_conv1d_layer(kernel_size) for kernel_size in kernel_sizes]
        )
        self.b2_conv_layers = nn.ModuleList(
            [self._get_conv1d_layer(kernel_size) for kernel_size in kernel_sizes]
        )
        self.b3_conv_layers = nn.ModuleList(
            [self._get_conv1d_layer(kernel_size) for kernel_size in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.linear_1 = nn.Linear(7 * len(kernel_sizes) * self.filters, linear_dim)
        self.linear_2 = nn.Linear(linear_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def _get_conv1d_layer(self, kernel_size: int):
        return nn.Sequential(
            nn.Conv1d(
                self.embed_dim, self.filters, kernel_size, padding="same", bias=False
            ),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )

    def forward(self, pep, a1, a2, a3, b1, b2, b3):
        # Normalization factor for BLOSUM encoding
        pep /= 5
        a1 /= 5
        a2 /= 5
        a3 /= 5
        b1 /= 5
        b2 /= 5
        b3 /= 5

        pep = torch.cat([conv(pep) for conv in self.pep_conv_layers], dim=1)
        a1 = torch.cat([conv(a1) for conv in self.a1_conv_layers], dim=1)
        a2 = torch.cat([conv(a2) for conv in self.a2_conv_layers], dim=1)
        a3 = torch.cat([conv(a3) for conv in self.a3_conv_layers], dim=1)
        b1 = torch.cat([conv(b1) for conv in self.b1_conv_layers], dim=1)
        b2 = torch.cat([conv(b2) for conv in self.b2_conv_layers], dim=1)
        b3 = torch.cat([conv(b3) for conv in self.b3_conv_layers], dim=1)
        cat = torch.cat([pep, a1, a2, a3, b1, b2, b3], dim=1)
        cat = cat.permute(0, 2, 1)
        cat = self.dropout(cat)
        cat = torch.sigmoid(self.linear_1(cat))
        return self.linear_2(cat).view(-1)

class CNN_CDR123_global_max_normalization(nn.Module):
    def __init__(
        self,
        dropout_rate: float = 0.6,
        embed_dim: int = 480,
        filters: int = 16,
        kernel_sizes: list = [1, 3, 5, 7, 9],
        linear_dim: int = 64,
    ):
        cdr_embed_dim = embed_dim
        pep_embed_dim = 20
        cdr_projected_dim = 64
        
        super(CNN_CDR123_global_max_normalization, self).__init__()

        self.cdr_embed_dim = cdr_embed_dim
        self.pep_embed_dim = pep_embed_dim
        self.cdr_projected_dim = cdr_projected_dim
        self.kernel_sizes = kernel_sizes
        self.filters = filters
        self.dropout_rate = dropout_rate

        # Projection only for CDRs
        self.cdr_proj = nn.Linear(cdr_embed_dim, cdr_projected_dim)

        # CNN layers
        self.pep_conv_layers = nn.ModuleList(
            [self._get_conv1d_layer(kernel_size, pep_embed_dim) for kernel_size in kernel_sizes]
        )
        self.a1_conv_layers = nn.ModuleList(
            [self._get_conv1d_layer(kernel_size, cdr_projected_dim) for kernel_size in kernel_sizes]
        )
        self.a2_conv_layers = nn.ModuleList(
            [self._get_conv1d_layer(kernel_size, cdr_projected_dim) for kernel_size in kernel_sizes]
        )
        self.a3_conv_layers = nn.ModuleList(
            [self._get_conv1d_layer(kernel_size, cdr_projected_dim) for kernel_size in kernel_sizes]
        )
        self.b1_conv_layers = nn.ModuleList(
            [self._get_conv1d_layer(kernel_size, cdr_projected_dim) for kernel_size in kernel_sizes]
        )
        self.b2_conv_layers = nn.ModuleList(
            [self._get_conv1d_layer(kernel_size, cdr_projected_dim) for kernel_size in kernel_sizes]
        )
        self.b3_conv_layers = nn.ModuleList(
            [self._get_conv1d_layer(kernel_size, cdr_projected_dim) for kernel_size in kernel_sizes]
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.linear_1 = nn.Linear(7 * len(kernel_sizes) * self.filters, linear_dim)
        self.linear_2 = nn.Linear(linear_dim, 1)

    def _get_conv1d_layer(self, kernel_size: int, in_channels: int):
        return nn.Sequential(
            nn.Conv1d(in_channels, self.filters, kernel_size, padding="same", bias=False),
            nn.BatchNorm1d(self.filters),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )

    def _project_cdr(self, x):
        # x: [B, L, 480] → [B, L, 64]
        return self.cdr_proj(x)

    def forward(self, pep, a1, a2, a3, b1, b2, b3):
        # Normalize each input along the embedding dimension
        print(a1.shape)
        pep = torch.nn.functional.normalize(pep, p=2, dim=2)
        a1 = torch.nn.functional.normalize(a1, p=2, dim=2)
        a2 = torch.nn.functional.normalize(a2, p=2, dim=2)
        a3 = torch.nn.functional.normalize(a3, p=2, dim=2)
        b1 = torch.nn.functional.normalize(b1, p=2, dim=2)
        b2 = torch.nn.functional.normalize(b2, p=2, dim=2)
        b3 = torch.nn.functional.normalize(b3, p=2, dim=2)

        # Permute to [B, C, L] for Conv1d
        a1 = a1.permute(0, 2, 1)
        a2 = a2.permute(0, 2, 1)
        a3 = a3.permute(0, 2, 1)
        b1 = b1.permute(0, 2, 1)
        b2 = b2.permute(0, 2, 1)
        b3 = b3.permute(0, 2, 1)

        # Project only CDR inputs
        print(a1.shape)
        a1 = self._project_cdr(a1)
        a2 = self._project_cdr(a2)
        a3 = self._project_cdr(a3)
        b1 = self._project_cdr(b1)
        b2 = self._project_cdr(b2)
        b3 = self._project_cdr(b3)

        # Permute to [B, C, L] for Conv1d
        a1 = a1.permute(0, 2, 1)
        a2 = a2.permute(0, 2, 1)
        a3 = a3.permute(0, 2, 1)
        b1 = b1.permute(0, 2, 1)
        b2 = b2.permute(0, 2, 1)
        b3 = b3.permute(0, 2, 1)

        pep = torch.cat([conv(pep) for conv in self.pep_conv_layers], dim=1)
        a1 = torch.cat([conv(a1) for conv in self.a1_conv_layers], dim=1)
        a2 = torch.cat([conv(a2) for conv in self.a2_conv_layers], dim=1)
        a3 = torch.cat([conv(a3) for conv in self.a3_conv_layers], dim=1)
        b1 = torch.cat([conv(b1) for conv in self.b1_conv_layers], dim=1)
        b2 = torch.cat([conv(b2) for conv in self.b2_conv_layers], dim=1)
        b3 = torch.cat([conv(b3) for conv in self.b3_conv_layers], dim=1)

        cat = torch.cat([pep, a1, a2, a3, b1, b2, b3], dim=1)
        cat = cat.view(cat.size(0), -1)
        cat = self.dropout(cat)
        cat = torch.relu(self.linear_1(cat))
        return self.linear_2(cat).view(-1)
