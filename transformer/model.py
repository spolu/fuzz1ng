import math
import torch
import torch.nn as nn


def gelu(
        x,
):
    return 0.5 * x * (1 + torch.tanh(
        math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
    ))


class LayerNorm(nn.Module):
    def __init__(
            self,
            hidden_size,
            variance_epsilon=1e-12,
    ):
        super(LayerNorm, self).__init__()

        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))

        self.variance_epsilon = variance_epsilon

    def forward(
            self,
            x,
    ):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class SelfAttention(nn.Module):
    def __init__(
            self,
            hidden_size,
            attention_head_count,
            dropout,
    ):
        super(SelfAttention, self).__init__()

        self.hidden_size = hidden_size
        self.attention_head_count = attention_head_count

        assert self.hidden_size % self.attention_head_count == 0

        self.attention_head_size = int(
            self.hidden_size / self.attention_head_count
        )

        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(
            self,
            x,
    ):
        new_x_shape = x.size()[:-1] + (
            self.attention_head_count, self.attention_head_size
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            input_tensor,
    ):
        mixed_query = self.query(input_tensor)
        mixed_key = self.key(input_tensor)
        mixed_value = self.value(input_tensor)

        query = self.transpose_for_scores(mixed_query)
        key = self.transpose_for_scores(mixed_key)
        value = self.transpose_for_scores(mixed_value)

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / \
            math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.hidden_size,)

        context = context.view(*new_context_shape)

        return context


class SelfOutput(nn.Module):
    def __init__(
            self,
            hidden_size,
            dropout,
    ):
        super(SelfOutput, self).__init__()

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            hidden_states,
            input_tensor,
    ):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)

        return hidden_states


class Attention(nn.Module):
    def __init__(
            self,
            hidden_size,
            attention_head_count,
            dropout,
    ):
        super(Attention, self).__init__()
        self.self = SelfAttention(hidden_size, attention_head_count, dropout)
        self.output = SelfOutput(hidden_size, dropout)

    def forward(
            self,
            input_tensor,
    ):
        hidden_states = self.self(input_tensor)
        attention_output = self.output(hidden_states, input_tensor)

        return attention_output


class Transformer(nn.Module):
    def __init__(
            self,
            hidden_size,
            attention_head_count,
            intermediate_size,
            dropout=0.1
    ):
        super(Transformer, self).__init__()
        self.attention = Attention(hidden_size, attention_head_count, dropout)

        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.dense = nn.Linear(intermediate_size, hidden_size)

        self.layer_norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.apply(self.init_weights)

    def init_weights(
            self,
            module,
    ):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            module.beta.data.normal_(mean=0.0, std=0.02)
            module.gamma.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
            self,
            input_tensor,
    ):
        attention_output = self.attention(input_tensor)
        intermediate_output = gelu(self.intermediate(attention_output))
        block_output = self.layer_norm(
            attention_output + self.dropout(self.dense(intermediate_output))
        )

        return block_output

class Upsample(nn.Module):
    def __init__(
            self,
            config,
    ):
        super(Upsample, self).__init__()

        self.hidden_size = config.get('transformer_hidden_size')

        self.conv_transpose = nn.ConvTranspose1d(
            self.hidden_size, self.hidden_size,
            2, 2,
        )

    def forward(
            self,
            input_tensor,
    ):
        x = input_tensor.transpose(1, 2)
        x = self.conv_transpose(x)
        return x.transpose(1, 2)


class Downsample(nn.Module):
    def __init__(
            self,
            config,
    ):
        super(Upsample, self).__init__()

        self.hidden_size = config.get('transformer_hidden_size')

        self.conv = nn.Conv1d(
            self.hidden_size, self.hidden_size,
            2, 2,
        )

    def forward(
            self,
            input_tensor,
    ):
        x = input_tensor.transpose(1, 2)
        x = self.conv(x)
        return x.transpose(1, 2)


class Coverage(nn.Module):
    def __init__(
            self,
            config,
            dict_size,
            input_size,
    ):
        super(Coverage, self).__init__()

        self.device = torch.device(config.get('device'))

        self.latent_size = config.get('transformer_latent_size')
        self.embedding_size = config.get('transformer_embedding_size')
        self.hidden_size = config.get('transformer_hidden_size')
        self.intermediate_size = config.get('transformer_intermediate_size')
        self.attention_head_count = \
            config.get('transformer_attention_head_count')

        self.input_embedding = nn.Embedding(
            dict_size, self.embedding_size,
        )
        self.position_embedding = nn.Embedding(
            input_size, self.embedding_size
        )

        layers = []

        layers += [
            nn.Linear(self.embedding_size, self.hidden_size),
            Transformer(
                self.hidden_size,
                self.attention_head_count,
                self.intermediate_size,
            ),
            Transformer(
                self.hidden_size,
                self.attention_head_count,
                self.intermediate_size,
            ),
            Transformer(
                self.hidden_size,
                self.attention_head_count,
                self.intermediate_size,
            ),
            Transformer(
                self.hidden_size,
                self.attention_head_count,
                self.intermediate_size,
            ),
        ]

        k = input_size
        while k != 256:
            if k > 256:
                layers += [Downsample(config)]
                k = int(k / 2)
            else:
                layers += [Upsample(config)]
                k = int(k * 2)

        layers += [
            Transformer(
                self.hidden_size,
                self.attention_head_count,
                self.intermediate_size,
            ),
            Transformer(
                self.hidden_size,
                self.attention_head_count,
                self.intermediate_size,
            ),
            Transformer(
                self.hidden_size,
                self.attention_head_count,
                self.intermediate_size,
            ),
            Transformer(
                self.hidden_size,
                self.attention_head_count,
                self.intermediate_size,
            ),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid(),
        ]

        self.layers = nn.Sequential(*layers)

    def forward(
            self,
            inputs,
    ):
        positions = torch.arange(
            inputs.size(1), dtype=torch.long
        ).to(self.device)
        positions = positions.unsqueeze(0).expand_as(inputs)

        embeds = \
            self.input_embedding(inputs) + self.position_embedding(positions)

        return self.layers(embeds)
