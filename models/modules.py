import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim, fine_tune=False):
        super(ImageEncoder, self).__init__()

        ##TODO: Dynamic for model selection
        self.cnn = models.inception_v3(pretrained=True, aux_logits=True)

        # Last 3 layers to identity otherwise this model won't work
        self.cnn.avgpool = nn.Identity()
        self.cnn.dropout = nn.Identity()
        self.cnn.fc = nn.Identity()

        ##TODO: DO we need freeze the network initially?
        for p in self.cnn.parameters():  # Freeze all of the cnn network
            p.requires_grad = False

        # Fine tune cnn (calculate gradients for backprop on last two bottlenecks)
        if fine_tune:
            for c in list(self.cnn.children())[-5:]:
                for p in c.parameters():
                    p.requires_grad = True

        self.fc = nn.Linear(2048, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = self.cnn(x)[0] if self.training else self.cnn(x)  # Get first element due to auxlogits option
        out = out.reshape(-1, 64, 2048)  ##TODO: Why reshape like this
        out = self.dropout(self.fc(out))
        
        return out


class AdditiveAttention(nn.Module):  # Bahdanau encoder-decoder attention (Additive attention)
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(AdditiveAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.encoder_linear = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.decoder_linear = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.attn_linear = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        # h_j and s_i refer to the variable names from the original formula of Bahdanau et al.
        h_j = self.encoder_linear(encoder_out)  # (batch_size, len, attention_dim)
        s_i = self.decoder_linear(decoder_hidden).squeeze(0)  # (batch_size, attention_dim)
        energy = self.attn_linear(self.tanh(h_j + s_i.unsqueeze(1))).squeeze(2)  # (batch_size, len)
        alpha = self.softmax(energy)  # (batch_size, len)
        attention_weighted_encoding = (alpha.unsqueeze(2) * h_j)  # (batch_size, len, encoder_dim)

        return attention_weighted_encoding, alpha