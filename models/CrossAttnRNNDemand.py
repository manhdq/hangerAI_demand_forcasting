from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.models as models
from fairseq.optim.adafactor import Adafactor

from .modules import FusionNetwork, GTrendEncoder, ImageEncoder, TemporalEncoder, AttributeEncoder, \
                    GTrendEncoder, FusionNetwork, PositionalEncoding, \
                    TSEncoder, AdditiveAttention


class CrossAttnRNN(pl.LightningModule):
    def __init__(self,
                attention_dim,
                embedding_dim,
                hidden_dim,
                cat_dict,
                col_dict,
                fab_dict,
                trend_len,
                num_trends,
                use_img,
                use_attribute,
                use_trends,
                out_len=12,
                use_teacher_forcing=False,
                teacher_forcing_ratio=0.5,
                lr=0.001):
        super().__init__()
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.use_teacher_forcing = use_teacher_forcing
        self.out_len = out_len
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.use_img = use_img
        self.use_attribute = use_attribute
        self.use_trends = use_trends
        self.lr = lr

        # Encoders
        self.trend_encoder = TSEncoder(num_trends, embedding_dim)
        self.image_encoder = ImageEncoder(fine_tune=False)
        self.temporal_encoder = TemporalEncoder(embedding_dim)
        self.attribute_encoder = AttributeEncoder(
            len(cat_dict) + 1,
            len(col_dict) + 1,
            len(fab_dict) + 1,
            embedding_dim
        )

        self.img_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.img_linear = nn.Linear(2048, embedding_dim)

        self.static_feature_encoder = FusionNetwork(embedding_dim, hidden_dim, use_img, use_attribute)
        self.ts_embedder = nn.GRU(1, embedding_dim, batch_first=True)

        # Attention modules
        self.trend_self_attention = nn.MultiheadAttention(embedding_dim, num_heads=4, dropout=0.1)
        self.trend_attention = AdditiveAttention(embedding_dim, hidden_dim, attention_dim)
        self.trend_linear = nn.Linear(trend_len*attention_dim, embedding_dim)
        self.multimodal_attention = AdditiveAttention(embedding_dim, hidden_dim, attention_dim)
        self.multimodal_embedder = nn.Linear(attention_dim, embedding_dim)

        # Decoder
        self.decoder = nn.GRU(
            input_size=embedding_dim + 1,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.decoder_fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, 
                ts, 
                categories, 
                colors, 
                fabrics, 
                stores, 
                temporal_features, 
                gtrends, 
                images):
        bs = ts.shape[0]

        # Encode input data
        gtrend_encoding = self.trend_encoder(gtrends.permute(0, 2, 1))
        img_encoding = self.image_encoder(images)
        temporal_encoding = self.temporal_encoder(temporal_features)
        attribute_encoding = self.attribute_encoder(categories, colors, fabrics)

        condensed_img = self.img_linear(self.img_pool(img_encoding).flatten(1))

        if self.use_trends:
            # Self-attention over the temporal features (this filters initial noise from the time series features)
            gtrend_encoding, trend_self_attn_weights = self.trend_self_attention(
                gtrend_encoding.permute(1, 0, 2),
                gtrend_encoding.permute(1, 0, 2),
                gtrend_encoding.permute(1, 0, 2),
            )

        # Predictions vector (will contain all forecasts)
        outputs = torch.zeros(bs, self.out_len, 1).to(self.device)

        # # Fuse static features together
        # static_feature_fusion = self.static_feature_encoder(img_encoding, attribute_encoding, temporal_encoding)

        # Predictions vector (will contain all forecasts)
        outputs = torch.zeros(bs, self.out_len, 1).to(self.device)

        # Init initial decoder status and outputs
        decoder_hidden = torch.zeros(1, bs, self.hidden_dim).to(self.device)
        decoder_output = torch.zeros(bs, 1, 1).to(self.device)

        decoder_out = None
        # Autoregressive rolling forecast
        for t in range(self.out_len):
            # Temporal (Exogenous Gtrends) Attention
            if self.use_trends:
                attended_trend_encoding, trend_alpha = self.trend_attention(
                    gtrend_encoding.permute(1, 0, 2), decoder_hidden
                )
                attended_trend_encoding = self.trend_linear(attended_trend_encoding.view(bs, -1))

            # Build multimodal input based on specified input modalities
            mm_in = temporal_encoding.unsqueeze(0)
            if self.use_img:
                mm_in = torch.cat([mm_in, condensed_img.unsqueeze(0)])
            if self.use_attribute:
                mm_in = torch.cat([mm_in, attribute_encoding.unsqueeze(0)])
            if self.use_trends:
                mm_in = torch.cat([mm_in, attended_trend_encoding.unsqueeze(0)])
            mm_in = mm_in.permute(1, 0, 2)

            # Multimodal attention
            attended_multimodal_encoding, multimodal_alpha = self.multimodal_attention(
                mm_in, decoder_hidden
            )

            final_encoder_output = self.multimodal_embedder(
                attended_multimodal_encoding.sum(1)  # reduce sum
            )  # BS x 1 x D

            # Concatenate last prediction to the encoder output -> autoregression
            x_input = torch.cat(
                [final_encoder_output.unsqueeze(1), decoder_output], dim=2
            )
            
            # GRU decoder
            decoder_out, decoder_hidden = self.decoder(x_input, decoder_hidden)
            decoder_output = self.decoder_fc(decoder_out)
            outputs[:, t, :] = decoder_output[:, 0, :]

            # Control teacher forcing
            teach_forcing = True if torch.rand(1) < self.teacher_forcing_ratio else False
            if self.use_teacher_forcing and teach_forcing and ts is not None:
                decoder_output = ts[:, t].unsqueeze(-1).unsqueeze(-1)

        return outputs

    def configure_optimizers(self):
        optimizer = Adafactor(
            self.parameters(),
            scale_parameter=True,
            relative_step=True,
            warmup_init=True,
            lr=None,
        )

        return [optimizer]

    def on_train_epoch_start(self):
        ##TODO: Set requires_grad for loss when teacher is applied for every item in batch
        self.use_teacher_forcing = False  # Allow for teacher forcing when training model

    def training_step(self, train_batch, batch_idx):
        (ts, categories, colors, fabrics, stores, temporal_features, gtrends), images = train_batch

        forecasted_sales = self.forward(
            ts,
            categories,
            colors,
            fabrics,
            stores,
            temporal_features,
            gtrends,
            images
        )
        loss = F.mse_loss(ts, forecasted_sales.squeeze())
        self.log("train_loss", loss)

        return loss

    def on_validation_epoch_start(self):
        self.use_teacher_forcing = False  # No teacher forcing when evaluating model

    def validation_step(self, test_batch, batch_idx):
        (ts, categories, colors, fabrics, stores, temporal_features, gtrends), images = test_batch

        forecasted_sales = self.forward(
            ts,
            categories,
            colors,
            fabrics,
            stores,
            temporal_features,
            gtrends,
            images
        )
        return ts, forecasted_sales

    def validation_epoch_end(self, val_step_outputs):
    
        item_sales, forecasted_sales = (
            [x[0] for x in val_step_outputs],
            [x[1] for x in val_step_outputs],
        )
        item_sales, forecasted_sales = (
            torch.vstack(item_sales),
            torch.vstack(forecasted_sales),
        )
        item_sales, forecasted_sales = item_sales.squeeze(), forecasted_sales.squeeze()
        rescaled_item_sales, rescaled_forecasted_sales = (
            item_sales * 53,
            forecasted_sales * 53,
        )  # 53 is the normalization factor (max of the sales of the training set)
        loss = F.mse_loss(item_sales, forecasted_sales)
        mae = F.l1_loss(rescaled_item_sales, rescaled_forecasted_sales)
        wape = 100 * torch.sum(torch.abs(rescaled_item_sales - rescaled_forecasted_sales)) / torch.sum(rescaled_item_sales)
        ts = 100 * torch.sum(rescaled_item_sales - rescaled_forecasted_sales) / mae

        self.log("val_mae", mae)
        self.log("val_wWAPE", wape)
        self.log("val_wTS", ts)
        self.log("val_loss", loss)

        print(
            "Validation MAE:",
            mae.detach().cpu().numpy(),
            "Validation WAPE:",
            wape.detach().cpu().numpy(),
            "Validation Tracking Signal:",
            ts.detach().cpu().numpy(),
            "LR:",
            self.optimizers().param_groups[0]["lr"],
        )