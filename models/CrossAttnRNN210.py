from thop import profile

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from fairseq.optim.adafactor import Adafactor

from .modules import ImageEncoder, TemporalEncoder, AttributeEncoder, \
                    GTrendEncoder, FusionNetwork, PositionalEncoding, \
                    AdditiveAttention


class CrossAttnRNN(pl.LightningModule):
    def __init__(
        self,
        attention_dim,
        embedding_dim,
        hidden_dim,
        use_img,
        use_attribute,
        out_len,
        cat_dict,
        col_dict,
        fab_dict,
        trend_len, 
        num_trends,
        use_encoder_mask,
        gpu_num,
        use_teacher_forcing=False,
        teacher_forcing_ratio=0.5,
    ):
        super().__init__()
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.use_teacher_forcing = use_teacher_forcing
        self.out_len = out_len
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.use_img = use_img
        self.use_attribute = use_attribute
        self.autoregressive = 1  # Using autoregressive as default

        # Encoders
        self.image_encoder = ImageEncoder(fine_tune=False)
        self.temporal_encoder = TemporalEncoder(embedding_dim)
        self.attribute_encoder = AttributeEncoder(embedding_dim, cat_dict, col_dict, fab_dict, gpu_num)
        # self.gtrend_encoder = GTrendEncoder(out_len, hidden_dim, use_encoder_mask, trend_len, num_trends, gpu_num)
        self.static_feature_encoder = FusionNetwork(embedding_dim, hidden_dim, use_img, use_attribute)
        self.ts_embedder = nn.GRU(1, embedding_dim, batch_first=True)

        # # Attention module
        # self.img_attention = AdditiveAttention(embedding_dim, hidden_dim, attention_dim)

        # Decoder
        self.decoder_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers= 3,
            batch_first=True
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim//2, 1)
        )

    def _generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to('cuda:'+str(self.gpu_num))
        return mask

    def forward(self, X, y, 
                categories, colors, fabrics, stores,
                temporal_features, gtrends,
                images):
        # B = X.shape[0]
        predictions = []  # List of store predictions of dynamically unrolled outputs.

        # Encode statuc input data
        img_encoding = self.image_encoder(images)
        temporal_encoding = self.temporal_encoder(temporal_features)
        attribute_encoding = self.attribute_encoder(categories, colors, fabrics)
        # gtrend_encoding = self.gtrend_encoder(gtrends)
        
        # Fuse static features together
        static_feature_fusion = self.static_feature_encoder(img_encoding, attribute_encoding, temporal_encoding)

        # Build multimodal input based on specified input modalities
        ts_input = X[:, 0, :].unsqueeze(-1)  # Only the first window is selected for succesive autoregressive forecasts
        ts_features = self.ts_embedder(ts_input)[1].permute(1, 0, 2)

        x = static_feature_fusion.unsqueeze(1) + ts_features

        decoder_out, decoder_hidden = self.decoder_gru(x)
        pred = self.decoder_fc(decoder_out).squeeze(-1)

        # Insert the first prediction
        predictions.append(pred)

        # Autoregressive rolling forecast
        for t in range(1, self.out_len):
            x = decoder_out

            #### Autoregressive decoding
            decoder_out, decoder_hidden = self.decoder_gru(x, decoder_hidden)
            pred = self.decoder_fc(decoder_out).squeeze(-1)

            # Control teacher forcing
            if self.use_teacher_forcing:
                teacher_forcing_prob = True if torch.rand(1) < self.teacher_forcing_ratio else False
                if teacher_forcing_prob and y is not None:
                    predictions.append(y[:, t, :])
                else:
                    predictions.append(pred)
            else:
                predictions.append(pred)

        outputs = torch.stack(predictions).squeeze().T

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

    def get_flops_and_params(self):
        X = torch.randn(1, 10, 2)
        y = torch.randn(1, 10, 1)
        categories = torch.randn(1)
        colors = torch.randn(1)
        fabrics = torch.randn(1)
        stores = torch.randn(1)
        temporal_features = torch.randn(1, 4)
        gtrends = torch.randn(1, 3, 52)
        images = torch.randn(1, 3, 299, 299)
        macs, params = profile(self, inputs=(X, y,
                                        categories, colors, fabrics, stores,
                                        temporal_features, gtrends,
                                        images))
        return macs * 2 / 10**9, params / 10**6

    def on_train_epoch_start(self):
        self.use_teacher_forcing = True  # Allow for teacher forcing when training model

    def training_step(self, train_batch, batch_idx):
        (X, y, categories, colors, fabrics, stores, temporal_features, gtrends), images = train_batch
        forecasted_sales = self.forward(X, y, 
                                        categories, colors, fabrics, stores,
                                        temporal_features, gtrends,
                                        images)
        y = y.squeeze()
        loss = F.mse_loss(y, forecasted_sales)
        self.log("train_loss", loss)

        return loss

    def on_validation_epoch_start(self):
        self.use_teacher_forcing = False  # No teacher forcing when evaluating model

    def validation_step(self, test_batch, batch_idx):
        (X, y, categories, colors, fabrics, stores, temporal_features, gtrends), images = test_batch
        forecasted_sales = self.forward(X, y, 
                                        categories, colors, fabrics, stores,
                                        temporal_features, gtrends,
                                        images)

        return y, forecasted_sales

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
        )  # 53 is the normalization factor (max of the sales of the training set == stfore_sales_norm_scalar.npy)
        loss = F.mse_loss(item_sales, forecasted_sales)
        mae = F.l1_loss(rescaled_item_sales, rescaled_forecasted_sales)
        wape = 100 * torch.sum(torch.abs(rescaled_item_sales - rescaled_forecasted_sales)) / torch.sum(rescaled_item_sales)
        ts = 100 * torch.sum(rescaled_item_sales - rescaled_forecasted_sales) / mae

        self.log("val_mae", mae)
        self.log("val_wWAPE", wape)
        self.log("val_wTS", ts)
        self.log("val_loss", loss)

        print(
            "Validation MAE:", mae.detach().cpu().numpy(),
            "Validation WAPE:", wape.detach().cpu().numpy(),
            "Validation Tracking Signal:", ts.detach().cpu().numpy(),
            "LR:", self.optimizers().param_groups[0]["lr"],
        )