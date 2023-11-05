# Variational Encoder Decoder
#  It contains the MIT licensed code from https://github.com/arpastrana/neu_vae/
#   notably from https://github.com/arpastrana/neu_vae/blob/main/src/neu_vae/training/run_model.py
#  ref. https://wandb.ai/arpastrana/beta_vae/reports/Disentangling-Variational-Autoencoders--VmlldzozNDQ3MDk

from functools import partial

import torch
import torch.utils.data
from torch.nn import functional

from neu_vae.models import EncoderFactory
from neu_vae.models import DecoderFactory
from neu_vae.models import VAEFactory
import torch.nn.utils as utils


class VariationalEncoderDecoder:

    def __init__(self, config):
        self.model = self.create(config)

    @classmethod
    def create(cls, config):
        # create encoder
        n_classes = config["n_classes"]
        enc_kwargs = {"input_dim": config["input_dim"],
                      "hidden_dim": config["encoder_hidden_dim"],
                      "z_dim": config["z_dim"],
                      "act_func": getattr(torch, config["encoder_act_func"]),
                      "n_classes": n_classes}

        encoder = EncoderFactory.create(config["encoder_name"])
        encoder_initialized = encoder(**enc_kwargs)

        # create decoder
        dec_kwargs = {"z_dim": config["z_dim"],
                      "hidden_dim": config["decoder_hidden_dim"],
                      "output_dim": config["output_dim"],  # change from AutoEncoder
                      "act_func": getattr(torch, config["decoder_act_func"]),
                      "pred_func": getattr(torch, config["decoder_pred_func"]),
                      "n_classes": n_classes}

        decoder = DecoderFactory.create(config["decoder_name"])
        decoder_initialized = decoder(**dec_kwargs)

        # assemble VAE
        reconstruction_loss = partial(getattr(functional, config["rec_loss"]),
                                      reduction="sum")

        vae_kwargs = {"encoder": encoder_initialized,
                      "decoder": decoder_initialized,
                      "recon_loss_func": reconstruction_loss,
                      "beta": config["beta"]}

        # selecte VAE model
        vae = VAEFactory.create(config["vae_name"])
        model = vae(**vae_kwargs)

        # print model summary
        print("----------------------------------------------------------------")
        print(f"Model: {model.name}")
        # summary(model, (1, config["input_dim"] + config["n_classes"]))

        return model.to(config['device'])

    def learn(self, x, y, optimizer):
        optimizer.zero_grad()
        # forward pass
        y_hat, z_mean, z_logvar = self.model(x, None)
        # loss
        loss_dict = self.model.loss(y, y_hat, z_mean, z_logvar)
        loss = loss_dict["loss"]

        # backward pass
        optimizer.zero_grad()
        loss.backward()

        # The following is inserted to prevent 'nan' output
        # Set the maximum norm value to 1.0
        max_norm = 1.0
        # Calculate the norm of the gradients
        utils.clip_grad_norm_(self.model.parameters(), max_norm)

        optimizer.step()
        return loss_dict
