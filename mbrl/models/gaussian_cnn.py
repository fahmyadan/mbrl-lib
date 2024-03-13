from .model import Ensemble
from .gaussian_mlp import GaussianMLP
import torch 
import torch.nn as nn 
from torch.nn import functional as F
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

class GaussianCnn(GaussianMLP):

    def __init__(self, input_channels, action_dim, latent_size, **kwargs):
        super().__init__(**kwargs)

        self.encoder = None
        self.decoder = None
        self.input_channels = input_channels 
        self.action_dim = action_dim
        self.latent_size = latent_size
        self.cnn_ensemble = None 
        self.setup_models()
    
    def setup_models(self):
        #Initialise the encoder and decoder
        self.encoder = self.init_encoder()
        self.decoder = self.init_decoder()

        if self.num_members > 1:

            self.cnn_ensemble = nn.ModuleList([nn.Sequential(self.encoder, self.decoder) \
                                               for i in range(self.num_members)])

        pass

    def init_encoder(self):
        conv_layers = nn.Sequential(
            nn.Conv2d(self.input_channels, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=1),
            nn.ReLU()
        )

        encoder_dict = {'conv': conv_layers, 'linear_embed': nn.Linear(self.action_dim, 512)}

        return nn.ModuleDict(encoder_dict)

    def init_decoder(self):

        deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=6, stride=2),
            nn.ReLU()
        )

        linear = nn.Sequential(
            nn.Linear(self.latent_size, 512),
            nn.Linear(512,128),
            nn.Linear(128,1)
        )

        decoder_dict = {'deconv': deconv_layers, 'linear_layer': linear}

        return nn.ModuleDict(decoder_dict)

    def encode(self, input: torch.Tensor) -> torch.Tensor:

        # Return some latent embedding, z.

        image, action = input[:-1], input[-1:]

        z1 = self.encoder['conv'](image)
        embed = self.encoder['linear_embed'](action)

        z = torch.concat([z1,embed], dim = 0)


        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:

        # decode the latent dimension z

        image = self.decoder['deconv'](z)
        reward = self.decoder['linear'](z)

        return image, reward

    def _forward_ensemble(self, x: torch.Tensor) -> torch.Tensor:
        img, act = x
        # img = img.to(dtype=torch.float32, device=self.device)  
        # act = act.to(dtype=torch.float32, device=self.device)
        if not self.cnn_ensemble:
            z = self.encode(x)
            z = nn.Linear(z.shape[0], 512)(z)
            out = self.decode(z)
        else:
            #forward pass for cnn ensembles
            outputs = []
            for model in self.cnn_ensemble:
                enc, dec = model[0], model[1]
                enc.to(self.device)  # Ensure the model is on the correct device
                # dec.to(self.device)

                z_act = enc.linear_embed(act.to(dtype=torch.float32))
                z_img = enc.conv(img.to(dtype=torch.float32))


                latent_z = torch.stack([z_img, z_act])
            out = [model(x) for model in self.cnn_ensemble]

        return out
    
    def eval_score(  # type: ignore
        self, model_in: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes the squared error for the model over the given input/target.

        When model is not an ensemble, this is equivalent to
        `F.mse_loss(model(model_in, target), reduction="none")`. If the model is ensemble,
        then return is batched over the model dimension.

        This function returns no metadata, so the second output is set to an empty dict.

        Args:
            model_in (tensor): input tensor. The shape must be ``B x Id``, where `B`` and ``Id``
                batch size, and input dimension, respectively.
            target (tensor): target tensor. The shape must be ``B x Od``, where ``B`` and ``Od``
                represent batch size, and output dimension, respectively.

        Returns:
            (tensor): a tensor with the squared error per output dimension, batched over model.
        """
        with torch.no_grad():
            pred_mean, _ = self.forward(model_in, use_propagation=False)
            target = target.repeat((self.num_members, 1, 1))
            return F.mse_loss(pred_mean, target, reduction="none"), {}

    
    def _default_forward(self, x: torch.Tensor, only_elite = False):

         return self._forward_ensemble(x)

    def loss(self, batch: torch.Tensor) -> torch.Tensor:


        pass


    
    