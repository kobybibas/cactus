import logging

import torch
from cornac.models import VAECF
from cornac.models.recommender import Recommender
from cornac.models.vaecf.vaecf import VAE, learn

logger = logging.getLogger(__name__)


class VAEWithBias(VAE):
    def __init__(self, z_dim, ae_structure, act_fn, likelihood):
        logger.info("VAEWithBias")
        super().__init__(z_dim, ae_structure, act_fn, likelihood)

        # Add bias
        num_items = ae_structure[0]
        self.item_bias = torch.nn.Embedding(num_items, 1)

    def decode(self, z):
        h = self.decoder(z)
        if self.likelihood == "mult":
            return torch.softmax(h + self.item_bias.weight.T, dim=1)
        else:
            raise NotImplementedError()
            return torch.sigmoid(h)


class VAECFWithBias(VAECF):
    def fit(self, train_set, val_set=None):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        self : object
        """
        Recommender.fit(self, train_set, val_set)

        self.device = (
            torch.device("cuda:0")
            if (self.use_gpu and torch.cuda.is_available())
            else torch.device("cpu")
        )

        if self.trainable:
            if self.seed is not None:
                torch.manual_seed(self.seed)
                torch.cuda.manual_seed(self.seed)

            if not hasattr(self, "vae"):
                data_dim = train_set.matrix.shape[1]
                self.vae = VAEWithBias(
                    self.k,
                    [data_dim] + self.autoencoder_structure,
                    self.act_fn,
                    self.likelihood,
                ).to(self.device)

            learn(
                self.vae,
                self.train_set,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                learn_rate=self.learning_rate,
                beta=self.beta,
                verbose=self.verbose,
                device=self.device,
            )

        elif self.verbose:
            logger.info("%s is trained already (trainable = False)" % (self.name))

        return self
