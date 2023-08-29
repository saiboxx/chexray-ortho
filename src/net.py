from __future__ import annotations
from typing import Sequence, Tuple, Optional
import warnings

import numpy as np
import pytorch_lightning as L
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from pytorch_lightning.loggers import CSVLogger
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torch import nn
from torch.utils.data import Dataset, DataLoader


class TensorDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.tensor(x_data, dtype=torch.float32)
        self.y_data = torch.tensor(y_data, dtype=torch.float32)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]


class ClassificationModule(L.LightningModule):
    def __init__(self, model: nn.Module, loss_func: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.loss_func = loss_func

    def forward(self, x):
        return self.model(x).squeeze()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_func(y_pred, y)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_func(y_pred, y)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


def train_linear_classifier(
    x_train: np.ndarray,
    y_train: Sequence,
    x_val: Tensor,
    y_val: Sequence,
    max_epochs: int = 3,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    # Create PyTorch datasets and data loaders
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2 * batch_size)

    # Initialize the PyTorch Lightning model
    model = ClassificationModule(x_train.shape[1])

    # Initialize the PyTorch Lightning Trainer
    warnings.filterwarnings('ignore', category=PossibleUserWarning)
    trainer = L.Trainer(logger=CSVLogger('lightning_logs'), max_epochs=max_epochs)

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Test the model
    model.eval()
    # Remove shuffle for train predictions
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    res = {}
    with torch.no_grad():
        for loader, name in [(train_loader, 'train'), (val_loader, 'val')]:
            y_pred_list = []
            y_true_list = []
            for x_batch, y_batch in loader:
                y_pred = model(x_batch)
                y_pred_list.extend(y_pred.detach().numpy())
                y_true_list.extend(y_batch.numpy())

            res[name] = y_pred_list

            # Calculate AUROC using scikit-learn
            auroc = roc_auc_score(y_true_list, y_pred_list)
            print(f'{name} AUROC: {auroc:.4f}')

    return np.asarray(res['train']), np.asarray(res['val'])


class NNClassifier:
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        loss_func: Optional[nn.Module] = None,
        max_epochs: int = 3,
        batch_size: int = 256,
    ) -> None:
        if model is None:
            warnings.filterwarnings('ignore', category=UserWarning)
            self.model = nn.LazyLinear(1)
        else:
            self.model = model

        self.loss_func = nn.BCEWithLogitsLoss() if loss_func is None else loss_func
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(
        self,
        x_train: np.ndarray,
        y_train: Sequence,
    ) -> NNClassifier:
        # Create PyTorch datasets and data loaders
        ds = TensorDataset(x_train, y_train)

        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        # Initialize the PyTorch Lightning model
        module = ClassificationModule(self.model, self.loss_func)

        # Initialize the PyTorch Lightning Trainer
        warnings.filterwarnings('ignore', category=PossibleUserWarning)
        trainer = L.Trainer(
            logger=CSVLogger('lightning_logs'),
            max_epochs=self.max_epochs,
            enable_model_summary=False,
            enable_progress_bar=False,
        )

        # Train the model
        trainer.fit(module, loader)

        return self

    @torch.no_grad()
    def decision_function(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()

        x = torch.tensor(x).to(self.device).float()
        model = self.model.to(self.device)

        preds = []
        for x_batch in torch.chunk(x, chunks=x.shape[0] // self.batch_size * 4):
            preds.append(model(x_batch))
        return torch.cat(preds).cpu().numpy()


    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    @torch.no_grad()
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()

        x = torch.tensor(x).to(self.device).float()
        model = self.model.to(self.device)

        preds = []
        for x_batch in torch.chunk(x, chunks=x.shape[0] // self.batch_size * 4):
            preds.append(model(x_batch))
        probs = torch.sigmoid(torch.cat(preds))
        # Tedious restructuring to align with sklearn API
        return torch.cat([1 - probs, probs], dim=1).cpu().numpy()
