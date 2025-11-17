import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pathlib
from torchvision.models import resnet18, ResNet18_Weights

file_path = pathlib.Path(__file__).parent.absolute()


def build_backbone(model="resnet18", pretrained=True, freeze=True, last_n_layers=2):
    if model == "resnet18":
        if pretrained:
            weights_param = ResNet18_Weights.DEFAULT
        else:
            weights_param = None
            freeze=False

        backbone = resnet18(weights=weights_param)

        if freeze:
            for param in backbone.parameters():
                param.requires_grad = False
        
        # quitar ultima capa para mejor transfer learning
        out_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        return backbone, out_dim
    else:
        raise Exception(f"Model {model} not supported")


class Network(nn.Module):
    def __init__(self, input_dim: int, n_classes: int) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # TODO: Calcular dimension de salida
        # out_dim = ...
        self.backbone, out_dim = build_backbone(pretrained=True, freeze=True)

        # TODO: Define las capas de tu red
        self.head = nn.Sequential(
            nn.Linear(out_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )

        self.to(self.device)

    def calc_out_dim(self, in_dim, kernel_size, stride=1, padding=0):
        out_dim = math.floor((in_dim - kernel_size + 2 * padding) / stride) + 1
        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Define la propagacion hacia adelante de tu red
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        features = self.backbone(x)
        logits = self.head(features)
        proba = F.softmax(logits, dim=1)
        
        return logits, proba

    def predict(self, x):
        self.eval()
        with torch.inference_mode():
            return self.forward(x)

    def save_model(self, model_name: str):
        """
        Guarda el modelo en el path especificado
        args:
        - net: definición de la red neuronal (con nn.Sequential o la clase anteriormente definida)
        - path (str): path relativo donde se guardará el modelo
        """
        models_path = file_path / "models" / model_name
        if not models_path.parent.exists():
            models_path.parent.mkdir(parents=True, exist_ok=True)
        # TODO: Guarda los pesos de tu red neuronal en el path especificado
        torch.save(self.state_dict(), models_path)

    def load_model(self, model_name: str):
        """
        Carga el modelo en el path especificado
        args:
        - path (str): path relativo donde se guardó el modelo
        """
        # TODO: Carga los pesos de tu red neuronal
        models_path = file_path / "models" / model_name
        if models_path.exists():
            print(f"Cargando modelo desde: {models_path}")
            self.load_state_dict(torch.load(models_path, map_location=self.device))
        else:
            print(f"No se encontró el archivo del modelo en: {models_path}")