### Uso de wandb
1. Hacer una cuenta en wandb y guardar una copia de su [api key](https://wandb.ai/authorize)
[https://wandb.ai/home](https://wandb.ai/home)
2. Instalar wandb y tqdm
```
conda activate mlenv
pip install wandb
pip install tqdm
```

3. En python loggearse con su api key

```
import wandb

your_api_key = "your_wandb_api_key_here" # Replace with your actual API key
wandb.login(key=your_api_key)
```

No dejen su api key en el repositorio de github ya que es público y si alguien mas lo usa puede loggear información en su cuenta.

4. Correr wandb_usage.py

5. Ir a wandb y visualizar su proyecto en la página que se indica