## Car Racing
### Configuración
Para poder monitorear sus experimentos en este proyecto, les recomiendo ampliamente el uso de weights and biases, van a poder ver en tiempo real las gráficas de entrenamiento y podrán descargarlas ahi mismo. 

1. Configuren una cuenta en [wandb](https://wandb.ai/trial_end)
2. Instalen wandb e ingresen sus credenciales (las pueden obtener de wandb, el api key te lo dan ahi mismo)
```
pip install wandb weave
wandb login
```

## DQN
Pueden utilizar como base el tutorial de pytorch sobre [DQN](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) o ver [ejemplos de como implementarlo](https://www.youtube.com/watch?v=qfovbG84EBg)
Consideren que ustedes tienen un problema distinto ya que en el car racing usarán como estado *la imagen* por lo que tienen que considerar otros aspectos (e.g. ¿cómo proveerle al agente información sobre la velocidad usando solo imágenes?)