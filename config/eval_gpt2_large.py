# Avaliação do gpt2 base
# n_layer=36, n_head=20, n_embd=1280
# 774M de parâmetros
batch_size = 8
eval_iters = 500 # usar mais iterações para obter uma boa estimativa
eval_only = True
wandb_log = False
init_from = 'gpt2-large'
