# Avaliar o GPT2 base
# n_camada=12, n_head=12, n_embd=768
# 124M parâmetros
batch_size = 8
eval_iters = 500 # usar mais iterações para obter uma boa estimativa
eval_only = True
wandb_log = False
init_from = 'gpt2'
