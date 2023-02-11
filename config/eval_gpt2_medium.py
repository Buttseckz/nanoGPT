# Avalie o GPT2 base
# n_layer = 24, n_head = 16, n_embd = 1024
# 350M de parâmetros
batch_size = 8
eval_iters = 500 # usar mais iterações para obter uma boa estimativa
eval_only = True
wandb_log = False
init_from = 'gpt2-medium'
