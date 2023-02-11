# avaliar o gpt2 base
# n_layer=48, n_head=25, n_embd=1600
# 1558M parâmetros
batch_size = 8
eval_iters = 500 # use mais iterações para obter uma boa estimativa
eval_only = True
wandb_log = False
init_from = 'gpt2-xl'
