import time

out_dir = 'out-shakespeare'
eval_interval = 5
eval_iters = 40
wandb_log = False # sentir-se livre para ligar
wandb_project = 'shakespeare'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'shakespeare'
init_from = 'gpt2-xl' # este é o maior modelo GPT-2 

# salvar checkpoints somente se a perda de validação melhorar
always_save_checkpoint = False

# o número de exemplos por iteração:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare tem 301.966 tokens, então 1 época ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 20

# afinar em LR constante
learning_rate = 3e-5
decay_lr = False
