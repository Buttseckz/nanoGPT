# Configuração para treinamento do GPT-2 (124M) até uma perda muito boa de ~ 2,85 em 1 nó de 8X A100 40GB
# Inicie da seguinte maneira (por exemplo, em uma sessão de tela) e espere cerca de 5 dias:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'owt'
wandb_run_name='gpt2-124M'

# Isso faz com que o tamanho total do lote seja ~ 0,5M
# 12 tamanho do lote * 1024 tamanho do bloco * 5 gradaccum * 8 GPUs = 491.520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5

# Isso faz com que o número total de tokens seja 300B
max_iters = 600000
lr_decay_iters = 600000

# avaliar coisas
eval_interval = 1000
eval_iters = 200
log_interval = 10

# decaimento de peso
weight_decay = 1e-1
