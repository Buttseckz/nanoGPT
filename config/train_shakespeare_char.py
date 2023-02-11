# Treinando um modelo miniatura de caracteres baseado em Shakespeare
# bom para debugar e brincar em MacBooks e similares

out_dir = 'out-shakespeare-char'
eval_interval = 250 # Mantenha frequente porque vamos sobreajustar
eval_iters = 200
log_interval = 10 # Não imprima muito frequentemente

# Esperamos sobreajustar neste pequeno conjunto de dados, então apenas salve quando val melhore
always_save_checkpoint = False

wandb_log = True # Sobrepor via linha de comando se você quiser
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt'

dataset = 'shakespeare_char'
batch_size = 64
block_size = 256 # Contexto de até 256 caracteres anteriores

# Modelo baby GPT :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # Com redes baby, podemos permitir ir um pouco mais alto
max_iters = 5000
lr_decay_iters = 5000 # Faça igual a max_iters geralmente
min_lr = 1e-4 # learning_rate / 10 geralmente
beta2 = 0.99 # Faça um pouco maior porque o número de tokens por iteração é pequeno

warmup_iters = 100 # Não é muito necessário potencialmente

# No MacBook também adicione
# device = 'cpu'  # execute apenas no CPU
# compile = False # não compile o modelo com torch
