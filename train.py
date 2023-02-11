"""
Este script de treinamento pode ser executado tanto em um único GPU no modo de depuração,
e também em uma execução de treinamento maior com paralelização de dados distribuídos (ddp).

Para executar em um único GPU, exemplo:
$ python train.py --batch_size=32 --compile=False

Para executar com DDP em 4 GPUs em 1 nó, exemplo:
$ torchrun --standalone --nproc_per_node=4 train.py

Para executar com DDP em 4 GPUs em 2 nós, exemplo:
- Execute no primeiro (mestre) nó com exemplo de IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Execute no nó de trabalho:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(Se sua classe não tiver interconexão Infiniband, adicione NCCL_IB_DISABLE=1)

"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Padrões de configuração para treinar um gpt2 (124M) no OpenWebText
# I / O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # se True, o script sai logo após a primeira avaliação
always_save_checkpoint = True # se True, sempre salva um ponto de verificação após cada avaliação
init_from = 'scratch' # 'scratch' ou 'resume' ou 'gpt2 *'
# registro do wandb
wandb_log = False # desativado por padrão
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str (time.time ())
# dados
dataset = 'openwebtext'
gradient_accumulation_steps = 1 # usado para simular tamanhos de lote maiores
batch_size = 12 # se gradient_accumulation_steps> 1, este é o tamanho do micro-batch
block_size = 1024
# modelo
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # para pré-treinamento 0 é bom, para ajuste fino experimente 0,1 +
bias = False # usamos viés dentro das camadas LayerNorm e Linear?
# otimizador adamw
learning_rate = 6e-4 # taxa máxima de aprendizado
max_iters = 600000 # número total de iterações de treinamento
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clipe gradientes neste valor ou desative se == 0,0
# configurações de decaimento da taxa de aprendizado
decay_lr = True # se deve decair a taxa de aprendizado
warmup_iters = 2000 # quantos passos para aquecer
lr_decay_iters = 600000 # deve ser ~ = max_iters por Chinchilla
min_lr = 6e-5 # taxa mínima de aprendizado, deve ser ~ = learning_rate / 10 por Chinchilla
# configurações DDP
backend = 'nccl' # 'nccl', 'gloo', etc.
# sistema
device = 'cuda' # exemplos: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., ou tente 'mps' em MacBooks
dtype = 'bfloat16' # 'float32', 'bfloat16' ou 'float16', o último implementará automaticamente um GradScaler
compile = True # use o PyTorch 2.0 para compilar o modelo para ser mais rápido
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # override pela linha de comando
config = {k: globals()[k] for k in config_keys} # logging
# -----------------------------------------------------------------------------

# Inicializações diversas, atributos derivados, configuração de I/O
ddp = int(os.environ.get('RANK', -1)) != -1 # É uma execução ddp?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # Este processo fará log, checkpointing, etc.
    seed_offset = ddp_rank # Cada processo tem uma semente diferente
else:
    # Se não for ddp, estamos executando em uma GPU única e um processo
    master_process = True
    seed_offset = 0

# configurações para o processo mestre e dispositivo
if master_process:
    os.makedirs(out_dir, exist_ok=True) # cria o diretório de saída, se não existir

torch.manual_seed(1337 + seed_offset) # define a semente com base no offset de semente
torch.backends.cuda.matmul.allow_tf32 = True # permite tf32 na multiplicação de matrizes
torch.backends.cudnn.allow_tf32 = True # permite tf32 na operação cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # tipo de dispositivo para uso posterior em torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.float16, 'float16': torch.float16}[dtype] # tipo de dado pytorch
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype) # contexto de autocast


# tentativa de data loader
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, o que nos permite mover dados para a GPU async (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# override com init_from='resume' (checkpoint)
iter_num = 0
best_val_loss = 1e9

# tenta derivar o vocab_size do dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")


# Inicializações diversas, atributos derivados e configurações de I/O
ddp = int(os.environ.get('RANK', -1)) != -1 # este é um processo ddp?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # este processo fará log, salvamento de checkpoint, etc.
    seed_offset = ddp_rank # cada processo obtém uma semente diferente
else:
    # se não for ddp, estamos executando em uma única GPU e em um único processo
    master_process = True
    seed_offset = 0

# Inicialização de um novo modelo ou continuidade do treinamento a partir de um checkpoint
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # comece com os argumentos do modelo da linha de comando
if init_from == 'scratch':
    # inicialização de um novo modelo a partir do zero
    print("Inicializando um novo modelo a partir do zero")
    # determine o tamanho do vocabulário que usaremos para o treinamento a partir do zero
    if meta_vocab_size is None:
        print("usando o padrão de tamanho de vocabulário do GPT-2 com 50304 (arredondado para 50257 para eficiência)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Retomando o treinamento de {out_dir}")
    # retomada do treinamento a partir de um checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # força esses atributos de configuração a serem iguais, caso contrário não poderemos retomar o treinamento
    # o resto dos atributos (por exemplo, dropout) pode ficar como desejado da linha de comando
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])


if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrapper para DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# escalonador de decaimento da taxa de aprendizado (cosseno com aquecimento)
def get_lr(it):
    # 1) aquecimento linear por warmup_iters etapas
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) se it > lr_decay_iters, retorna a taxa de aprendizado mínima
    if it > lr_decay_iters:
        return min_lr
    # 3) entre eles, usa o decaimento cosseno até a taxa de aprendizado mínima
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # o coeff varia de 0 a 1
    return min_lr + coeff * (learning_rate - min_lr)


# registro
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# loop de treinamento
X, Y = get_batch('train') # buscar a primeira porção
t0 = time.time()
local_iter_num = 0 # número de iterações na vida deste processo
raw_model = model.module if ddp else model # desembrulhar o contêiner DDP se necessário
running_mfu = -1.0
while True:

    # determina e seta a learning rate
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # avalia a perda em train/val sets e salva checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # converte para %
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # Atualização forward backward, com acumulação opcional de gradientes para simular tamanho de lote maior
    # e usando o GradScaler se o tipo de dados for float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # no treinamento DDP, precisamos sincronizar os gradientes apenas no último micro passo.
            # a maneira oficial de fazer isso é com o gerenciador de contexto model.no_sync(), mas
            # eu realmente não gosto disso porque isso incha o código e nos obriga a repetir o código
            # olhando para a fonte desse gerenciador de contexto, ele apenas alterna esta variável
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
        # busque imediatamente o próximo lote de forma assíncrona enquanto o modelo está fazendo a passagem forward na GPU
        X, Y = get_batch('train')
        # passagem backward, com escalonamento de gradiente se estiver treinando em fp16
        scaler.scale(loss).backward()
    # clipe o gradiente
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # passe o otimizador e o escalonador se estiver treinando em fp16
    scaler.step(optimizer)
    scaler.update()
    # esvazie os gradientes assim que pudermos, não precisamos mais desse memória
    optimizer.zero_grad(set_to_none=True)

    # tempo e log
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() # perda como float. note: este é um ponto de sincronização CPU-GPU
        if local_iter_num >= 5: # deixe o loop de treinamento se estabelecer um pouco
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1
    
    # condições de término
    if iter_num > max_iters:
        break
    
    if ddp:
        destroy_process_group()
    