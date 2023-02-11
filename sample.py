"""
Amostra de um modelo treinado
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Inicialização do modelo: 'resume' (de um diretório) ou uma variante de GPT-2 (por exemplo, 'gpt2-xl')
init_from = 'resume' 
# Diretório ignorado se init_from não for 'resume'
out_dir = 'out' 
# Início do prompt: "\n" ou "" ou etc. Pode ser especificado um arquivo, usado como: "FILE:prompt.txt"
start = "\n" 
# Número de amostras a serem geradas
num_samples = 10 
# Número de tokens gerados em cada amostra
max_new_tokens = 500 
# Temperatura: 1.0 = sem mudanças, < 1.0 = menos aleatório, > 1.0 = mais aleatório nas previsões
temperature = 0.8 
# Retenha apenas os top_k tokens mais prováveis, limitando outros para ter probabilidade 0
top_k = 200 
# Semente
seed = 1337
# Dispositivo: 'cpu' ou 'cuda' ou 'cuda:0' ou 'cuda:1' etc.
device = 'cuda' 
# Tipo de dado: 'float32' ou 'bfloat16' ou 'float16'
dtype = 'float16' 
# Usar PyTorch 2.0 para compilar o modelo para ser mais rápido
compile = False 
# Sobrescreve do comando de linha ou arquivo de configuração
exec(open('configurator.py').read()) 
# -----------------------------------------------------------------------------

# Configura a semente
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# Permitir tf32 nas operações de matmul e cudnn
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True 
# Tipo de dispositivo para uso futuro em torch.autocast
device_type = 'cuda' if 'cuda' in device else 'cpu' 
# Tipo de dado para uso futuro
ptdtype = {'float32': torch.float32, 'float16': torch.float16, 'float16': torch.float16}[dtype]
# Contexto para otimização
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Inicialização do modelo
if init_from == 'resume':
    # Inicialização a partir de um modelo salvo em um diretório específico
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # Inicialização a partir de uma determinada versão do GPT-2
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

# Configuração do modelo como avaliação
model.eval()
model.to(device)
# Compilação do modelo com PyTorch 2.0 (opcional)
if compile:
    model = torch.compile(model) # requer PyTorch 2.0


# procurar pelo arquivo pickle "meta" caso ele esteja disponível na pasta de dados
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # checkpoints antigos podem não ter esses dados...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO quero tornar isso mais geral para esquemas de codificação/decodificação arbitrários
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok, vamos assumir as codificações gpt-2 por padrão
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode o inicio do prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# começa a geração
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
