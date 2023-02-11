"""
Uma versão muito mais curta do train.py para benchmarking
"""
import os
from contextlib import nullcontext
import numpy as np
import time
import torch
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
batch_size = 12
block_size = 1024
bias = False
real_data = True
seed = 1337
device = 'cuda' # exemplos: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' # 'float32' ou 'bfloat16' ou 'float16'
compile = True # usar PyTorch 2.0 para compilar o modelo para ser mais rápido
profile = False # usar o perfilador do pytorch, ou apenas benchmarking simples?
exec(open('configurator.py').read()) # sobrescreve a partir da linha de comando ou arquivo de configuração
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # permitir tf32 na matmul
torch.backends.cudnn.allow_tf32 = True # permitir tf32 no cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # para uso posterior em torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Inicialização de carregamento de dados
if real_data:
    dataset = 'openwebtext'
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    def get_batch(split):
#Esta é uma função "get_batch", que serve para obter um lote de dados de treinamento.
        data = train_data # Aqui é atribuído a variável "data" os dados de treinamento. Observe que a divisão do conjunto de dados é ignorada neste script de benchmarking.
        ix = torch.randint(len(data) - block_size, (batch_size,)) # Aqui é gerado um índice aleatório dentro dos limites dos dados de treinamento (exceto pelo tamanho do bloco).
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix]) # Aqui é criado uma tensor PyTorch a partir dos dados de treinamento, utilizando o índice gerado anteriormente.
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix]) # Aqui é criado outro tensor PyTorch a partir dos dados de treinamento, utilizando o índice gerado anteriormente e o próximo item.
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True) # Aqui os tensores "x" e "y" são carregados na memória do dispositivo e são convertidos para serem usados pelo dispositivo.
        return x, y # Por fim, a função "get_batch" retorna os tensores "x" e "y".
else:
    # alternativamente, se desejar dados fixos para não se preocupar com o carregamento de dados
    x = torch.randint(50304, (batch_size, block_size), device=device)
    y = torch.randint(50304, (batch_size, block_size), device=device)
    get_batch = lambda split: (x, y)

# inicialização do modelo
gptconf = GPTConfig(
    block_size = block_size, # até onde o modelo olha para trás? isto é, tamanho do contexto
    n_layer = 12, n_head = 12, n_embd = 768, # tamanho do modelo
    dropout = 0, # para determinismo
    bias = bias,
)

model = GPT(gptconf)
model.to(device)

optimizer = model.configure_optimizers(weight_decay=1e-2, learning_rate=1e-4, betas=(0.9, 0.95), device_type=device_type)

if compile:
    print("Compilando modelo...")
    model = torch.compile(model) # pytorch 2.0

if profile:
    # Documentação sobre o perfilador do PyTorch
    # - tutorial: https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
    # - API: https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile
    wait, warmup, active = 5, 5, 5 # Tempos de espera, aquecimento e atividade para o perfilador
    num_steps = wait + warmup + active # Número total de etapas para o perfilador
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], # Atividades a serem perfiladas
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1), # Agendamento para o perfilador
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./bench_log'), # Evento disparado quando o perfilador estiver pronto
        record_shapes=False, # Não gravar formas das operações
        profile_memory=False, # Não perfilar a memória
        with_stack=False, # Não causar overhead, desabilitar se necessário
        with_flops=True, # Gravar operações FLOPs
        with_modules=False, # Apenas para modelos torchscript por enquanto
    ) as prof: # Início do bloco de perfilamento
        
        X, Y = get_batch('train') # Obter lote de treinamento
        for k in range(num_steps): # Para cada etapa
            with ctx:
                logits, loss = model(X, Y) # Executar a passada do modelo
            X, Y = get_batch('train') # Obter novo lote de treinamento
            optimizer.zero_grad(set_to_none=True) # Zera gradientes
            loss.backward() # Executar backpropagation
            optimizer.step() # Atualizar os pesos
            lossf = loss.item() # Obter valor da perda
            print(f"{k}/{num_steps} loss: {lossf:.4f}") # Imprimir informações sobre a perda

            prof.step() # Notificar o perfilador ao final de cada etapa
else:

    # simple benchmarking
    torch.cuda.synchronize()
# Sincroniza todas as operações pendentes na GPU antes de iniciar o benchmark
    for stage, num_steps in enumerate([10, 20]): # burnin, then benchmark
        # O loop itera duas vezes: uma vez para o aquecimento (burn-in) e outra para o benchmark.
        # stage é o número de iterações (0 ou 1), e num_steps é o número de etapas em cada iteração.
        t0 = time.time()
        X, Y = get_batch('train')
        # obtém um lote de dados para treinar o modelo
        for k in range(num_steps):
            with ctx:
                logits, loss = model(X, Y)
            # executa a foward pass do modelo no lote de dados
            X, Y = get_batch('train')
            # obtém o próximo lote de dados
            optimizer.zero_grad(set_to_none=True)
            # zera os gradientes acumulados
            loss.backward()
            # calcula o gradiente da perda com relação aos pesos
            optimizer.step()
            # atualiza os pesos do modelo com base nos gradientes
            lossf = loss.item()
            # pega o valor escalar da perda
            print(f"{k}/{num_steps} loss: {lossf:.4f}")
            # exibe o valor da perda a cada etapa
        torch.cuda.synchronize()
        t1 = time.time()
        # sincroniza as operações pendentes na GPU antes de medir o tempo
        dt = t1-t0
        # tempo total gasto para realizar as etapas do modelo
        mfu = model.estimate_mfu(batch_size * 1 * num_steps, dt)
        # estimativa da taxa de operações de ponto flutuante por segundo (FLOPs)
        if stage == 1:
            # imprime os resultados somente na segunda iteração (benchmark)
            print(f"time per iteration: {dt/num_steps*1000:.4f}ms, MFU: {mfu*100:.2f}%")
            # tempo médio por iteração e a taxa MFU em porcentagem

