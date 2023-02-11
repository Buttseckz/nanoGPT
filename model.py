"""
Definição completa de um Modelo de Linguagem GPT, tudo isso em um único arquivo.
Referências:
1) a implementação oficial do TensorFlow GPT-2 lançada pelo OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) a implementação PyTorch do huggingface/transformers:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# @torch.jit.script # bom habilitar quando não estiver usando torch.compile, desabilitar quando estiver usando (o padrão)
def new_gelu(x):
    """
    Implementação da função de ativação GELU atualmente no repositório Google BERT (idêntica ao OpenAI GPT).
    Referência: artigo Gaussian Error Linear Units (GELU): https://arxiv.org/abs/1606.08415

    Vamos desmembrar isso:

math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)) é o argumento da função tangente hiperbólica (tanh). É uma combinação de operações matemáticas que transformam a entrada x.

torch.tanh calcula a tangente hiperbólica do argumento passado, que é o resultado da expressão anterior.

1.0 + torch.tanh(...) adiciona 1 ao resultado da tangente hiperbólica, então o resultado agora está no intervalo [1, 2].

0.5 * x * (1.0 + torch.tanh(...)) multiplica a entrada x pelo resultado da operação anterior, efetivamente a escalando por 0,5. Este é o valor final da ativação, que é retornado.


    """

    # A linha de código está retornando o resultado da seguinte expressão:
    # 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

    # Primeiro, é calculado o valor de dentro do parenteses:
    # math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
    # É calculada a raiz quadrada de 2.0 / math.pi e multiplicado pelo resultado da soma:
    # x + 0.044715 * torch.pow(x, 3.0)
    # onde é calculado x elevado a 3.0

    # Em seguida, o resultado é passado como argumento para a função torch.tanh:
    # torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
    # A função torch.tanh retorna o hiperbólico tangente do argumento

    # O resultado da função torch.tanh é somado a 1.0:
    # 1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))

    # Finalmente, o resultado da soma é multiplicado por 0.5 * x:
    # 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

    # Este é o resultado final retornado pela função

    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class LayerNorm(nn.Module):
    """ LayerNorm mas com o bias opcional. PyTorch não suporta bias=False
     A classe LayerNorm é uma implementação do normalização de camada, que é uma técnica 
     de normalização de entrada de uma camada de uma rede neural. 
     Isso é importante para evitar o problema de explodir ou sumir gradiente,
       o que é comum em treinamentos de redes neurais profundas. A normalização de 
       camada também ajuda a estabilizar o treinamento e a acelerar a convergência."""
    def __init__(self, ndim, bias=True):
        # Inicialização da classe mãe (nn.Module)
        super().__init__()
        # Cria um parâmetro para os pesos do layer normalization com valores iniciados com 1
        self.weight = nn.Parameter(torch.ones(ndim))
        # Cria um parâmetro para o viés do layer normalization com valores iniciados com 0, ou nulo se bias=False
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        # Aplica o layer normalization no tensor de entrada
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Projeções para chave, consulta e valor para todas as cabeças, mas em lote
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Projeção de saída
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Regularização
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # Atenção flash faz o GPU ir rápido, mas suporte só está disponível no PyTorch nightly e ainda é um pouco assustador
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and self.dropout == 0.0
        if not self.flash:
            print("WARNING: usando atenção lenta. Atenção Flash atualmente precisa do PyTorch nightly e dropout=0.0")
            # máscara causal para garantir que a atenção é aplicada apenas à esquerda na seqüência de entrada
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # tamanho do lote, comprimento da sequência, dimensionalidade de embreagem (n_embd)

        # calcule a consulta, a chave, os valores para todas as cabeças no lote e mova a cabeça para a frente para ser a dimensão do lote
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    
    #Atribuição causal de auto; Auto-atenção: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # atenção eficiente usando kernels CUDA Flash Attention
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        else:
            # implementação manual de atenção
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # reúne todos os resultados das cabeças lado a lado
        # projeção de saída
        y = self.resid_dropout(self.c_proj(y))
        return y


# Comentários sobre o código:
class MLP(nn.Module):
    # Construtor da classe MLP, que herda as propriedades da classe nn.Module
    def __init__(self, config):
        # Chama o construtor da classe mãe (nn.Module)
        super().__init__()
        # Adiciona a primeira camada totalmente conectada (Linear) com pesos que ligam as "config.n_embd" entradas à saída com 4 * "config.n_embd" unidades
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # Adiciona a segunda camada totalmente conectada (Linear) com pesos que ligam as "4 * config.n_embd" entradas à saída com "config.n_embd" unidades
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        # Adiciona uma camada de Dropout com taxa de dropout "config.dropout"
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        # Aplica a primeira camada linear à entrada "x"
        x = self.c_fc(x)
        # Aplica a função new_gelu à entrada "x"
        x = new_gelu(x)
        # Aplica a segunda camada linear à entrada "x"
        x = self.c_proj(x)
        # Aplica a camada de dropout à entrada "x"
        x = self.dropout(x)
        # Retorna o resultado "x"
        return x


# Definição da classe Block
class Block(nn.Module):
    # Método de inicialização
    def __init__(self, config):
        # Chamar o método de inicialização da classe pai
        super().__init__()        
        # Inicializar a normalização de camada com o número especificado de vetores incorporados
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)        
        # Inicializar a auto-atenção causal com a configuração especificada
        self.attn = CausalSelfAttention(config)        
        # Inicializar a normalização de camada com o número especificado de vetores incorporados
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)        
        # Inicializar a perceptron de múltiplas camadas com a configuração especificada
        self.mlp = MLP(config)    
    # Definir a passagem para a frente
    def forward(self, x):
        # Aplicar a normalização de camada e auto-atenção
        x = x + self.attn(self.ln_1(x))        
        # Aplicar a normalização de camada e perceptron de múltiplas camadas
        x = x + self.mlp(self.ln_2(x))        
        # Retornar o resultado
        return x

# Definição da classe GPTConfig usando dataclass
@dataclass
class GPTConfig:
    # Tamanho do bloco
    block_size: int = 1024    
    # Tamanho do vocabulário
    vocab_size: int = 50304 # Nota: GPT-2 tem um tamanho de vocabulário de 50257, preenchido até o múltiplo mais próximo de 64 para eficiência
    # Número de camadas
    n_layer: int = 12    
    # Número de cabeças
    n_head: int = 12    
    # Número de vetores incorporados
    n_embd: int = 768    
    # Taxa de dropout
    dropout: float = 0.0    
    # Booleano que indica se incluir viés nas camadas lineares e de normalização de camadas
    bias: bool = True 
    # Nota: Verdadeiro: viés incluído, como no GPT-2. False: ligeiramente melhor e mais rápido.


# Esta é a definição da classe GPT
class GPT(nn.Module):
    # Método de inicialização da classe GPT
    def __init__(self, config):
        # Chame o método de inicialização da classe pai
        super().__init__()
        # Assegure que a configuração tenha tamanho de vocabulário e tamanho de bloco definidos
        assert config.vocab_size is not None
        assert config.block_size is not None
        # Armazene a configuração
        self.config = config

        # Crie um nn.ModuleDict para armazenar os diferentes componentes do modelo
        self.transformer = nn.ModuleDict(dict(
            # Crie uma camada de embedding de palavras usando o tamanho de vocabulário e o número de dimensões de embedding especificados
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # Crie uma camada de embedding de posição usando o tamanho de bloco e o número de dimensões de embedding especificados
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # Crie uma camada de dropout com a taxa de dropout especificada
            drop = nn.Dropout(config.dropout),
            # Crie uma lista de blocos, com o número de blocos determinado pelo número de camadas na configuração
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # Crie uma camada de normalização de camada
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        # Crie uma camada linear para ser usada como cabeça do modelo de linguagem
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Vincule os pesos da camada de embedding de palavra e da cabeça do modelo de linguagem
        self.transformer.wte.weight = self.lm_head.weight

        # Inicialize todos os pesos do modelo usando o método _init_weights
        self.apply(self._init_weights)
        # Aplique uma inicialização escalonada especial para as projeções residuais, conforme especificado no papel do GPT-2
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # Imprima o número de parâmetros no modelo
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))


    def get_num_params(self, non_embedding=True):
        """
        Esta função retorna o número de parâmetros do modelo.
        Por padrão, a contagem não inclui os embeddings de posição.
        Já os embeddings de token são incluídos, pois devido ao compartilhamento de parâmetros,
        eles são usados como pesos na camada final.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """
        Inicialização dos pesos do modelo.
        Se o módulo é uma camada linear, inicializa os pesos com uma distribuição normal
        com média 0.0 e desvio padrão 0.02 e zera os biases, se houver.
        Se o módulo é um embedding, inicializa os pesos com uma distribuição normal
        com média 0.0 e desvio padrão 0.02.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Função forward da rede GPT.
        Recebe como entrada o índice dos tokens na sequência e, opcionalmente, os alvos desejados.
        Calcula as embeddings de token e posição, adiciona os dois e passa pelas camadas do modelo.
        Se houver alvos desejados, calcula a loss. Caso contrário, realiza uma mini-otimização
        na inferência, passando somente a última posição pelo lm_head.
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # avançar o modelo GPT em si
        tok_emb = self.transformer.wte(idx) # token embeddings  shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings  shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # se formos dados alguns alvos desejados, também calculamos a perda
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # mini-otimização no tempo de inferência: apenas avance lm_head na última posição
            logits = self.lm_head(x[:, [-1], :]) # nota: usando lista [-1] para preservar a dimensão do tempo
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # cirurgia no modelo para diminuir o tamanho do bloco, se necessário
        # por exemplo, podemos carregar o checkpoint do modelo pré-treinado do GPT2 (tamanho de bloco 1024)
        # mas queremos usar um tamanho de bloco menor para algum modelo menor e mais simples
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        # Verifica se o tipo de modelo está dentro dos modelos disponíveis
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        # Se não for fornecido um dicionário de argumentos, ele será inicializado como um dicionário vazio
        override_args = override_args or {} 
        # Somente o dropout pode ser substituído, veja mais notas abaixo
        assert all(k == 'dropout' for k in override_args)
        # Importa o modelo da biblioteca transformers
        from transformers import GPT2LMHeadModel
        print("Carregando pesos de um modelo GPT pré-treinado: %s" % model_type)

        # n_layer, n_head e n_embd são determinados pelo tipo de modelo
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("Forçando vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # sempre 50257 para pontos de verificação do modelo GPT
        config_args['block_size'] = 1024 # sempre 1024 para pontos de verificação do modelo GPT
        config_args['bias'] = True # sempre verdadeiro para pontos de verificação do modelo GPT
        # Podemos substituir a taxa de dropout, se desejado
        if 'dropout' in override_args:
            print(f"Substituindo a taxa de dropout para {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # Cria um modelo minGPT inicializado do zero
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        # Descarta esta máscara/buffer, que não é um parâmetro
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] 

        # Inicializa um modelo da biblioteca huggingface/transformers
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

       # Copiar enquanto garante que todos os parâmetros estejam alinhados e combinem em nomes e formas
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignorar estes, apenas um buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # mesmo, apenas a máscara (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basicamente, os pontos de verificação openai usam um módulo "Conv1D", mas queremos apenas usar uma Linear vanilla
        # isso significa que precisamos transpor esses pesos ao importá-los
        assert len(sd_keys_hf) == len(sd_keys), f"chaves desigualadas: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # tratamento especial para os pesos Conv1D que precisamos transpor
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # cópia simples sobre os outros parâmetros
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        
        return model


    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Esta função longa, infelizmente, está fazendo algo muito simples e sendo muito defensiva:
        Estamos separando todos os parâmetros do modelo em dois grupos: aqueles que experimentarão
        decaimento de peso para regularização e aqueles que não (viés e pesos layernorm/embedding).
        Em seguida, retornamos o objeto otimizador PyTorch.
        """

        # separa todos os parâmetros em aqueles que terão e não terão decaimento de peso regularizador
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # nome completo do parâmetro
                # nota aleatória: porque named_modules e named_parameters são recursivos
                # veremos os mesmos tensores p muitas e muitas vezes. mas fazendo isso desta maneira
                # nos permite saber a qual módulo pai pertence qualquer tensor p...
                if pn.endswith('bias'):
                    # todos os viés não serão decaídos
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # pesos dos módulos da whitelist serão decaídos de peso
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # pesos dos módulos da blacklist NÃO serão decaídos de peso
                    no_decay.add(fpn)

        # sutil: 'transformer.wte.weight' e 'lm_head.weight' estão ligados, então eles
        # aparecerão nos conjuntos de decaimento e não decaimento, respectivamente, após o acima.
        # Além disso, porque named_parameters() não retorna duplicatas, ele
        # só retornará a primeira ocorrência, chaveada por 'transformer.wte.weight', abaixo.
        # então vamos remover manualmente 'lm_head.weight' do conjunto de decaimento. Isso incluirá
        # este tensor na otimização via transformer.wte.weight apenas e não decaiu.
        decay.remove('lm_head.weight')


        # Valida se todos os parâmetros foram considerados
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "os parâmetros %s foram colocados nas configurações de decay/no_decay!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "os parâmetros %s não foram separados nas configurações decay/no_decay!" \
                                                    % (str(param_dict.keys() - union_params), )

        # Cria o objeto otimizador do PyTorch
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # A nova versão noturna do PyTorch tem uma nova opção 'fused' para AdamW que é muito mais rápida
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"usando AdamW fundido: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        Estima a utilização do modelo em FLOPs (MFU) em unidades de pico de FLOPs bfloat16 A100.
        """
        # Primeiro, estimamos o número de FLOPs por iteração.
        # Veja o Apêndice B do PaLM como referência: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # Expressamos o desempenho de FLOPs como a razão dos FLOPs de pico bfloat16 A100.
        flops_achieved = flops_per_iter * (1.0/dt) # por segundo
        flops_promised = 312e12 # FLOPs de pico GPU bfloat16 A100 é 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Tome uma sequência de condicionamento de índices idx (LongTensor de forma (b,t)) e complete
        a sequência max_new_tokens vezes, alimentando as previsões de volta ao modelo a cada vez.
        Provavelmente, você vai querer ter certeza de estar no modo de operação model.eval() para isso.
        """
        for _ in range(max_new_tokens):
            # se o contexto da sequência estiver ficando muito longo, devemos cortá-lo no tamanho do bloco
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # faça o modelo avançar para obter os logits para o índice na sequência
            logits, _ = self(idx_cond)
            # retire os logits no passo final e os escale pela temperatura desejada
            logits = logits[:, -1, :] / temperature
            # opcionalmente, corte os logits para as top k opções
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))                
                logits[logits < v[:, [-1]]] = -float('Inf')
            #Aplique softmax para converter logits em probabilidades (normalizadas)    
            probs = F.softmax(logits, dim=-1)
            # amostre a partir da distribuição
            idx_next = torch.multinomial(probs, num_samples=1)
            # anexe o índice amostrado à sequência em execução e continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

