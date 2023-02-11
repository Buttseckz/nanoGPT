
# nanoGPT

![nanoGPT](assets/nanogpt.jpg)

O repositório mais simples e mais rápido para treinamento/afinação de GPTs de tamanho médio. É uma reescrita do minGPT que prioriza a performance sobre a educação. Ainda está em desenvolvimento ativo, mas atualmente o arquivo train.py reproduz o GPT-2 (124M) no OpenWebText, rodando em um único nó 8XA100 40GB em cerca de 4 dias de treinamento. O código em si é simples e legível: o `train.py` é um loop de treinamento de ~300 linhas e o `model.py` é uma definição de modelo GPT de ~300 linhas, que pode opcionalmente carregar os pesos do GPT-2 do OpenAI. É isso.

![repro124m](assets/gpt2_124M_loss.png)

Como o código é tão simples, é muito fácil personalizá-lo de acordo com suas necessidades, treinar novos modelos do zero ou afinar pontos de verificação pré-treinados (por exemplo, o maior disponível atualmente como ponto de partida seria o modelo GPT-2 1.3B do OpenAI).

## install

Dependencias:

- [pytorch](https://pytorch.org) 
- [numpy](https://numpy.org/install/) 
- `pip install transformers` 
- `pip install datasets` 
- `pip install tiktoken` 
- `pip install wandb` 
- `pip install tqdm`

## quick start

Se você não é um profissional de aprendizado profundo e apenas quer sentir a mágia e mergulhar, a maneira mais rápida de começar é treinar um GPT de nível de caractere nas obras de Shakespeare. Primeiro, baixamos como um único arquivo (1MB) e o transformamos de texto bruto em uma grande corrente de números:

```
$ python data/shakespeare_char/prepare.py
```

Isso cria um train.bin e um val.bin no diretório de dados. Agora é hora de treinar seu GPT. O tamanho dele depende muito das recursos computacionais do seu sistema:

Eu tenho uma GPU. Ótimo, podemos rapidamente treinar um GPT bebê com as configurações fornecidas no arquivo de configuração `config/train_shakespeare_char.py`.

```
$ python train.py config/train_shakespeare_char.py
```

Se você olhar para dentro dele, verá que estamos treinando um GPT com um tamanho de contexto de até 256 caracteres, 384 canais de recurso e é uma Transformadora de 6 camadas com 6 cabeças em cada camada. Em uma GPU A100, este treinamento leva cerca de 3 minutos e a melhor perda de validação é 1,4697. Com base na configuração, os pontos de verificação do modelo estão sendo escritos no diretório --out_dir out-shakespeare-char. Então, assim que o treinamento terminar, podemos amostrar do melhor modelo apontando o script de amostragem para este diretório:

```
$ python sample.py --out_dir=out-shakespeare-char
```


```
ANGELO:
And cowards it be strawn to my bed,
And thrust the gates of my threats,
Because he that ale away, and hang'd
An one with him.

DUKE VINCENTIO:
I thank your eyes against it.

DUKE VINCENTIO:
Then will answer him to save the malm:
And what have you tyrannous shall do this?

DUKE VINCENTIO:
If you have done evils of all disposition
To end his power, the day of thrust for a common men
That I leave, to fight with over-liking
Hasting in a roseman.
```

Nada mau para um modelo de nível de caractere após 3 minutos de treinamento em uma GPU. Resultados melhores são bastante prováveis de serem obtidos ao invés disso, ajustando fino um modelo GPT-2 pré-treinado neste conjunto de dados (veja a seção de ajuste fino posteriormente).


Eu só tenho um MacBook (ou outro computador barato). Não se preocupe, ainda podemos treinar um GPT, mas queremos diminuir as coisas. Eu recomendo pegar o PyTorch mais recente (selecione: https://pytorch.org/get-started/locally/) pois, atualmente, ele é muito provável de tornar seu código mais eficiente. Mas mesmo sem ele, uma simples execução de treinamento pode ser como segue:

```
$ python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

Aqui, como estamos executando em CPU em vez de GPU, devemos definir --device=cpu e também desativar o PyTorch 2.0 compile com `--compile=False`. Em seguida, ao avaliar, obtemos uma estimativa um pouco mais barulhenta, mas mais rápida (`--eval_iters=20`, em vez de 200), o tamanho do contexto é apenas 64 caracteres em vez de 256 e o tamanho do lote é apenas 12 exemplos por iteração, não 64. Também usaremos um Transformer muito menor (4 camadas, 4 cabeçalhos, tamanho de incorporação 128), e diminuiremos o número de iterações para 2000 (e, correspondentemente, geralmente decai o taxa de aprendizagem para ao redor de max_iters com `--lr_decay_iters`). Como nossa rede é tão pequena, também aliviamos a regularização (`--dropout=0.0`). Isso ainda leva cerca de 3 minutos, mas nos dá uma perda de apenas 1,88 e, portanto, também amostras piores, mas ainda é divertido.

```
GLEORKEN VINGHARD III:
Whell's the couse, the came light gacks,
And the for mought you in Aut fries the not high shee
bot thou the sought bechive in that to doth groan you,
No relving thee post mose the wear
```



## reproducing GPT-2


```
$ python data/openwebtext/prepare.py
```

```
$ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

Isso irá executar por cerca de 4 dias usando o PyTorch Distributed Data Parallel (DDP) e descer para uma perda de ~ 2,85. Agora, um modelo GPT-2 avaliado apenas no OWT tem uma perda de validação de cerca de 3,11, mas se você finetuná-lo ele descerá para o território ~ 2,85 (devido a uma diferença aparente de domínio), tornando os dois modelos ~ iguais.

Se você estiver em um ambiente de cluster e tiver a bênção de múltiplos nós GPU, pode agilizar o processo, por exemplo, em 2 nós como:

```
Roda o primeiro node de  IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py

Roda o worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```


## finetuning

# Finetuning

O finetuning não é diferente do treinamento, apenas precisamos inicializar a partir de um modelo pré-treinado e treinar com uma taxa de aprendizagem menor. Para um exemplo de como finetunar um GPT em novo texto, vá para `data/shakespeare` e execute `prepare.py` para baixar o pequeno conjunto de dados de shakespeare e transformá-lo em um `train.bin` e um `val.bin`, usando o tokenizador BPE OpenAI do GPT-2. Ao contrário do OpenWebText, isso irá executar em segundos. O finetuning pode levar muito pouco tempo, por exemplo, em uma única GPU apenas alguns minutos. Execute um exemplo de finetuning como:

```
$ python train.py config/finetune_shakespeare.py
```


Isso carregará as sobreposições de parâmetros de configuração em `config/finetune_shakespeare.py` (eu não os ajustei muito). Basicamente, inicializamos a partir de um ponto de verificação do GPT2 com `init_from` e treinamos como normal, exceto por ser mais curto e com uma taxa de aprendizagem pequena. Se você estiver ficando sem memória, tente diminuir o tamanho do modelo (são `{'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}`) ou possivelmente diminuir o `block_size` (comprimento do contexto). O melhor ponto de verificação (perda de validação mais baixa) estará no diretório `out_dir`, por exemplo, em `out-shakespeare` por padrão, de acordo com o arquivo de configuração. Então você pode executar o código em `sample.py --out_dir=out-shakespeare`:

```
THEODORE:
Thou shalt sell me to the highest bidder: if I die,
I sell thee to the first; if I go mad,
I sell thee to the second; if I
lie, I sell thee to the third; if I slay,
I sell thee to the fourth: so buy or sell,
I tell thee again, thou shalt not sell my
possession.

JULIET:
And if thou steal, thou shalt not sell thyself.

THEODORE:
I do not steal; I sell the stolen goods.

THEODORE:
Thou know'st not what thou sell'st; thou, a woman,
Thou art ever a victim, a thing of no worth:
Thou hast no right, no right, but to be sold.
```

## Tarefas

- Investigue e adicione FSDP em vez de DDP
- Avalie as perplexidades zero-shot em avaliações padrão (por exemplo, LAMBADA? HELM? etc.)
- Refine o script de aperfeiçoamento, acho que os hiperparâmetros não são ótimos
- Programe aumento linear do tamanho do lote durante o treinamento
- Incorporar outros embeddings (rotatórios, alibis)
- Separe os buffers de otimização dos parâmetros do modelo em pontos de verificação, acho
- Log de saúde da rede adicional (por exemplo, eventos de clipe de gradiente, magnitude)
- Algumas investigações adicionais sobre inicialização melhor etc.

## Solução de problemas

Observe que, por padrão, este repositório usa o PyTorch 2.0 (ou seja, `torch.compile`). Isso é bastante novo e experimental e ainda não está disponível em todas as plataformas (por exemplo, Windows). Se você estiver enfrentando mensagens de erro relacionadas, tente desativá-lo adicionando a flag `--compile=False`. Isso irá desacelerar o código, mas pelo menos ele irá funcionar.



