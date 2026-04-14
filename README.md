# Lab 07 – Fine-Tuning de LLM com LoRA e QLoRA

## Objetivo

O objetivo deste laboratório foi realizar o fine-tuning de um modelo de linguagem utilizando técnicas de otimização que permitem treinar modelos grandes mesmo em ambientes com recursos limitados.

Para isso foram utilizadas duas abordagens principais:

- **LoRA (Low Rank Adaptation)**
- **QLoRA (Quantized LoRA)**

Essas técnicas permitem atualizar apenas uma pequena parte dos parâmetros do modelo, reduzindo bastante o uso de memória da GPU.

---

# Modelo utilizado

Modelo base:

TinyLlama/TinyLlama-1.1B-Chat-v1.0

Esse modelo possui aproximadamente **1.1 bilhões de parâmetros** e foi utilizado como base para o treinamento.

---

# Dataset

O dataset foi gerado de forma sintética utilizando a **API da OpenAI**.

Foram criadas perguntas e respostas relacionadas ao desenvolvimento de jogos utilizando a engine **Unity**.

Exemplo de dado:

```
{
"prompt": "Como funciona o Rigidbody e o sistema de física na Unity?",
"response": "O Rigidbody é um componente responsável por aplicar física em objetos dentro da Unity..."
}
```

Quantidade de dados gerados:

- 50 pares de pergunta e resposta
- 45 exemplos utilizados para treino
- 5 exemplos utilizados para teste

Os dados foram salvos em arquivos `.jsonl`.

Arquivos do dataset:

dataset_treino.jsonl  
dataset_teste.jsonl

---

# Quantização (QLoRA)

Para reduzir o consumo de memória da GPU foi utilizada quantização em **4 bits** através da biblioteca **BitsAndBytes**.

Configuração utilizada:

```
load_in_4bit=True
bnb_4bit_quant_type="nf4"
bnb_4bit_compute_dtype=torch.float16
```

Essa técnica permite carregar modelos grandes mesmo em GPUs com pouca memória.

---

# Configuração do LoRA

O LoRA foi configurado com os seguintes hiperparâmetros:

Rank (r): 64  
Alpha: 16  
Dropout: 0.1  

Durante o treinamento foi possível observar que apenas uma pequena fração dos parâmetros do modelo foi atualizada.

Resultado apresentado:

trainable params: 9,011,200  
all params: 1,109,059,584  
trainable%: 0.8125

Ou seja, **menos de 1% dos parâmetros foram treinados**.

---

# Configuração do treinamento

O treinamento foi realizado utilizando o `SFTTrainer` da biblioteca **TRL**.

Principais parâmetros utilizados:

Epochs: 3  
Batch size: 4  
Gradient accumulation steps: 2  
Learning rate: 2e-4  
Scheduler: cosine  
Optimizer: paged_adamw_32bit  

Essas configurações ajudam a evitar picos de memória durante o treinamento.

---

# Resultado

Após o treinamento, o adaptador LoRA foi salvo no diretório:

unity_lora_final/

Esse diretório contém os pesos treinados que podem ser carregados posteriormente junto ao modelo base.

---

# Estrutura do projeto

```
Lab07/

dataset_treino.jsonl  
dataset_teste.jsonl  
Lab07_Dimmy.py  
unity_lora_final/  
README.md  
```

---

# Como executar o projeto

1. Instalar as bibliotecas necessárias:

```
pip install transformers datasets peft trl bitsandbytes accelerate
```

2. Executar o script principal:

```
python Lab07_Dimmy.py
```

3. O modelo treinado será salvo na pasta:

```
unity_lora_final/
```

---

# Tecnologias utilizadas

- Python
- HuggingFace Transformers
- TRL
- PEFT
- BitsAndBytes
- OpenAI API
- Google Colab

---

# Observação sobre uso de IA

Partes geradas pela IA foi uso pra gerar as perguntas e respostas relacionada a materia de desenvolvimento de jogos, assuntos aplicados na construção da gamificação e utilizada pra consertos na minha penultima célula do código na qual tive dificuldades na implementação por causa dos SFTT e a implementação de treino pras respostas, além disso, utilizei para encontrar certas lib para o código poder rodar.

---

## Anexo 

**Google Colab:**
[https://colab.research.google.com/drive/1FT6qgDFJfDYxi3HZxUeUWxVfpPSaYRmc?usp=sharing]


**Referência:**  
* GOODFELLOW, Ian; BENGIO, Yoshua; COURVILLE, Aaron. Deep Learning. [S. l.]: MIT Press, 2016..
 * JURAFSKY, Daniel; MARTIN, James H. Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition with Language Models. 3. ed. draft. [S. l.]: Stanford University/University of Colorado at Boulder, 2026..
 * RASCHKA, Sebastian. Build a Large Language Model (From Scratch). 1. ed. [S. l.]: Manning (MEAP), 2021..
 * UNIVERSIDADE FEDERAL DO PIAUÍ. Estágio Curricular Supervisionado - Fábrica de Software I: normas para o estágio supervisionado. Teresina: UFPI, 2026..
 * VASWANI, Ashish et al. Atenção é tudo o que você precisa. Tradução de Machine Translated by Google. [S. l.]: Google Brain/Google Research, 2017..
