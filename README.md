# tech-challenger-03

Este repositório reúne experimentos da Fase 3 do desafio técnico da FIAP para construir um modelo pre-treinado em perguntas e respostas sobre cancer, alem de a utilizacao de RAG (Retrieval-Augmented Generation) e LangChain para a construcao do fluxo. O fluxo principal inclui:

- Preparação de dados: extração e tratamento de pares pergunta-resposta a partir de arquivos XML do dataset CancerGov e exportação para `cancer_QA_treated.json`.
- Fine-tuning: treinamento supervisionado (SFT) de um modelo de linguagem (usando a biblioteca `unsloth` + `trl`) com os pares QA para especializar o modelo em respostas clínicas sobre câncer.
- Indexação & Recuperação: criação de uma base vetorial com `chromadb` e embeddings (`all-MiniLM-L6-v2`) para recuperar contexto relevante.
- Integração RAG: combinação do retriever com o modelo fine-tuned para responder perguntas usando apenas o contexto recuperado.

#### Arquivos principais

- `FINE_TUNNING.ipynb`: pipeline de pré-processamento dos XML, formatação dos prompts, configuração do SFTTrainer e salvamento do modelo treinado no Google Drive.
- `MODEL_RAG.ipynb`: carregamento do modelo salvo, criação do wrapper LLM personalizado e execução do fluxo RetrievalQA com LangChain.
- `RAG.ipynb`: construção do banco vetorial Chroma a partir do JSON tratado e exemplos de consultas para validar a recuperação.

#### Dependências principais

chromadb, sentence-transformers, langchain-community, unsloth, trl, transformers, datasets, peft, accelerate, bitsandbytes, xformers

#### Observações

- Os notebooks foram desenvolvidos para execução em Colab usando caminhos de `Google Drive`.
- É necessário GPU compatível (preferencialmente com suporte a 4-bit / quantização) para treinar e inferir modelos grandes.
- Ajuste de hiperparâmetros, tamanho do batch e passos de treino foi deixado como exemplo; para produção sugere-se validação e testes adicionais.

#### Dataset

[github.com/abachaa/MedQuAD/tree/master/1_CancerGov_QA](https://github.com/abachaa/MedQuAD/tree/master/1_CancerGov_QA)
