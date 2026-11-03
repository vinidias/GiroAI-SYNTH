# Atlas vs GIRO-SYNTH: o que um ground truth revela sobre seus modelos

*Post de portfólio — Vinicius Dias*

---

Todo cientista de dados já passou por isso: o modelo roda, as métricas saem, o dashboard fica bonito. Mas **como você sabe que o modelo aprendeu o sinal certo?**

A resposta padrão é "validação cruzada". Mas validação cruzada só mede se o modelo generaliza para dados parecidos. Ela não responde à pergunta mais importante: *o modelo está capturando o mecanismo que gera os dados, ou apenas decorando ruído?*

Para responder isso, você precisa de algo que quase nenhum dataset público tem: **ground truth documentado**.

Foi por isso que construí o **GIRO-SYNTH**, e o que ele revelou sobre o meu próprio pipeline foi desconfortável — e extremamente útil.

---

## O problema: datasets reais não têm verdade

Pegue o **Atacado Atlas**, um dataset sintético excelente de distribuição B2B de alimentos. 22 tabelas, 730 dias, 144 SKUs, 52 mil pedidos. Perfeito para engenharia de dados: joins, FEFO, rotas, financeiro.

Rodei meu pipeline de ciência de dados nele. Os resultados:

| Modelo | Métrica | Valor |
|---|---|---|
| Elasticidade de preço | R² | 0.004 |
| Churn | ROC-AUC | 0.537 |
| Recomendação de SKUs | recall@10 | 0.208 |
| Segmentação | silhouette | 0.413 |

Elasticidade com R² de 0.004. Churn com AUC de 0.537 — **pior que jogar uma moeda**. Algo estava errado.

A pergunta óbvia: o problema é o dataset ou o meu código?

O Atlas não responde. Ele não tem elasticidade verdadeira registrada, não tem churners reais (todos os 480 clientes compram em todos os trimestres), não tem afinidade latente documentada. **Não há como saber se o modelo falhou em aprender ou se não havia nada para aprender.**

---

## A solução: um dataset com verdade documentada

Construí o **GIRO-SYNTH** com a mesma estrutura operacional do Atlas (12 categorias, 6 segmentos, 5 tabelas de preço, BRL, 730 dias), mas com três sinais comportamentais **injetados a partir de parâmetros conhecidos**:

1. **Elasticidade de preço**: `qty = k * (preco/preco_cheio)^(-epsilon)`, com epsilon por SKU (0.8–2.0)
2. **Churn**: hazard mensal que cresce quando receita, frequência e pagamento deterioram
3. **Afinidade latente**: vetores de 8 dimensões por cliente e SKU, que fazem clientes com gosto similar comprarem SKUs similares

Cada sinal tem seus parâmetros publicados em `gt_*.csv`. Qualquer pessoa pode medir se um modelo recupera a verdade.

Escala: **5.000 clientes, 500 SKUs, 475 mil pedidos, 2,1 milhões de itens**. Gerado em 50 segundos.

---

## O que o ground truth revelou: dois bugs no meu pipeline

Rodei o mesmo pipeline do Giro AI no GIRO-SYNTH. Os resultados foram melhores, mas não perfeitos:

| Modelo | Atlas | GIRO-SYNTH (1ª rodada) |
|---|---|---|
| Elasticidade (R²) | 0.004 | 0.106 |
| Churn (AUC) | 0.537 | 0.743 |
| Recomendação (recall@10) | 0.208 | 0.015 |

Melhorou, mas ainda errado. E aqui está o valor do ground truth: **eu sabia que estava errado**, porque a elasticidade verdadeira era 1.2 e o recall deveria ser > 0.20.

Investiguei os dois casos.

### Bug 1: a especificação da elasticidade estava errada

O modelo original era:

```python
log_qty ~ log_preco * C(categoria)
```

Parece razoável. Mas `log_preco` no Atlas (e no meu dataset) mistura **três fontes de variação**:
- a tabela de preço atribuída ao cliente (fator 0.88–1.12)
- a mudança mensal do preço de lista
- a promoção por pedido

Só a **terceira** é resposta à demanda. As duas primeiras são variação de oferta. O OLS pooled capturava a mistura e produzia coeficientes sem sentido (até positivos).

A correção: **efeito fixo por SKU x mês**. Isso absorve tabela e mês, deixando só a variação de promoção — exatamente o sinal de elasticidade.

```python
log_qty_c ~ log_preco_c - 1   # within SKU x month
```

Resultado: **R² subiu de 0.106 para 0.700**. Todas as 12 categorias com elasticidade negativa (-1.44 a -0.82). Comparando com a verdade (0.8–2.0), a correlação é **0.995**.

### Bug 2: a avaliação de recomendação era enviesada

O protocolo original era "esconder o último item comprado por usuário". Parece leave-one-out, mas não é: o "último item" pelo volume é sempre o **menor** — o que o modelo nunca ranqueia bem.

A correção: **leave-one-out aleatório** sobre 500 usuários, escondendo um item qualquer e pedindo top-10.

Resultado: **recall@10 subiu de 0.015 para 0.244**. Idêntico à validação independente (0.246).

---

## O comparativo final

Com os dois bugs corrigidos, o benchmark honesto:

| Modelo | Atlas | GIRO-SYNTH | O que mudou |
|---|---|---|---|
| Elasticidade (R²) | 0.004 | **0.700** | Atlas não simula resposta a preço |
| Churn (AUC) | 0.537 | **0.743** | Atlas não tem churners |
| Recomendação (recall@10) | 0.208 | **0.244** | GIRO-SYNTH tem afinidade latente |
| Segmentação (silhouette) | 0.413 | **0.559** | comportamento mais rico |

E a validação **contra a verdade** (não contra held-out):

| Sinal | Métrica | Valor |
|---|---|---|
| Elasticidade | Pearson(verdade, estimado) | **0.995** |
| Churn | AUC (features RFM) | **0.988** |
| Recomendação | ALS recall@10 | **0.246** |

---

## Por que isso importa para uma vaga de Cientista de Dados Sênior

A vaga pede "Exploração e Qualidade: EDA profunda e homologação dos dados antes da fase de treinamento" e "Cultura de Testes: metodologias de mensuração de impacto".

O que este exercício demonstra não é que eu sei treinar um modelo. É que eu sei:

1. **Diagnosticar** por que um modelo falha — separando problema de dado de problema de especificação
2. **Especificar corretamente** — reconhecer que pooled OLS conflita fontes de variação e corrigir com fixed effects
3. **Avaliar com rigor** — identificar viés no protocolo de avaliação (o "último item" era um proxy ruim)
4. **Construir o instrumento de medição** quando o dado real não permite — que é exatamente o que um time sênior faz quando não há ground truth disponível

O dataset e o pipeline estão públicos:

- **GIRO-SYNTH** (dataset + gerador): https://github.com/vinidias/GiroAI-SYNTH
- **Giro AI** (pipeline com os 4 modelos): https://github.com/vinidias/GiroAI-B2B-SKU-recommendation-price-elasticity-churn-and-segmentation

O dataset será publicado no Kaggle com licença CC BY 4.0, incluindo o ground truth — para que qualquer pessoa possa auditar seus próprios modelos, não só os meus.

---

*Se você trabalha com modelagem preditiva e nunca mediu seus modelos contra uma verdade conhecida, recomendo o exercício. O resultado costuma ser humilhante — e depois, muito melhor.*