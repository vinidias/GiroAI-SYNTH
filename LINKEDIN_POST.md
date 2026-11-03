Construí um dataset sintético para auditar meu próprio pipeline de ML. Ele encontrou dois bugs que passariam despercebidos em dados reais.

O contexto: eu tinha um pipeline de ciência de dados rodando no Atacado Atlas (distribuição B2B de alimentos). Os resultados:

- Elasticidade de preço: R² 0.004
- Churn: AUC 0.537 (pior que jogar uma moeda)
- Recomendação: recall@10 0.208

A pergunta óbvia: o problema é o dataset ou o meu código?

O Atlas não responde. Ele não tem elasticidade verdadeira registrada, não tem churners reais, não tem afinidade latente documentada. Não há como saber se o modelo falhou em aprender ou se não havia nada para aprender.

Então construí o GIRO-SYNTH: mesma estrutura operacional (12 categorias, 5 tabelas de preço, 730 dias), mas com três sinais comportamentais injetados a partir de parâmetros conhecidos — elasticidade por SKU (ε 0.8–2.0), churn com hazard correlacionado, e afinidade latente de 8 dimensões.

Escala: 5.000 clientes, 500 SKUs, 475 mil pedidos, 2,1 milhões de itens.

O ground truth revelou dois bugs:

1. Elasticidade: o modelo pooled misturava três fontes de variação de preço (tabela, mês, promoção). Corrigi para efeito fixo SKU×mês — só a promoção é resposta à demanda. R² subiu de 0.106 para 0.700, correlação com a verdade 0.995.

2. Recomendação: o protocolo "esconder o último item" escondia sempre o menor item comprado, que o modelo nunca ranqueia. Corrigi para leave-one-out aleatório. Recall@10 subiu de 0.015 para 0.244.

Benchmark final:

| Modelo | Atlas | GIRO-SYNTH |
| Elasticidade (R²) | 0.004 | 0.700 |
| Churn (AUC) | 0.537 | 0.743 |
| Recomendação | 0.208 | 0.244 |
| Segmentação | 0.413 | 0.559 |

O ponto não é que eu sei treinar um modelo. É que, sem uma verdade conhecida, você não sabe se seu modelo aprendeu o mecanismo ou apenas decorou ruído.

Dataset e pipeline públicos:
→ GIRO-SYNTH: https://github.com/vinidias/GiroAI-SYNTH
→ Giro AI: https://github.com/vinidias/GiroAI-B2B-SKU-recommendation-price-elasticity-churn-and-segmentation
→ Report visual: https://vinidias.github.io/GiroAI-SYNTH/report.html

#DataScience #MachineLearning #MLOps #CrispDM #Python