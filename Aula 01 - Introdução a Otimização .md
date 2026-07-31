# Aula prática de Pesquisa Operacional: fábrica de mesas e cadeiras com blocos de encaixe

> **Tema:** modelagem de um problema de mix de produção
>
> **Conteúdos:** variáveis de decisão, função objetivo, restrições, solução factível, solução ótima, recursos escassos, análise de sensibilidade e integralidade.
>
> **Duração sugerida:** 90 a 120 minutos.

---

## 1. Ideia da dinâmica

Nesta atividade, cada grupo representará uma pequena fábrica de móveis. A fábrica dispõe de uma quantidade limitada de peças de encaixe e pode produzir apenas dois produtos:

- mesas;
- cadeiras.

Cada produto utiliza diferentes quantidades de recursos e gera um lucro específico. O desafio dos grupos será decidir **quantas mesas e quantas cadeiras produzir para obter o maior lucro possível**, sem ultrapassar a quantidade de peças disponível.

A atividade começa de forma concreta e intuitiva, com a manipulação dos blocos. Depois, as decisões tomadas pelos grupos serão transformadas em um modelo matemático de Pesquisa Operacional.

---

## 2. Objetivos de aprendizagem

Ao final da aula, espera-se que os estudantes sejam capazes de:

1. identificar variáveis de decisão em um problema de produção;
2. formular uma função objetivo de maximização;
3. representar matematicamente as limitações de recursos;
4. distinguir soluções possíveis, inviáveis e ótimas;
5. perceber que produzir somente o item de maior lucro unitário nem sempre gera o maior lucro total;
6. analisar o efeito de mudanças na disponibilidade dos recursos;
7. compreender por que alguns problemas exigem variáveis inteiras.

---

## 3. Materiais necessários

Para cada grupo:

- 6 peças grandes;
- 8 peças pequenas;
- 7 peças de conexão ou peças de uma terceira cor;
- uma ficha com as regras de produção;
- uma ficha para registrar o plano de produção;
- calculadora ou celular;
- papel quadriculado, caso a solução gráfica seja realizada manualmente.

### Sugestão de cores

| Recurso | Cor sugerida | Quantidade inicial |
|---|---:|---:|
| Peça grande | Azul | 6 |
| Peça pequena | Amarela | 8 |
| Conector ou reforço | Vermelha | 7 |

> Caso o terceiro tipo de peça não se encaixe fisicamente no móvel, ele pode ser colocado ao lado do produto, representando um componente obrigatório do respectivo kit de fabricação.

---

## 4. Organização dos grupos

Organize a turma em grupos de quatro ou cinco estudantes. Cada integrante pode assumir uma função:

| Função | Responsabilidade |
|---|---|
| Gerente de produção | Coordena a decisão do grupo |
| Planejador | Calcula as combinações de produtos |
| Responsável pelo estoque | Controla o uso das peças |
| Montador | Constrói os produtos planejados |
| Analista financeiro | Calcula o lucro obtido |

Em grupos menores, um estudante pode assumir mais de uma função.

---

# Parte I — Roteiro dos estudantes

## 5. Situação-problema

Uma fábrica produz mesas e cadeiras utilizando três tipos de componentes. Os produtos possuem os seguintes preços de venda:

- **Mesa:** R$ 16,00;
- **Cadeira:** R$ 10,00.

Não serão considerados outros custos. Portanto, para a dinâmica, o valor de venda será tratado como o lucro obtido com cada unidade produzida.

### Recursos necessários por produto

| Produto | Peças grandes | Peças pequenas | Conectores | Lucro unitário |
|---|---:|---:|---:|---:|
| Mesa | 2 | 2 | 1 | R$ 16,00 |
| Cadeira | 1 | 2 | 2 | R$ 10,00 |

### Estoque disponível

| Recurso | Quantidade disponível |
|---|---:|
| Peças grandes | 6 |
| Peças pequenas | 8 |
| Conectores | 7 |

---

## 6. Regras da fábrica

1. Somente produtos completamente montados podem ser vendidos.
2. Produtos incompletos não geram lucro.
3. O grupo não pode utilizar mais peças do que recebeu.
4. Não é permitido quebrar, dividir ou substituir uma peça por outra.
5. Na primeira rodada, não é permitido trocar peças com outros grupos.
6. Antes de começar a montagem, o grupo deve registrar seu plano de produção.
7. Após o início da montagem, o grupo poderá alterar o plano, mas deverá registrar a mudança e explicar o motivo.
8. Vence a rodada o grupo que alcançar o maior lucro com uma solução válida.

---

## 7. Rodada 1 — Decisão intuitiva

### Desafio

Sem formular inicialmente um modelo matemático, decidam:

- quantas mesas produzir;
- quantas cadeiras produzir;
- qual será o lucro esperado;
- quais peças deverão sobrar.

### Registro do plano

| Informação | Resposta do grupo |
|---|---|
| Mesas planejadas |  |
| Cadeiras planejadas |  |
| Peças grandes utilizadas |  |
| Peças pequenas utilizadas |  |
| Conectores utilizados |  |
| Lucro esperado |  |

### Registro da produção

| Informação | Resultado obtido |
|---|---|
| Mesas concluídas |  |
| Cadeiras concluídas |  |
| Peças grandes restantes |  |
| Peças pequenas restantes |  |
| Conectores restantes |  |
| Lucro total |  |

### Questões para discussão no grupo

1. O produto com maior lucro unitário deve necessariamente ser produzido em maior quantidade?
2. Qual recurso parece limitar mais a produção?
3. Houve sobra de algum recurso?
4. Uma peça que sobrou possui valor econômico nesta situação?
5. O grupo consegue provar que sua solução é a melhor possível?

---

## 8. Socialização das decisões

Cada grupo deverá apresentar sua estratégia em até dois minutos.

O professor registrará no quadro os resultados utilizando uma tabela semelhante à seguinte:

| Grupo | Mesas | Cadeiras | Lucro | Solução válida? | Estratégia utilizada |
|---|---:|---:|---:|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |

Durante a apresentação, os grupos devem explicar se adotaram alguma das seguintes estratégias:

- produzir primeiro o item de maior lucro;
- utilizar completamente um dos recursos;
- testar várias combinações;
- comparar o lucro por peça utilizada;
- montar os produtos sem planejamento prévio.

---

## 9. Transformando a situação em um modelo matemático

Defina:

- $x_1$: quantidade de mesas produzidas;
- $x_2$: quantidade de cadeiras produzidas.

### Função objetivo

O objetivo é maximizar o lucro total:

$$
\text{Maximizar } Z = 16x_1 + 10x_2
$$

### Restrição de peças grandes

Cada mesa utiliza duas peças grandes e cada cadeira utiliza uma:

$$
2x_1 + x_2 \leq 6
$$

### Restrição de peças pequenas

Cada mesa e cada cadeira utilizam duas peças pequenas:

$$
2x_1 + 2x_2 \leq 8
$$

### Restrição de conectores

Cada mesa utiliza um conector e cada cadeira utiliza dois:

$$
x_1 + 2x_2 \leq 7
$$

### Não negatividade e integralidade

$$
x_1, x_2 \geq 0
$$

Como não é possível produzir uma fração de mesa ou cadeira:

$$
x_1, x_2 \in \mathbb{Z}
$$

### Modelo completo

$$
\begin{aligned}
\text{Maximizar } & Z = 16x_1 + 10x_2 \\
\text{sujeito a } & 2x_1 + x_2 \leq 6 \\
                  & 2x_1 + 2x_2 \leq 8 \\
                  & x_1 + 2x_2 \leq 7 \\
                  & x_1, x_2 \geq 0 \\
                  & x_1, x_2 \in \mathbb{Z}
\end{aligned}
$$

---

## 10. Verificação das combinações inteiras

Complete a tabela com as combinações que respeitam todas as restrições.

| Mesas ($x_1$) | Cadeiras ($x_2$) | Grandes usadas | Pequenas usadas | Conectores usados | Lucro | Factível? |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 |  |  |  |  |  |
| 0 | 1 |  |  |  |  |  |
| 0 | 2 |  |  |  |  |  |
| 0 | 3 |  |  |  |  |  |
| 0 | 4 |  |  |  |  |  |
| 1 | 0 |  |  |  |  |  |
| 1 | 1 |  |  |  |  |  |
| 1 | 2 |  |  |  |  |  |
| 1 | 3 |  |  |  |  |  |
| 2 | 0 |  |  |  |  |  |
| 2 | 1 |  |  |  |  |  |
| 2 | 2 |  |  |  |  |  |
| 3 | 0 |  |  |  |  |  |
| 3 | 1 |  |  |  |  |  |

---

## 11. Solução gráfica

Para construir o gráfico, utilize:

- eixo horizontal: número de mesas ($x_1$);
- eixo vertical: número de cadeiras ($x_2$).

### Retas das restrições

| Restrição | Intercepto em $x_1$ | Intercepto em $x_2$ |
|---|---:|---:|
| $2x_1+x_2=6$ | 3 | 6 |
| $x_1+x_2=4$ | 4 | 4 |
| $x_1+2x_2=7$ | 7 | 3,5 |

> A segunda restrição foi simplificada de $2x_1+2x_2\leq8$ para $x_1+x_2\leq4$.

Após desenhar as três retas:

1. identifique o lado permitido por cada restrição;
2. destaque a região factível;
3. encontre os vértices da região;
4. calcule o valor da função objetivo em cada vértice;
5. compare a solução contínua com as combinações inteiras que podem ser montadas.

---

## 12. Rodada 2 — Uma peça defeituosa

O fornecedor informou que **um dos conectores está defeituoso**. O estoque passa de sete para seis conectores.

A nova restrição é:

$$
x_1 + 2x_2 \leq 6
$$

Responda antes de montar novamente:

1. A solução da primeira rodada continua sendo possível?
2. O lucro máximo será alterado?
3. Qual era o valor econômico do sétimo conector?
4. Um recurso adicional sempre aumenta o lucro?

Registre a nova decisão:

| Informação | Resultado |
|---|---|
| Mesas |  |
| Cadeiras |  |
| Lucro |  |
| Diferença em relação à Rodada 1 |  |

---

## 13. Rodada 3 — Falta de conectores

Um segundo conector apresentou defeito. Agora existem apenas cinco conectores disponíveis.

A nova restrição é:

$$
x_1 + 2x_2 \leq 5
$$

### Desafio

1. Encontre a melhor solução utilizando o gráfico, sem impor inicialmente a integralidade.
2. Verifique se essa solução pode ser montada com os blocos.
3. Encontre a melhor solução inteira.
4. Compare o lucro da solução contínua com o lucro da solução inteira.
5. Explique por que o modelo precisa considerar variáveis inteiras.

| Tipo de solução | Mesas | Cadeiras | Lucro | Pode ser montada? |
|---|---:|---:|---:|---|
| Solução contínua |  |  |  |  |
| Melhor solução inteira |  |  |  |  |

---

## 14. Rodada opcional — Compra emergencial de matéria-prima

O fornecedor oferece **uma peça grande adicional**. A empresa passará a dispor de sete peças grandes.

A restrição correspondente muda para:

$$
2x_1+x_2\leq7
$$

Perguntas:

1. Quanto o lucro máximo pode aumentar?
2. Qual combinação de produtos deverá ser utilizada?
3. Qual seria o valor máximo que a empresa deveria pagar por essa peça adicional?
4. A resposta seria a mesma se fossem oferecidas dez peças extras?

Esta rodada pode ser utilizada para introduzir os conceitos de:

- preço-sombra;
- valor marginal de um recurso;
- intervalo de validade da análise de sensibilidade.

---

## 15. Fechamento da atividade

Cada grupo deverá responder, por escrito, às seguintes questões:

1. Qual foi a principal diferença entre decidir intuitivamente e utilizar o modelo matemático?
2. Qual recurso foi o gargalo em cada rodada?
3. Por que uma peça disponível pode não possuir valor marginal?
4. Em qual rodada surgiu a necessidade de utilizar programação inteira?
5. Como essa dinâmica se relaciona com decisões reais de produção?
6. Que outros elementos poderiam ser incluídos para tornar o problema mais próximo de uma fábrica real?

---

# Parte II — Orientações para o professor

## 16. Condução sugerida da aula

| Etapa | Tempo sugerido | Ação |
|---|---:|---|
| Apresentação do problema | 10 min | Explicar o cenário sem mostrar o modelo matemático |
| Formação dos grupos e distribuição dos materiais | 5 min | Entregar estoque e fichas |
| Rodada 1 | 15 min | Planejamento, montagem e cálculo do lucro |
| Socialização | 10 min | Comparar as estratégias dos grupos |
| Formulação matemática | 20 min | Construir o modelo com participação da turma |
| Resolução gráfica ou por enumeração | 20 min | Identificar região factível e solução ótima |
| Rodadas de alteração de recursos | 15 a 25 min | Trabalhar sensibilidade e integralidade |
| Síntese e avaliação | 10 min | Retomar os conceitos aprendidos |

---

## 17. Perguntas de mediação

Durante a dinâmica, evite informar imediatamente se uma solução é ótima. Utilize perguntas como:

- Como vocês sabem que não existe uma combinação melhor?
- Qual peça impede a produção de mais uma unidade?
- O lucro por unidade é suficiente para tomar a decisão?
- Uma mesa utiliza os recursos da mesma forma que uma cadeira?
- Que informação poderia ser representada por uma variável?
- Como escrever uma desigualdade que represente o estoque?
- O que significa um ponto fora da região factível?
- Por que o resultado do Simplex pode não ser diretamente aplicável à produção?

---

## 18. Gabarito da Rodada 1

<details>
<summary><strong>Exibir solução</strong></summary>

O modelo é:

$$
\begin{aligned}
\text{Maximizar } & Z=16x_1+10x_2 \\
\text{sujeito a } & 2x_1+x_2\leq6 \\
                  & x_1+x_2\leq4 \\
                  & x_1+2x_2\leq7 \\
                  & x_1,x_2\geq0
\end{aligned}
$$

Os principais vértices da região factível são:

| Vértice | Mesas | Cadeiras | Lucro |
|---|---:|---:|---:|
| A | 0 | 0 | R$ 0,00 |
| B | 0 | 3,5 | R$ 35,00 |
| C | 1 | 3 | R$ 46,00 |
| D | 2 | 2 | **R$ 52,00** |
| E | 3 | 0 | R$ 48,00 |

A melhor solução é:

- $x_1=2$ mesas;
- $x_2=2$ cadeiras;
- lucro máximo de **R$ 52,00**.

Consumo de recursos:

| Recurso | Utilizado | Disponível | Sobra |
|---|---:|---:|---:|
| Peças grandes | 6 | 6 | 0 |
| Peças pequenas | 8 | 8 | 0 |
| Conectores | 6 | 7 | 1 |

O conector que sobra ajuda a preparar a discussão da rodada seguinte. Ele está disponível, mas não aumenta o lucro na solução atual.

</details>

---

## 19. Gabarito da Rodada 2

<details>
<summary><strong>Exibir solução</strong></summary>

Com seis conectores, a restrição passa a ser:

$$
x_1+2x_2\leq6
$$

A combinação de duas mesas e duas cadeiras utiliza exatamente seis conectores. Portanto:

- a solução continua sendo $x_1=2$ e $x_2=2$;
- o lucro permanece em **R$ 52,00**;
- a redução de sete para seis conectores não altera o lucro máximo;
- o sétimo conector possuía valor marginal igual a zero nessa situação.

Essa rodada mostra que a disponibilidade de um recurso somente possui valor quando o recurso efetivamente limita a solução.

</details>

---

## 20. Gabarito da Rodada 3

<details>
<summary><strong>Exibir solução</strong></summary>

Com cinco conectores, a restrição passa a ser:

$$
x_1+2x_2\leq5
$$

### Solução da relaxação linear

A interseção das restrições

$$
2x_1+x_2=6
$$

e

$$
x_1+2x_2=5
$$

resulta em:

$$
x_1=\frac{7}{3}\approx2,33
$$

$$
x_2=\frac{4}{3}\approx1,33
$$

O lucro da solução contínua é:

$$
Z=16\left(\frac{7}{3}\right)+10\left(\frac{4}{3}\right)
=\frac{152}{3}\approx R\$50,67
$$

Essa solução não pode ser montada, pois exige frações de produtos.

### Melhor solução inteira

As principais soluções inteiras possíveis são:

| Mesas | Cadeiras | Lucro |
|---:|---:|---:|
| 3 | 0 | **R$ 48,00** |
| 2 | 1 | R$ 42,00 |
| 1 | 2 | R$ 36,00 |
| 0 | 2 | R$ 20,00 |

Portanto, a melhor solução inteira é:

- três mesas;
- nenhuma cadeira;
- lucro de **R$ 48,00**.

A diferença entre a solução da relaxação linear e a melhor solução inteira permite introduzir o conceito de **gap de integralidade**.

</details>

---

## 21. Gabarito da rodada opcional: peça grande adicional

<details>
<summary><strong>Exibir solução</strong></summary>

Mantendo o estoque inicial dos demais recursos e aumentando o número de peças grandes de seis para sete, uma solução possível é:

- três mesas;
- uma cadeira.

Consumo:

| Recurso | Cálculo | Total utilizado | Disponível |
|---|---|---:|---:|
| Peças grandes | $2(3)+1(1)$ | 7 | 7 |
| Peças pequenas | $2(3)+2(1)$ | 8 | 8 |
| Conectores | $1(3)+2(1)$ | 5 | 7 |

Lucro:

$$
Z=16(3)+10(1)=R\$58,00
$$

O lucro aumenta de R$ 52,00 para R$ 58,00. Assim, dentro dessa alteração, o valor marginal da peça grande adicional é de **R$ 6,00**.

</details>

---

## 22. Avaliação da aprendizagem

A atividade pode valer uma pequena pontuação, considerando:

| Critério | Pontuação sugerida |
|---|---:|
| Participação e organização do grupo | 1,0 |
| Registro correto dos recursos utilizados | 1,0 |
| Formulação das variáveis de decisão | 1,0 |
| Formulação da função objetivo | 1,0 |
| Formulação das restrições | 2,0 |
| Identificação da solução ótima | 1,5 |
| Justificativa da solução | 1,5 |
| Análise das mudanças de estoque | 1,0 |
| **Total** | **10,0** |

---

## 23. Possíveis extensões

### Alteração dos preços

Apresente uma nova condição de mercado, por exemplo:

- mesa: R$ 16,00;
- cadeira: R$ 18,00.

Peça aos estudantes que verifiquem se a solução ótima muda.

### Pedido mínimo

Inclua uma exigência comercial:

$$
x_2\geq1
$$

A empresa deve produzir pelo menos uma cadeira para atender a um contrato.

### Capacidade de montagem

Cada mesa demanda 15 minutos e cada cadeira demanda 10 minutos. A equipe possui apenas 50 minutos:

$$
15x_1+10x_2\leq50
$$

### Custo de matéria-prima

Substitua o lucro direto por receitas e custos, exigindo que os estudantes calculem a margem de contribuição de cada produto.

### Mais produtos

Depois que os estudantes compreenderem o problema com duas variáveis, acrescente um terceiro produto. Nesse caso, a representação gráfica deixa de ser simples e passa a ser interessante utilizar o Simplex ou um Solver.

---

## 24. Encaminhamento após a aula

Como atividade posterior, os estudantes podem:

1. implementar o modelo no Solver do Excel ou LibreOffice Calc;
2. resolver o modelo com e sem a restrição de integralidade;
3. comparar a solução computacional com a solução obtida com os blocos;
4. criar uma nova alteração de cenário e explicar seu efeito sobre a solução ótima;
5. escrever uma breve reflexão sobre a importância da modelagem matemática na tomada de decisão.

---

## 25. Síntese conceitual

A dinâmica evidencia que uma decisão de produção depende da combinação entre:

- retorno econômico dos produtos;
- quantidade de recursos consumidos;
- disponibilidade dos recursos;
- possibilidade de produzir unidades fracionárias ou apenas inteiras;
- mudanças no mercado e no estoque.

A Pesquisa Operacional transforma essas informações em um modelo que permite comparar alternativas e justificar matematicamente a decisão escolhida.
