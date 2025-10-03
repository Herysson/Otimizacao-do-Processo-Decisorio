
# Aula prática — Modelagem **gráfica** de um PL em Python

**Objetivo:** implementar, do zero, uma visualização completa do problema de Programação Linear (PL) abaixo, incluindo:
- grade cartesiana;
- retas das restrições;
- área viável;
- vetor gradiente da função objetivo;
- (opcional) retas de nível;
- pontos candidatos (cantos) com o valor da função objetivo \(Z\).

**Nota didática:** Usaremos a função auxiliar **`finalizar_plot()`** como um **rodapé fixo** em todas as etapas.  
> Isso garante que cada trecho de código gere um **gráfico visível** imediatamente, ajudando quem está começando.


**Problema-alvo**
Maximizar $Z=30x+50y$ sujeito a

$$
\begin{cases}
2x+4y\le 100 \\
4x+3y\le 120 \\
x\ge 0,\; y\ge 0
\end{cases}
$$

---

## 0) Preparando o ambiente

```python
import numpy as np
import matplotlib.pyplot as plt
```

* **NumPy (`np`)**: contas numéricas e vetores (arranjos).
* **Matplotlib (`plt`)**: desenhar gráficos.

Criamos também a **função objetivo**:

```python
def objective(x, y):  # Z = 30x + 50y
    return 30*x + 50*y
```

> Dica: manter a FO em função separada facilita reuso (cálculo de $Z$ nos pontos).

---

## 1) Criação de uma **grade cartesiana**

Aqui definimos uma “janela” (limites do gráfico) e o **eixo $x$** amostrado:

```python
xmax = 50   # um pouco maior que os interceptos para sobrar espaço
ymax = 50

x = np.linspace(0, xmax, 400)  # 400 pontos igualmente espaçados entre 0 e xmax
```

Em seguida:

```python
plt.figure(figsize=(6, 6))                          # tamanho do gráfico
plt.grid(True, which='both', linestyle='--', linewidth=0.5)   # grade “quadriculada”
```

* A **grade** ajuda a ler coordenadas e comparar alturas.

---

## 2) Adição das **Restrições** (como retas)

As restrições são desenhadas primeiro **como igualdades** (as fronteiras):

* $2x+4y=100 \Rightarrow y = 25 - 0{,}5x$
* $4x+3y=120 \Rightarrow y = 40 - \tfrac{4}{3}x$

```python
y1 = 25 - 0.5*x           # 2x + 4y = 100
y2 = 40 - (4/3)*x         # 4x + 3y = 120

plt.plot(x, y1, label='2x + 4y = 100')
plt.plot(x, y2, label='4x + 3y = 120')
```

* **Ideia**: em PL 2D, cada restrição linear vira uma **reta**. A desigualdade “$\le$” indica que a região viável fica **abaixo** dessas retas (e no 1º quadrante).

---

## 3) Adição da **área viável**

A região viável é a **interseção** de:

* o **1º quadrante** ($x\ge 0$, $y\ge 0$),
* o lado **abaixo** de cada reta.

Se, para cada $x$, a maior altura permitida é o **mínimo** entre as duas retas, então a “tampa” da região é `np.minimum(y1, y2)`:

```python
y_feasible = np.minimum(y1, y2)

plt.fill_between(
    x, 0, y_feasible,
    where=(y_feasible > 0),   # só preenche onde a altura é positiva
    color='gray', alpha=0.5,
    label='Região viável'
)
```

* **`fill_between`** preenche a área entre duas curvas: aqui, de **0** até a altura viável.
* O `where=(y_feasible > 0)` evita preencher trechos abaixo do eixo $x$.

> Observação: essa é uma forma **simples e didática** de preencher a região. Outra forma é preencher diretamente o **polígono** dos vértices (ver etapa 6).

---

## 4) **Vetor gradiente** da FO

O gradiente de $Z=30x+50y$ é $\nabla Z=(30,50)$. Ele aponta para onde $Z$ cresce **mais rápido**.
Você plota uma seta proporcional a isso:

```python
plt.arrow(0, 0, 6, 10, head_width=1, head_length=1.5,
          fc='red', ec='red', label='∇Z (30, 50)')
```

* A seta vai de $(0,0)$ até $(6,10)$, que é proporcional a $(30,50)$.
* **Nota**: `plt.arrow` não aparece automaticamente na legenda (mesmo com `label=`).
  Se quiser que **apareça na legenda**, troque por:

  ```python
  plt.quiver(0, 0, 6, 10, angles='xy', scale_units='xy', scale=1, label='∇Z (30, 50)')
  ```

---

## 5) **Retas de nível** (opcional, **antes** da próxima etapa)

As **retas de nível** de $Z$ são as linhas onde $Z$ vale um número constante $c$: $30x+50y=c$.
Elas são **perpendiculares** ao gradiente. A ideia didática: “deslizar” essas retas na direção do gradiente até tocar a região viável no ponto ótimo.

Se quiser desenhar várias:

```python
# (opcional) malha para curvas de nível
x_grid = np.arange(0, xmax+1, 1)
y_grid = np.arange(0, ymax+1, 1)
X, Y = np.meshgrid(x_grid, y_grid)

Z = objective(X, Y)
contour = plt.contour(X, Y, Z, levels=20, colors='k', linestyles='--', linewidths=1.2)
plt.clabel(contour, inline=True, fontsize=8)
```

> Como você comentou, **vamos remover** essas curvas antes da etapa seguinte para deixar o gráfico mais limpo — mas é bom mostrar uma vez em aula, para o conceito de **“deslizar isóclinas”**.

---

## 6) **Pontos** da área de factibilidade e **interseção** das retas, com $Z$

Para encontrar o **vértice de interseção** das duas retas (o ponto onde elas se cruzam), resolvemos o **sistema linear**:

$$
\begin{cases}
2x+4y=100 \\
4x+3y=120
\end{cases}
$$

Em forma matricial $A\mathbf{u}=\mathbf{b}$, com

$$
A=\begin{bmatrix}2 & 4\\4 & 3\end{bmatrix},\quad
\mathbf{b}=\begin{bmatrix}100\\120\end{bmatrix}.
$$

No código:

```python
A = np.array([[2, 4],
              [4, 3]], dtype=float)
b = np.array([100, 120], dtype=float)

intersection = np.linalg.solve(A, b)   # -> array([18., 16.])
```

Agora listamos os **vértices** do polígono viável no 1º quadrante, na **ordem de contorno**:

```python
pts = np.array([
    (0.0, 0.0),      # origem
    (30.0, 0.0),     # intercepto viável no eixo x (da 2ª restrição)
    intersection,    # interseção das retas
    (0.0, 25.0),     # intercepto viável no eixo y (da 1ª restrição)
], dtype=float)
```

Plote e **anote** cada ponto, já exibindo o valor de $Z$:

```python
plt.scatter(pts[:,0], pts[:,1], s=40, label='Pontos candidatos')

for px, py in pts:
    z = objective(px, py)  # Z = 30x + 50y
    plt.annotate(f'({px:.1f}, {py:.1f})\nZ={z:.0f}',
                 (px, py),
                 xytext=(5,5), textcoords='offset points',
                 ha='left', va='bottom')
```

* Esses são os **cantos** da região. Em PL com região convexa e limitada, o **ótimo** ocorre em um desses cantos.
* (Se quiser **encontrar e destacar** o melhor já no código: `Zvals = objective(pts[:,0], pts[:,1]); i = np.argmax(Zvals); plt.scatter([pts[i,0]],[pts[i,1]], marker='*', s=150)`.)

---

## 7) Ajustes finais de eixos, rótulos e legenda

```python
plt.xlim(0, xmax); plt.ylim(0, ymax)
plt.xlabel('x'); plt.ylabel('y')
plt.legend()
plt.title('Restrições, região viável, ∇Z e pontos candidatos')
plt.show()
```

* **Legenda**: as linhas das restrições e a “região viável” aparecem porque receberam `label`.
  Como dito, a **seta** com `plt.arrow` não entra na legenda (mesmo com `label`). Se quiser na legenda, troque `arrow` por `quiver`.

---

## 8) Interpretação do gráfico (fechamento)

* As duas **retas** delimitam, junto com os eixos, um **polígono** no 1º quadrante: essa é a **região viável**.
* O **gradiente** mostra para onde “puxar” as **retas de nível** $30x+50y=c$ para aumentar $Z$.
* O **ponto ótimo** é o vértice onde a maior isóclina ainda toca a região. Aqui, é a **interseção** $(18,16)$, com $Z=30\cdot 18 + 50\cdot 16 = 1340$.
* Os **outros vértices** (0,0), (30,0), (0,25) ajudam a confirmar que $(18,16)$ tem o maior $Z$.

---

## Anexo: seu código final (com título e lembretes)

```python
# Importamos as bibliotecas necessárias:
# - numpy para cálculos numéricos e vetoriais
# - matplotlib.pyplot para gerar os gráficos
import numpy as np
import matplotlib.pyplot as plt

# Função objetivo Z = 30x + 50y
# Dado um ponto (x, y), retorna o valor de Z nesse ponto.
# Usamos isso para anotar/avaliar os "pontos candidatos" (vértices da região viável).
def objective(x, y):  # Z = 30x + 50y
    return 30*x + 50*y

# Definimos limites dos eixos para a janela do gráfico.
# Não tem relação com as restrições; é apenas para “enquadrar” a visualização.
xmax = 50
ymax = 50

# Criamos um vetor de 400 valores igualmente espaçados no intervalo [0, xmax].
# Ele será usado para plotar as retas das restrições como funções de x.
x = np.linspace(0, xmax, 400)

# =====================================================================
# DEFINIÇÃO DAS RESTRIÇÕES (em forma de igualdade para traçar as retas)
# =====================================================================
# Cada equação abaixo vem de uma restrição linear do tipo <=, reescrita como igualdade:
# 1) 2x + 4y = 100  ->  isolando y: y = (100 - 2x)/4 = 25 - 0.5x
# 2) 4x + 3y = 120  ->  isolando y: y = (120 - 4x)/3 = 40 - (4/3)x
# Essas expressões nos permitem desenhar as linhas de fronteira das restrições.
y1 = 25 - 0.5*x           # 2x + 4y = 100
y2 = 40 - (4/3)*x         # 4x + 3y = 120

# Criamos a figura (tamanho 6x6) e configuramos grade para facilitar leitura.
plt.figure(figsize=(6, 6))
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Desenhamos as linhas (igualdades) das duas restrições com rótulos.
plt.plot(x, y1, label='2x + 4y = 100')
plt.plot(x, y2, label='4x + 3y = 120')

# =======================================
# CONSTRUÇÃO DA REGIÃO VIÁVEL (sombras)
# =======================================
# Para problemas típicos com restrições do tipo:
#   2x + 4y <= 100 e 4x + 3y <= 120, com x >= 0 e y >= 0,
# a região viável é o conjunto de pontos abaixo (ou sobre) de cada reta e no 1º quadrante.
#
# A ideia aqui: para cada x, pegamos o menor valor permitido de y dentre as duas retas
# (porque precisamos satisfazer ambas as restrições), e "preenchemos" de 0 até esse valor.
# Isso gera a faixa sombreadapara baixo das duas retas simultaneamente.
y_feasible = np.minimum(y1, y2)

# Preenchemos a área entre y=0 e y=y_feasible (onde ela é positiva).
# 'where=(y_feasible > 0)' evita colorir regiões onde y_feasible fica negativa.
plt.fill_between(x, 0, y_feasible, where=(y_feasible > 0),
                 color='gray', alpha=0.5, label='Região viável')

# ==========================================
# VETOR DIREÇÃO/GRADIENTE DA FUNÇÃO OBJETIVO
# ==========================================
# A função objetivo Z = 30x + 50y tem gradiente (30, 50).
# Desenhamos uma setinha a partir da origem só para indicar a direção de aumento de Z.
# Observação: plt.arrow não entra na legenda automaticamente (por isso não tem label).
plt.arrow(0, 0, 6, 10, head_width=1, head_length=1.5, fc='red', ec='red')  # label ignorado na legenda

# ==========================================
# INTERSEÇÃO DAS RETAS (vértice importante)
# ==========================================
# Para encontrar a interseção entre:
#   2x + 4y = 100
#   4x + 3y = 120
# resolvemos o sistema linear A*[x y]^T = b, com:
#   A = [[2, 4],
#        [4, 3]],  b = [100, 120]
A = np.array([[2, 4],
              [4, 3]], dtype=float)
b = np.array([100, 120], dtype=float)
intersection = np.linalg.solve(A, b)  # retorna [x_intersec, y_intersec]

# ==========================================
# PONTOS CANDIDATOS (CANTOS DA REGIÃO VIÁVEL)
# ==========================================
# Em Programação Linear, a solução ótima (se existir) ocorre em um vértice da região viável.
# Para este problema (com x,y >= 0), os vértices típicos são:
# - (0, 0)                         -> origem (interseção dos eixos)
# - Interseção de cada reta com os eixos:
#     2x + 4y = 100  -> (0,25) e (50,0)  [mas note que (50,0) pode ser "cortado" pela outra restrição]
#     4x + 3y = 120  -> (0,40) e (30,0)  [idem observação]
# - Interseção entre as duas retas (calculada acima).
#
# Aqui foram escolhidos quatro pontos candidatos que de fato pertencem à região viável:
#   (0,0), (30,0), (x*, y*) (interseção), (0,25)
# Observação: (50,0) e (0,40) ficam fora por violar a outra restrição.
pts = np.array([
    (0.0, 0.0),
    (30.0, 0.0),
    intersection,
    (0.0, 25.0),
], dtype=float)

# Desenhamos os pontos candidatos no gráfico.
plt.scatter(pts[:,0], pts[:,1], s=40, label='Pontos candidatos')

# Para cada ponto candidato, calculamos Z e criamos uma anotação com
# as coordenadas e o valor da função objetivo naquele ponto.
for px, py in pts:
    z = objective(px, py)
    plt.annotate(f'({px:.1f}, {py:.1f})\nZ={z:.0f}',
                 (px, py), xytext=(5,5), textcoords='offset points',
                 ha='left', va='bottom')

# Ajustes finais do gráfico: limites, rótulos, legenda e título.
plt.xlim(0, xmax); plt.ylim(0, ymax)
plt.xlabel('x'); plt.ylabel('y')
plt.legend()
plt.title('Restrições, região viável, ∇Z e pontos candidatos')

# Exibimos o gráfico.
plt.show()
```
---

## 9) Exercícios sugeridos

1. **Mude os coeficientes** da FO (ex.: `20x + 60y`) e explique por que o ponto ótimo se desloca.
2. **Remova uma restrição** e discuta se o problema fica **ilimitado**.
3. **Adicione** `x + y ≤ 30` e replote a região viável: qual é o novo ótimo?
4. Altere a janela do gráfico e discuta como **escolher limites** adequados ajuda a evitar interpretações erradas.
5. Substitua o preenchimento por **amostragem** da região (grid de pontos) e marque os que são viáveis — compare as abordagens.
