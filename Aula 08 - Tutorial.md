# Tutorial — Resolução Gráfica de um Problema de Programação Linear com Python

## Exemplo 1 — Produção de móveis

Neste tutorial vamos utilizar Python para representar graficamente e resolver um problema de Programação Linear com duas variáveis de decisão.

O objetivo é reproduzir, com auxílio computacional, as mesmas etapas utilizadas na resolução pelo **método gráfico**:

1. identificar as variáveis de decisão;
2. construir o modelo matemático;
3. representar as restrições no plano cartesiano;
4. identificar a região viável;
5. determinar os vértices da região viável;
6. avaliar a função objetivo nos vértices;
7. interpretar a solução obtida.

> **Importante:** o Python será utilizado como apoio para visualizar e calcular o problema. A lógica de resolução continua sendo a mesma do método gráfico apresentado em aula.

---

## 1. Enunciado

Uma pequena fábrica produz **mesas** e **estantes**. Cada mesa proporciona lucro de **R$ 80,00** e cada estante lucro de **R$ 60,00**. Para produzir uma mesa são necessárias 4 horas de marcenaria e 2 horas de acabamento. Para uma estante são necessárias 3 horas de marcenaria e 4 horas de acabamento. Durante a próxima semana, a empresa dispõe de, no máximo, 240 horas de marcenaria e 160 horas de acabamento. Além disso, devido à previsão de demanda, poderão ser comercializadas no máximo 50 mesas.

Determine quantas mesas e estantes devem ser produzidas para **maximizar o lucro semanal**.

---

## 2. Definindo as variáveis de decisão

Vamos representar as quantidades produzidas por:

$$
x = \text{quantidade de mesas}
$$

$$
y = \text{quantidade de estantes}
$$

Como não é possível produzir uma quantidade negativa de móveis:

$$
x \geq 0
$$

$$
y \geq 0
$$

---

## 3. Construindo a função objetivo

Cada mesa gera lucro de R$ 80,00 e cada estante gera lucro de R$ 60,00.

Portanto, queremos maximizar:

$$
Z = 80x + 60y
$$

Logo:

$$
\boxed{\max Z = 80x + 60y}
$$

---

## 4. Construindo as restrições

### 4.1. Horas de marcenaria

Cada mesa utiliza 4 horas de marcenaria e cada estante utiliza 3 horas.

Existem no máximo 240 horas disponíveis:

$$
4x + 3y \leq 240
$$

### 4.2. Horas de acabamento

Cada mesa utiliza 2 horas de acabamento e cada estante utiliza 4 horas.

Existem no máximo 160 horas disponíveis:

$$
2x + 4y \leq 160
$$

### 4.3. Limite de demanda

A empresa pode comercializar no máximo 50 mesas:

$$
x \leq 50
$$

### 4.4. Não negatividade

$$
x \geq 0
$$

$$
y \geq 0
$$

---

## 5. Modelo matemático completo

O problema pode ser escrito como:

$$
\max Z = 80x + 60y
$$

sujeito a:

$$
4x + 3y \leq 240
$$

$$
2x + 4y \leq 160
$$

$$
x \leq 50
$$

$$
x,y \geq 0
$$

Agora vamos representar esse modelo graficamente utilizando Python.

---

# 6. Preparando o ambiente Python

O exemplo pode ser executado no **Google Colab**, **Jupyter Notebook**, **VS Code** ou em qualquer ambiente Python que possua as bibliotecas `numpy` e `matplotlib`.

No Google Colab essas bibliotecas normalmente já estão instaladas.

Caso seja necessário instalá-las em seu computador, execute:

```bash
pip install numpy matplotlib
```

Depois importe as bibliotecas:

```python
import numpy as np
import matplotlib.pyplot as plt
```

---

# 7. Transformando as restrições em retas

Para desenhar uma restrição no gráfico, inicialmente substituímos a desigualdade por uma igualdade.

## Restrição de marcenaria

Temos:

$$
4x + 3y = 240
$$

Isolando \(y\):

$$
3y = 240 - 4x
$$

$$
y = \frac{240 - 4x}{3}
$$

Em Python:

```python
x = np.linspace(0, 65, 500)

y_marcenaria = (240 - 4*x) / 3
```

---

## Restrição de acabamento

Temos:

$$
2x + 4y = 160
$$

Isolando \(y\):

$$
4y = 160 - 2x
$$

$$
y = \frac{160 - 2x}{4}
$$

Em Python:

```python
y_acabamento = (160 - 2*x) / 4
```

---

## Restrição de demanda

A terceira restrição é:

$$
x \leq 50
$$

Sua fronteira é uma reta vertical:

$$
x = 50
$$

No Matplotlib podemos desenhá-la com:

```python
plt.axvline(x=50, label="Demanda: x = 50")
```

---

# 8. Primeiro gráfico das restrições

Vamos desenhar as três fronteiras:

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 65, 500)

y_marcenaria = (240 - 4*x) / 3
y_acabamento = (160 - 2*x) / 4

plt.figure(figsize=(10, 7))

plt.plot(x, y_marcenaria, label="Marcenaria: 4x + 3y = 240")
plt.plot(x, y_acabamento, label="Acabamento: 2x + 4y = 160")
plt.axvline(x=50, label="Demanda: x = 50")

plt.xlim(0, 65)
plt.ylim(0, 85)

plt.xlabel("Mesas (x)")
plt.ylabel("Estantes (y)")
plt.title("Restrições do problema")

plt.grid(True)
plt.legend()

plt.show()
```

Ao executar o código, observe onde as retas se cruzam e qual lado de cada reta atende às desigualdades.

---

# 9. Identificando a região viável

Todas as restrições utilizam `<=`. Portanto, precisamos encontrar a região que esteja simultaneamente:

$$
4x + 3y \leq 240
$$

$$
2x + 4y \leq 160
$$

$$
x \leq 50
$$

e também:

$$
x \geq 0,\qquad y \geq 0
$$

A interseção dessas condições forma a **região viável**.

Para destacá-la no gráfico, precisamos conhecer seus vértices.

---

# 10. Encontrando os pontos de interseção

## 10.1. Interseção entre marcenaria e acabamento

As duas equações são:

$$
4x + 3y = 240
$$

$$
2x + 4y = 160
$$

Podemos resolver esse sistema com `numpy.linalg.solve`.

```python
A = np.array([
    [4, 3],
    [2, 4]
])

b = np.array([240, 160])

ponto = np.linalg.solve(A, b)

print(ponto)
```

Resultado:

```text
[48. 16.]
```

Logo, as duas restrições se cruzam em:

$$
(48,16)
$$

---

## 10.2. Interseção entre a demanda e a marcenaria

Para \(x=50\):

$$
4(50)+3y=240
$$

$$
200+3y=240
$$

$$
y=\frac{40}{3}
$$

$$
y\approx 13,33
$$

Assim, temos o ponto:

$$
(50,13,33)
$$

---

## 10.3. Interseções com os eixos

Quando \(x=0\), a restrição de acabamento resulta em:

$$
4y=160
$$

$$
y=40
$$

Portanto:

$$
(0,40)
$$

No eixo \(x\), o limite de demanda determina:

$$
(50,0)
$$

Também temos a origem:

$$
(0,0)
$$

---

# 11. Vértices da região viável

Os vértices da região viável são:

| Vértice | x | y |
|---|---:|---:|
| A | 0 | 0 |
| B | 50 | 0 |
| C | 50 | 13,33 |
| D | 48 | 16 |
| E | 0 | 40 |

Esses são os pontos candidatos à solução ótima.

---

# 12. Plotando a região viável

Agora podemos preencher o polígono formado pelos vértices:

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 65, 500)

y_marcenaria = (240 - 4*x) / 3
y_acabamento = (160 - 2*x) / 4

vertices_x = [0, 50, 50, 48, 0]
vertices_y = [0, 0, 40/3, 16, 40]

plt.figure(figsize=(10, 7))

plt.plot(x, y_marcenaria, label="Marcenaria: 4x + 3y = 240")
plt.plot(x, y_acabamento, label="Acabamento: 2x + 4y = 160")
plt.axvline(x=50, label="Demanda: x = 50")

plt.fill(
    vertices_x,
    vertices_y,
    alpha=0.25,
    label="Região viável"
)

plt.scatter(vertices_x, vertices_y)

for vx, vy in zip(vertices_x, vertices_y):
    plt.annotate(
        f"({vx:.0f}, {vy:.2f})",
        (vx, vy),
        textcoords="offset points",
        xytext=(5, 5)
    )

plt.xlim(0, 65)
plt.ylim(0, 85)

plt.xlabel("Mesas (x)")
plt.ylabel("Estantes (y)")
plt.title("Região viável")

plt.grid(True)
plt.legend()

plt.show()
```

A área preenchida representa todas as combinações de mesas e estantes que respeitam simultaneamente as restrições do problema.

---

# 13. Avaliando a função objetivo

A função objetivo é:

$$
Z = 80x + 60y
$$

Pelo método dos vértices, calculamos \(Z\) em cada ponto extremo da região viável.

Podemos fazer isso com Python:

```python
vertices = [
    (0, 0),
    (50, 0),
    (50, 40/3),
    (48, 16),
    (0, 40)
]

for x, y in vertices:
    lucro = 80*x + 60*y
    print(f"x = {x:.2f}, y = {y:.2f} -> Lucro = R$ {lucro:.2f}")
```

Saída esperada:

```text
x = 0.00,  y = 0.00  -> Lucro = R$ 0.00
x = 50.00, y = 0.00  -> Lucro = R$ 4000.00
x = 50.00, y = 13.33 -> Lucro = R$ 4800.00
x = 48.00, y = 16.00 -> Lucro = R$ 4800.00
x = 0.00,  y = 40.00 -> Lucro = R$ 2400.00
```

Observe que **dois vértices apresentam o mesmo lucro máximo de R$ 4.800,00**.

---

# 14. Um detalhe importante: múltiplas soluções ótimas

Neste problema ocorre uma situação especial.

A função objetivo é:

$$
Z = 80x + 60y
$$

Podemos colocar 20 em evidência:

$$
Z = 20(4x+3y)
$$

Compare com a restrição de marcenaria:

$$
4x+3y\leq240
$$

Isso significa que as retas da função objetivo possuem a **mesma inclinação** da fronteira da restrição de marcenaria.

Consequentemente, não existe apenas um ponto ótimo.

Todo ponto da fronteira:

$$
4x+3y=240
$$

que esteja entre:

$$
(48,16)
$$

e

$$
(50,13,33)
$$

é uma solução ótima.

Todos esses pontos produzem:

$$
Z=4800
$$

Portanto:

$$
\boxed{Z_{\max}=R\$\,4.800,00}
$$

e existem **múltiplas soluções ótimas**.

---

# 15. Visualizando a função objetivo

Podemos incluir no gráfico uma das retas da função objetivo para o lucro ótimo.

Sabemos que:

$$
80x+60y=4800
$$

Isolando \(y\):

$$
60y=4800-80x
$$

$$
y=\frac{4800-80x}{60}
$$

Adicione ao gráfico:

```python
y_objetivo = (4800 - 80*x) / 60

plt.plot(
    x,
    y_objetivo,
    "--",
    linewidth=2,
    label="Função objetivo: Z = 4800"
)
```

O gráfico completo fica:

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 65, 500)

y_marcenaria = (240 - 4*x) / 3
y_acabamento = (160 - 2*x) / 4
y_objetivo = (4800 - 80*x) / 60

vertices_x = [0, 50, 50, 48, 0]
vertices_y = [0, 0, 40/3, 16, 40]

plt.figure(figsize=(10, 7))

plt.plot(x, y_marcenaria, label="Marcenaria: 4x + 3y = 240")
plt.plot(x, y_acabamento, label="Acabamento: 2x + 4y = 160")
plt.axvline(x=50, label="Demanda: x = 50")

plt.plot(
    x,
    y_objetivo,
    "--",
    linewidth=2,
    label="Função objetivo: Z = 4800"
)

plt.fill(
    vertices_x,
    vertices_y,
    alpha=0.25,
    label="Região viável"
)

plt.scatter(vertices_x, vertices_y)

for vx, vy in zip(vertices_x, vertices_y):
    plt.annotate(
        f"({vx:.0f}, {vy:.2f})",
        (vx, vy),
        textcoords="offset points",
        xytext=(5, 5)
    )

plt.xlim(0, 65)
plt.ylim(0, 85)

plt.xlabel("Mesas (x)")
plt.ylabel("Estantes (y)")
plt.title("Resolução gráfica do problema")

plt.grid(True)
plt.legend()

plt.show()
```

Observe que a reta da função objetivo ótima coincide com uma parte da fronteira da restrição de marcenaria. Essa é a representação gráfica das múltiplas soluções ótimas.

---

# 16. Verificando automaticamente se um ponto é viável

Também podemos criar uma função que verifica se determinada combinação de produção respeita todas as restrições:

```python
def eh_viavel(x, y):
    return (
        4*x + 3*y <= 240 and
        2*x + 4*y <= 160 and
        x <= 50 and
        x >= 0 and
        y >= 0
    )
```

Por exemplo:

```python
print(eh_viavel(48, 16))
print(eh_viavel(60, 10))
```

Resultado:

```text
True
False
```

O ponto `(48,16)` é viável, enquanto `(60,10)` viola pelo menos uma das restrições.

---

# 17. Código completo

Abaixo está uma versão compacta que reúne toda a resolução:

```python
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------
# 1. Valores para o eixo x
# ---------------------------------

x = np.linspace(0, 65, 500)

# ---------------------------------
# 2. Restrições
# ---------------------------------

y_marcenaria = (240 - 4*x) / 3
y_acabamento = (160 - 2*x) / 4

# ---------------------------------
# 3. Vértices da região viável
# ---------------------------------

vertices = [
    (0, 0),
    (50, 0),
    (50, 40/3),
    (48, 16),
    (0, 40)
]

vertices_x = [p[0] for p in vertices]
vertices_y = [p[1] for p in vertices]

# ---------------------------------
# 4. Avaliação da função objetivo
# ---------------------------------

print("Avaliação dos vértices:")

for vx, vy in vertices:
    lucro = 80*vx + 60*vy
    print(
        f"x = {vx:.2f}, "
        f"y = {vy:.2f}, "
        f"Z = R$ {lucro:.2f}"
    )

# ---------------------------------
# 5. Função objetivo ótima
# ---------------------------------

z_otimo = 4800
y_objetivo = (z_otimo - 80*x) / 60

# ---------------------------------
# 6. Construção do gráfico
# ---------------------------------

plt.figure(figsize=(10, 7))

plt.plot(
    x,
    y_marcenaria,
    label="Marcenaria: 4x + 3y = 240"
)

plt.plot(
    x,
    y_acabamento,
    label="Acabamento: 2x + 4y = 160"
)

plt.axvline(
    x=50,
    label="Demanda: x = 50"
)

plt.plot(
    x,
    y_objetivo,
    "--",
    linewidth=2,
    label="Função objetivo: Z = 4800"
)

plt.fill(
    vertices_x,
    vertices_y,
    alpha=0.25,
    label="Região viável"
)

plt.scatter(vertices_x, vertices_y)

for vx, vy in vertices:
    plt.annotate(
        f"({vx:.0f}, {vy:.2f})",
        (vx, vy),
        textcoords="offset points",
        xytext=(5, 5)
    )

plt.xlim(0, 65)
plt.ylim(0, 85)

plt.xlabel("Mesas (x)")
plt.ylabel("Estantes (y)")
plt.title("Programação Linear — Método Gráfico")

plt.grid(True)
plt.legend()

plt.show()
```

---

# 18. Interpretação da solução

O maior lucro possível é:

$$
\boxed{R\$\,4.800,00}
$$

Entretanto, neste exemplo existem **múltiplas soluções ótimas**.

Duas delas são:

$$
x=48,\qquad y=16
$$

e

$$
x=50,\qquad y\approx13,33
$$

Além desses dois vértices, todos os pontos do segmento que os conecta também apresentam lucro de R$ 4.800,00.

Assim, considerando o modelo como um problema de Programação Linear com variáveis contínuas, a empresa pode escolher qualquer combinação pertencente a esse segmento.

> **Observação sobre quantidades inteiras:** como mesas e estantes normalmente são produzidas em unidades inteiras, poderíamos acrescentar essa condição ao problema. Entretanto, isso transforma o modelo em um problema de Programação Linear Inteira. Neste tutorial estamos trabalhando com o método gráfico clássico de Programação Linear.

---

# 19. Exercício para o aluno

Após executar o código, faça as seguintes alterações e observe o que acontece:

1. Modifique o lucro de uma mesa de **R$ 80,00 para R$ 90,00**.
2. Mantenha o lucro da estante em **R$ 60,00**.
3. Recalcule a função objetivo nos vértices.
4. Identifique a nova solução ótima.
5. Desenhe a nova reta da função objetivo.
6. Explique por que agora a solução ótima deixa de ocorrer ao longo de um segmento.

A nova função objetivo será:

$$
\max Z=90x+60y
$$

Compare sua inclinação com a inclinação da restrição de marcenaria.

---

## Conclusão

Com Python conseguimos reproduzir todas as principais etapas do método gráfico:

- representar as restrições;
- visualizar a região viável;
- calcular pontos de interseção;
- identificar os vértices;
- avaliar a função objetivo;
- representar graficamente a solução ótima;
- identificar situações especiais, como múltiplas soluções ótimas.

Nos próximos exercícios, a mesma estrutura pode ser reutilizada alterando apenas a função objetivo, as equações das restrições e os vértices da região viável.
