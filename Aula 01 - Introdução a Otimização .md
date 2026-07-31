# Desafio de Programação Linear com blocos

Nesta atividade, você irá analisar um problema de **Pesquisa Operacional / Programação Linear** utilizando blocos de encaixe.

O objetivo é decidir **quantas mesas** e **quantas cadeiras** devem ser produzidas para **maximizar o lucro**, respeitando a quantidade disponível de peças.

---

## 1. Produtos

Os dois produtos considerados nesta atividade são:
<p align="center">
 <img width="500"  alt="Sem título" src="https://github.com/user-attachments/assets/6c4bf383-6b09-4d40-9f45-0f71534c6761" />
</p>

### Composição dos produtos

- **Mesa**
  - 2 peças grandes
  - 2 peças pequenas
  - Lucro: **R$ 16,00**

- **Cadeira**
  - 1 peça grande
  - 2 peças pequenas
  - Lucro: **R$ 10,00**

---

## 2. Materiais disponíveis

A disponibilidade inicial de materiais é a seguinte:
<p align="center">
  <img width="800" alt="image" src="https://github.com/user-attachments/assets/bb22b1e1-9a57-4019-952b-f722f586d377" />
</p>
### Quantidades disponíveis

- **Peças grandes (amarelas): 6**
- **Peças pequenas (azuis): 8**

---

## 3. Enunciado do problema

Deseja-se produzir **mesas** e **cadeiras** utilizando as peças disponíveis.

Sabendo que:

- cada **mesa** gera **R$ 16,00** de lucro;
- cada **cadeira** gera **R$ 10,00** de lucro;

determine:

> **Quantas mesas e quantas cadeiras devem ser produzidas para maximizar o lucro?**

---

## 4. Variáveis de decisão

Considere:

- **x1** = número de mesas produzidas  
- **x2** = número de cadeiras produzidas  

---

## 5. Modelo matemático

### Função objetivo

Maximizar o lucro:

**Max Z = 16x1 + 10x2**

### Restrições

#### Restrição de peças grandes
Cada mesa consome **2 peças grandes** e cada cadeira consome **1 peça grande**.  
Como existem **6 peças grandes** disponíveis:

**2x1 + x2 <= 6**

#### Restrição de peças pequenas
Cada mesa consome **2 peças pequenas** e cada cadeira consome **2 peças pequenas**.  
Como existem **8 peças pequenas** disponíveis:

**2x1 + 2x2 <= 8**

#### Não negatividade

**x1 >= 0**  
**x2 >= 0**

---

## 6. Resolução gráfica

Abaixo está a representação gráfica do problema:
<p align="center">
<img width="500" alt="image" src="https://github.com/user-attachments/assets/c5bc142f-0d23-4706-98b6-dd67fc559873" />
</p>

### Interpretação

A região factível corresponde ao conjunto de soluções que respeitam simultaneamente todas as restrições do problema.

Os principais pontos da região factível são:

- **A = (0,0)**
- **B = (0,4)**
- **C = (2,2)**
- **D = (3,0)**

Agora, avaliando a função objetivo em cada ponto:

- **A = (0,0)**  
  Z = 16(0) + 10(0) = **0**

- **B = (0,4)**  
  Z = 16(0) + 10(4) = **40**

- **C = (2,2)**  
  Z = 16(2) + 10(2) = 32 + 20 = **52**

- **D = (3,0)**  
  Z = 16(3) + 10(0) = **48**

### Solução ótima

O maior valor obtido é:

**Z = 52**

Esse valor ocorre no ponto:

- **x1 = 2**
- **x2 = 2**

Portanto:

> A melhor solução do problema inicial é produzir **2 mesas** e **2 cadeiras**, obtendo lucro máximo de **R$ 52,00**.

---

# 7. Desafio final

Agora, considere uma nova situação.

A composição dos produtos continua a mesma:

- **Mesa** = 2 peças grandes + 2 peças pequenas  
- **Cadeira** = 1 peça grande + 2 peças pequenas  

Porém, a disponibilidade de materiais foi alterada.

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/f482bed8-38da-472c-ad3c-1ddb0f6b8e5b"
    alt="Imagem"
    width="800"
  />
</p>


### Nova disponibilidade

- **Peças grandes (amarelas): 8**
- **Peças pequenas (azuis): 10**

---

## 8. Sua tarefa

Com base na nova disponibilidade de peças, faça o seguinte:

1. Defina as variáveis de decisão;
2. Escreva a função objetivo;
3. Monte as restrições;
4. Resolva o problema;
5. Informe:
   - quantas mesas devem ser produzidas;
   - quantas cadeiras devem ser produzidas;
   - qual é o lucro máximo obtido.

---

## 9. Perguntas para orientar sua análise

- Qual recurso limita mais a produção: peças grandes ou peças pequenas?
- Vale a pena produzir apenas mesas?
- Vale a pena produzir apenas cadeiras?
- Existe uma combinação melhor entre os dois produtos?
- Qual é a solução ótima?

---

## 10. Entrega esperada

Sua resposta deve apresentar:

- o modelo matemático do problema;
- os cálculos realizados;
- a solução final;
- a interpretação da resposta em termos do problema.

---

## 11. Resposta final

Ao final, sua conclusão deve estar no formato:

> Devem ser produzidas **___ mesas** e **___ cadeiras**, obtendo lucro máximo de **R$ ___**.
