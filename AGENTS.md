# 第零章 格式模板和规范指南

这是一个适用于本项目的 `markdown` 格式模板，适用于所有以 `.md` 结尾的文档。文档的主要语言以中文为主，也存在部分英文作为辅助语言。所有的标题和内容均为示例，需要在实际的文档中替换成实际内容。项目整体采用模块化目录结构，每章拥有独立文件夹，文件夹内存放本章相关的所有文件。`.md` 文件推荐在编辑器，如 `Typora` 中以 `GitHub` 主题进行预览，保证最终部署网页版后在线阅读的兼容性和正确显示。

`markdown` 的基本语法和进阶使用请参考 [Markdown 教程](https://markdown.com.cn/)。




## 1 结构

如[表0-1](#tab0-1)所示，文档的结构由符号 `#` 定义，符合通用的 `markdown` 标准。建议采用三级标题体系，禁止出现孤立的三级标题。

章节标题之后需添加一段无标题的序言，概括本章的核心内容。

每个标题下的段落的长度区间需要合理，不宜过长或过短，适合学习者阅读并保持注意力集中。如果检查时发现过长或过短，可以考虑拆分段落或合并段落。每个一级标题段落前后需要空行。<span id="tab0-1"> </span>

<div align="center">
  <p>表0-1 文档结构层级</p>
  <table>
    <thead>
      <tr>
        <th>符号</th>
        <th>结构层级</th>
        <th>示例标题</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>#</td>
        <td>章节标题</td>
        <td>第一章 引言</td>
      </tr>
      <tr>
        <td>##</td>
        <td>一级标题</td>
        <td>1 示例一级标题</td>
      </tr>
      <tr>
        <td>###</td>
        <td>二级标题</td>
        <td>1.2 示例二级标题</td>
      </tr>
      <tr>
        <td>####</td>
        <td>三级标题</td>
        <td>1.3.3 示例三级标题</td>
      </tr>
    </tbody>
  </table>
</div>



## 2 内容

文字内容的部分是文档的核心，文档的主要语言以中文为主，也存在部分英文作为辅助语言。所有英文的部分需要使用反引号进行限定，例如 `transformer`, `this is an English sentence`。当存在中英文交替的时候，位于句中的英文需要和两侧的中文存在一个空格的距离；若存在英文两侧存在标点或符号，则存在标点或符号的一侧不需要空格。此外，段落中如果需要首行缩进，使用符号 `&emsp;` 实现。

内容的语言表达中，统一使用“我们”作为叙述主体，并且注意表达的逻辑清晰和书面化，避免长难句的使用。

对于需要强调的正文中的文字部分，统一使用 `<strong> </strong>` 包裹；正文中出现的引用标记，统一使用 `<sup> </sup>` 进行包裹。例如，<strong>被强调的文字部分</strong>请参考<sup>[[*]](# 7 参考文献)</sup>



## 3 图像

图像统一使用 `png` 格式或 `gif` 格式，保证图像的分辨率足够清晰。图中若存在文字，中文使用宋体，英文使用 `Times New Roman`。

如[图0.1](#fig0.1)所示，每个图像必须包含 `caption`，位于图像的下方居中，`caption`  的内容准确概括图像的核心信息。如果存在图像，则正文中必须包含对应的图像引用。每个章节的图像编号独立，形式为 `x.y`，`x` 为章节的编号，`y` 从 $1$ 开始，逐次递增。<span id="fig0.1"> </span>

<div align="center">
  <img src="./assets/figure.png" width="90%"/>
  <span id="Figure 1"><p>图0.1 示例图像</p></span>
</div>



## 4 表格

如[表0-2](#tab0-2)所示，每个表格必须包含 `caption`，位于表格的上方居中，`caption`  的内容描述概括表格的核心信息。如果存在表格，则正文中必须包含对应的表格引用。每个章节的表格编号独立，形式为 `x.y`，`x` 为章节的编号，`y` 从 $1$ 开始，逐次递增。<span id="tab0-2"> </span>

<div align="center">
  <p>表0-2 示例表格标题</p>
  <table>
    <thead>
      <tr>
        <th>列1</th>
        <th>列2</th>
        <th>列3</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>数据1</td>
        <td>数据2</td>
        <td>数据3</td>
      </tr>
      <tr>
        <td>数据4</td>
        <td>数据5</td>
        <td>数据6</td>
      </tr>
    </tbody>
  </table>
</div>


## 5 代码

代码块是文档中的重要组成部分，用于展示具体的实现细节和示例。代码块需要使用三重反引号包裹，并在开头标注代码语言类型，以便实现语法高亮。

### 5.1 代码块格式

代码块的基本格式如下，需要在开头的三重反引号后紧跟语言标识符（如 `python`、`bash`、`json` 等）：

```python
def hello_world():
    print("Hello, World!")
```

### 5.2 代码风格

代码风格统一遵循 [`Google Python Style Guide`](https://google.github.io/styleguide/pyguide.html)，包括函数文档字符串、类型注解和返回类型说明。

#### 5.2.1 函数文档字符串

函数必须包含文档字符串（`docstring`），使用三引号 `"""` 包裹，包含函数描述、参数说明（`Args`）、返回值说明（`Returns`）和异常说明（`Raises`，如有）：

```python
def train_model(model: torch.nn.Module, 
                dataloader: torch.utils.data.DataLoader,
                epochs: int = 10,
                learning_rate: float = 1e-4) -> dict:
    """训练深度学习模型。
    
    Args:
        model: 需要训练的神经网络模型。
        dataloader: 训练数据加载器，包含批次化的训练数据。
        epochs: 训练的轮数，默认为 10。
        learning_rate: 学习率，默认为 1e-4。
    
    Returns:
        包含训练历史的字典，包括每个 epoch 的损失值和准确率。
        示例：{'loss': [0.5, 0.3, 0.2], 'accuracy': [0.8, 0.85, 0.9]}
    
    Raises:
        ValueError: 如果 epochs 小于 1。
        RuntimeError: 如果模型训练过程中出现错误。
    """
    if epochs < 1:
        raise ValueError("epochs 必须大于等于 1")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {'loss': [], 'accuracy': []}
    
    for epoch in range(epochs):
        loss, acc = train_one_epoch(model, dataloader, optimizer)
        history['loss'].append(loss)
        history['accuracy'].append(acc)
    
    return history
```

#### 5.2.2 类型注解

所有函数参数和返回值都应该包含类型注解，使用 `Python` 的类型提示（`Type Hints`）：

```python
from typing import List, Tuple, Optional, Dict, Any

def process_batch(inputs: torch.Tensor, 
                  labels: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, float]:
    """处理单个批次的数据。
    
    Args:
        inputs: 输入张量，形状为 (batch_size, seq_len, hidden_dim)。
        labels: 标签张量，形状为 (batch_size, seq_len)。
        mask: 可选的掩码张量，形状为 (batch_size, seq_len)。
    
    Returns:
        包含两个元素的元组：
        - 输出张量，形状为 (batch_size, seq_len, vocab_size)
        - 损失值（标量）
    """
    output = model(inputs, attention_mask=mask)
    loss = criterion(output, labels)
    return output, loss.item()
```

#### 5.2.3 类的文档字符串

类也需要包含文档字符串，说明类的用途和属性：

```python
class TransformerModel(nn.Module):
    """基于 Transformer 架构的语言模型。
    
    该模型实现了标准的 Transformer 编码器-解码器结构，
    适用于序列到序列的任务。
    
    Attributes:
        vocab_size: 词汇表大小。
        d_model: 模型的隐藏层维度。
        nhead: 多头注意力的头数。
        num_layers: Transformer 层数。
        embedding: 词嵌入层。
        transformer: Transformer 主体结构。
    """
    
    def __init__(self, 
                 vocab_size: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6) -> None:
        """初始化 Transformer 模型。
        
        Args:
            vocab_size: 词汇表大小。
            d_model: 模型的隐藏层维度，默认为 512。
            nhead: 多头注意力的头数，默认为 8。
            num_layers: Transformer 层数，默认为 6。
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """前向传播。
        
        Args:
            src: 源序列张量，形状为 (seq_len, batch_size)。
            tgt: 目标序列张量，形状为 (seq_len, batch_size)。
        
        Returns:
            输出张量，形状为 (seq_len, batch_size, vocab_size)。
        """
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)
        output = self.transformer(src_emb, tgt_emb)
        return output
```

### 5.3 行内代码

在正文中引用代码片段、函数名、变量名、命令等时，使用单个反引号包裹。例如：调用 `torch.nn.Module` 类的 `forward()` 方法来实现前向传播。

### 5.4 命令行代码

对于命令行操作，统一使用 `bash`  作为语言标识符：

```bash
pip install -r requirements.txt
python train.py --epochs 10 --batch_size 32
```

### 5.5 代码块长度

文档中代码块的长度应适中，一般不超过 $50$ 行。如果代码较长，可以：

- 只展示关键部分，例如使用注释 `# ... 省略部分代码 ...` 表示省略
- 将完整代码放在项目的 `code/` 目录下，并在文档中进行引用



## 6 公式

公式是表达数学概念和算法原理的重要方式。在 `Markdown` 中，公式使用 `LaTeX` 语法编写，支持行内公式和独立公式两种形式。

### 6.1 行内公式

行内公式使用单符号 `$` 包裹，嵌入在文本段落中。例如：模型的损失函数定义为 $L = -\sum_{i=1}^{n} y_i \log(\hat{y}_i)$，其中 $y_i$ 表示真实标签，$\hat{y}_i$ 表示预测标签。

### 6.2 独立公式

独立公式使用双美元符号 `$$` 包裹，单独成行并居中显示。独立公式应在正文中被引用，并在必要时添加编号。例如：

`Transformer` 模型的自注意力机制计算公式如下：

$$
{\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V}
$$

其中，$Q、K、V$ 分别表示查询（`Query`）、键（`Key`）和值（`Value`）矩阵，$d_k$ 是键向量的维度。

### 6.3 公式编号

对于需要在正文中多次引用的重要公式，应添加编号。编号格式为 `(x.y)`，其中 `x` 为章节编号，`y` 从 $1$ 开始递增。例如：

多头注意力机制的计算公式为：<span id="eq0.1"> </span>

$$
{\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O}\tag{0.1}
$$

其中每个注意力头的计算方式为：<span id="eq0.2"> </span>

$$
{\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)\tag{0.2}}
$$

在正文中可以引用公式 [(0.1)](#eq0.1) 和公式 [(0.2)](#eq0.2) 来说明多头注意力的数学原理和计算过程。

### 6.4 常用数学符号

在编写公式时，使用标准的 `LaTeX` 符号。参考 `Deep Learning Book Notation`<sup>[[1]](#ref1)</sup> 的符号规范，建议在所有文档中统一使用以下常用数学符号表示方法。

#### 6.4.1 数字和数组

如[表0-3](#tab0-3)所示，标量、向量、矩阵和张量的表示方法。<span id="tab0-3"> </span>
<div align="center">
	<p>表0-3 数字和数组符号</p>
</div>

<div align="center">

| 符号 | LaTeX 代码 | 说明 |
| :--- | :--- | :--- |
| $a$ | `a` | 标量（整数或实数） |
| $\mathbf{a}$ | `\mathbf{a}` | 向量 |
| $\mathbf{A}$ | `\mathbf{A}` | 矩阵 |
| $\mathsf{A}$ | `\mathsf{A}` | 张量 |
| $\mathbf{I}_n$ | `\mathbf{I}_n` | $n$ 行 $n$ 列的单位矩阵 |
| $\mathbf{e}^{(i)}$ | `\mathbf{e}^{(i)}` | 标准基向量，第 $i$ 个位置为 $1$ |

</div>

#### 6.4.2 集合和索引

如[表0-4](#tab0-4)所示，集合和索引的表示方法。<span id="tab0-4"> </span>

<div align="center">
	<p>表0-4 集合和索引符号</p>
</div>

<div align="center">

| 符号 | LaTeX 代码 | 说明 |
| :--- | :--- | :--- |
| $\mathbb{R}$ | `\mathbb{R}` | 实数集 |
| $\mathbb{N}$ | `\mathbb{N}` | 自然数集 |
| $\mathbb{Z}$ | `\mathbb{Z}` | 整数集 |
| $\mathcal{D}$ | `\mathcal{D}` | 数据集（花体） |
| $[a, b]$ | `[a, b]` | 闭区间，包含 $a$ 和 $b$ |
| $(a, b]$ | `(a, b]` | 半开区间，不包含 $a$ |
| $a_i$ | `a_i` | 向量 $a$ 的第 $i$ 个元素 |
| $\mathbf{A}_{i,j}$ | `\mathbf{A}_{i,j}` | 矩阵 $\mathbf{A}$ 的第 $(i,j)$ 个元素 |
| $\mathbf{A}_{i,:}$ | `\mathbf{A}_{i,:}` | 矩阵 $\mathbf{A}$ 的第 $i$ 行 |
| $\mathbf{A}_{:,j}$ | `\mathbf{A}_{:,j}` | 矩阵 $\mathbf{A}$ 的第 $j$ 列 |

</div>

#### 6.4.3 线性代数运算

如[表0-5](#tab0-5)所示，常用的线性代数运算符号。<span id="tab0-5"> </span>

<div align="center">
	<p>表0-5 线性代数运算符号</p>
</div>

<div align="center">

| 符号 | LaTeX 代码 | 说明 |
| :--- | :--- | :--- |
| $\mathbf{A}^\top$ | `\mathbf{A}^\top` | 矩阵 $\mathbf{A}$ 的转置 |
| $\mathbf{A}^{-1}$ | `\mathbf{A}^{-1}` | 矩阵 $\mathbf{A}$ 的逆 |
| $\mathbf{A} \odot \mathbf{B}$ | `\mathbf{A} \odot \mathbf{B}` | 逐元素乘积（`Hadamard` 积） |
| $\det(\mathbf{A})$ | `\det(\mathbf{A})` | 矩阵 $\mathbf{A}$ 的行列式 |
| $\|\mathbf{x}\|_2$ | `\|\mathbf{x}\|_2` | $L^2$ 范数 |
| $\|\mathbf{x}\|_p$ | `\|\mathbf{x}\|_p` | $L^p$ 范数 |

</div>

#### 6.4.4 微积分

如[表0-6](#tab0-6)所示，微积分相关的符号。<span id="tab0-6"> </span>

<div align="center">
	<p>表0-6 微积分符号</p>
</div>

<div align="center">

| 符号 | LaTeX 代码 | 说明 |
| :--- | :--- | :--- |
| $\frac{dy}{dx}$ | `\frac{dy}{dx}` | $y$ 对 $x$ 的导数 |
| $\frac{\partial y}{\partial x}$ | `\frac{\partial y}{\partial x}` | $y$ 对 $x$ 的偏导数 |
| $\nabla_{\mathbf{x}} y$ | `\nabla_{\mathbf{x}} y$ | $y$ 对 $\mathbf{x}$ 的梯度 |
| $\nabla^2_{\mathbf{x}} f(\mathbf{x})$ | `\nabla^2_{\mathbf{x}} f(\mathbf{x})` | $f$ 在 $\mathbf{x}$ 处的 `Hessian` 矩阵 |
| $\int f(x)dx$ | `\int f(x)dx` | $f(x)$ 在 $x$ 上的定积分 |
| $\sum_{i=1}^{n}$ | `\sum_{i=1}^{n}$ | 求和 |
| $\prod_{i=1}^{n}$ | `\prod_{i=1}^{n}` | 连乘 |

</div>

#### 6.4.5 概率和信息论


如[表0-7](#tab0-7)所示，概率和信息论相关的符号。<span id="tab0-7"> </span>

<div align="center">
	<p>表0-7 概率和信息论符号</p>
</div>

<div align="center">

| 符号 | LaTeX 代码 | 说明 |
| :--- | :--- | :--- |
| $P(a)$ | `P(a)` | 离散变量的概率分布 |
| $p(a)$ | `p(a)` | 连续变量的概率密度 |
| $a \sim P$ | `a \sim P` | 随机变量 $a$ 服从分布 $P$ |
| $\mathbb{E}_{x \sim P}[f(x)]$ | `\mathbb{E}_{x \sim P}[f(x)]` | $f(x)$ 关于 $P(x)$ 的期望 |
| $\text{Var}(f(x))$ | `\text{Var}(f(x))` | $f(x)$ 的方差 |
| $\text{Cov}(f(x), g(x))$ | `\text{Cov}(f(x), g(x))` | $f(x)$ 和 $g(x)$ 的协方差 |
| $H(x)$ | `H(x)` | 随机变量 $x$ 的香农熵 |
| $D_{\text{KL}}(P\|Q)$ | `D_{\text{KL}}(P\|Q)` | $P$ 和 $Q$ 的 `KL` 散度 |
| $\mathcal{N}(x; \mu, \sigma^2)$ | `\mathcal{N}(x; \mu, \sigma^2)` | 均值为 $\mu$，方差为 $\sigma^2$ 的高斯分布 |

</div>

#### 6.4.6 常用函数

如[表0-8](#tab0-8)所示，常用的数学函数。<span id="tab0-8"> </span>

<div align="center">
	<p>表0-8 常用函数符号</p>
</div>

<div align="center">

| 符号 | LaTeX 代码 | 说明 |
| :--- | :--- | :--- |
| $\log x$ | `\log x` | 自然对数 |
| $\exp(x)$ | `\exp(x)` | 指数函数 |
| $\sigma(x)$ | `\sigma(x)` | `Sigmoid` 函数 |
| $\text{softmax}(x)$ | `\text{softmax}(x)` | `Softmax` 函数 |
| $\max(x, y)$ | `\max(x, y)` | 最大值函数 |
| $\arg\max_x f(x)$ | `\arg\max_x f(x)` | 使 $f(x)$ 最大的 $x$ 值 |
| $\arg\min_x f(x)$ | `\arg\min_x f(x)` | 使 $f(x)$ 最小的 $x$ 值 |

</div>


### 6.5 矩阵和向量

矩阵和向量的表示使用 `LaTeX` 的矩阵环境。例如：

权重矩阵 $\mathbf{W}$ 和输入向量 $\mathbf{x}$ 的乘积为：

$$
\mathbf{y} = \mathbf{W}\mathbf{x} = \begin{bmatrix}
w_{11} & w_{12} & \cdots & w_{1n} \\
w_{21} & w_{22} & \cdots & w_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
w_{m1} & w_{m2} & \cdots & w_{mn}
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{bmatrix}
$$

### 6.6 公式排版规范

在书写公式时，应注意以下规范：

- 公式中的变量使用斜体，例如 $x, y, z$
- 函数名使用正体，例如 $\text{softmax}, \log, \exp, \text{Attention}$
- 向量和矩阵使用粗体，例如 $\mathbf{x}, \mathbf{W}$
- 集合使用花体或空心体，例如 $\mathcal{D}, \mathbb{R}$
- 复杂公式应适当换行，保证良好的可读性 



## 7 参考文献

参考文献需要遵循 [`APA Style`](https://apastyle.apa.org/style-grammar-guidelines/references), 一般的链接引用可以直接在文档行间使用 `markdown` 语法完成，但是如果涉及到出版物或者论文的正式引用，则需要添加引用标记及其对应的参考文献。

<span id="ref1">[1]</span> Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. https://www.deeplearningbook.org/contents/notation.html