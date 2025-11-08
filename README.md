 <div align='center'>
    <img src="assets/logo.png" alt="logo" width="800"/>
  <h1>Code Your Own LLM</h1>
  <h3>动手训练你自己的大语言模型</h3>
  <p><em>这是一份全栈式的参考指南，用最简洁的代码帮助你端到端定义大语言模型的每一个细节</em></p>
  <img src="https://img.shields.io/github/stars/datawhalechina/code-your-own-llm?style=flat&logo=github" alt="GitHub stars"/>
  <a rel="License" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="License" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python"/>
  <a href="https://datawhalechina.github.io/code-your-own-llm/"><img src="https://img.shields.io/badge/在线阅读-Online%20Reading-green?style=flat&logo=gitbook" alt="Online Reading"></a>
</div>



---



## 📖 项目简介

**Code Your Own LLM** 是一个基于 [nanochat](https://github.com/karpathy/nanochat) 的深度扩展项目，旨在提供一份从原理到实践、从算法到工程的全栈式大语言模型参考指南。我们将提供最简洁的代码实现，没有复杂封装、层层继承和不必要的抽象，用极致的扁平化结构，零层级嵌套和极简的项目依赖来实现大语言模型从零训练到工程落地的整套流程。此外，项目补充了深入的算法原理讲解、详细的复现教程、前沿的优化技术分析，以及丰富的可视化交互。

我们希望从**原理解读、工程实现、实践指导、优化分析、可视化展示**五个维度，系统且详尽地诠释大语言模型算法及工程中从理论设计到代码实现的每一个细节。帮助学习者踏入大语言模型的美妙世界。通过本项目的学习，你将有能力定义独属于自己的大语言模型，能够理解每一行代码背后的数学原理与工程考量，能够根据实际需求调整模型架构与训练策略。



## 📚 结构预览

<div align="center">

| 章节                                                    | 摘要                         | 状态 |
| ------------------------------------------------------- | ---------------------------- | ---- |
| [第1章 引言](./docs/Chapter01/Introduction.md)          | 项目概述以及学习建议         | ✅    |
| [第2章 环境配置](./docs/Chapter02/Environment.md)       | 开发环境搭建与硬件要求解析   | ✅    |
| [第3章 数据](./docs/Chapter03/Data.md)                  | 数据采集、清洗及质量评估     | ✅    |
| [第4章 分词](./docs/Chapter04/Tokenization.md)          | 分词算法原理与词表构建实践   | 🚧    |
| [第5章 模型架构](./docs/Chapter05/Architecture.md)      | 从零实现模型核心架构         | ✅    |
| [第6章 预训练](./docs/Chapter06/Pretrain.md)            | 大规模预训练的目标与优化策略 | 🚧    |
| [第7章 中期训练](./docs/Chapter07/Midtrain.md)          | 领域适应与持续学习技术       | 🚧    |
| [第8章 有监督微调](./docs/Chapter08/SFT.md)             | 指令微调与参数高效微调方法   | ✅    |
| [第9章 强化学习](./docs/Chapter09/RL.md)                | 强化学习与模型偏好对齐       | ✅    |
| [第10章 模型推理](./docs/Chapter10/Inference.md)        | 高效推理与量化加速技术       | 🚧    |
| [第11章 模型评估](./docs/Chapter11/Evaluation.md)       | 评估指标体系与基准测试方法   | 🚧    |
| [第12章 模型安全与红队测试](./docs/Chapter12/Safety.md) | 安全对齐与红队测试实践       | 🚧    |
| [附录](./docs/Appendix/README.md)                       | 数学基础与前沿论文解读       | 🚧    |

</div>

## 🤝 项目成员

感谢以下项目的核心贡献者，排名不分先后



## 🙏致谢

本项目受益于 [nanochat](https://github.com/karpathy/nanochat), [nanoGPT](https://github.com/karpathy/nanoGPT), [transformers](https://github.com/huggingface/transformers), [pytorch](https://github.com/pytorch/pytorch), [llms-from-scratch](https://github.com/rasbt/LLMs-from-scratch) 感谢以上开源项目的出色工作！



## 📜 开源协议

项目采用[知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议](http://creativecommons.org/licenses/by-nc-sa/4.0/)进行许可。
