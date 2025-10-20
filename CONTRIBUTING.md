# 贡献指南

感谢您对 [Datawhale](https://github.com/datawhalechina) 和 [code -your-own-llm](https://github.com/datawhalechina/code-your-own-llm) 项目的关注！我们非常欢迎并由衷感谢任何形式的贡献。无论是改进文档、优化代码，添加测试用例或者实现新功能，您的贡献都将帮助这个项目变得更好。

## 📜 行为准则

参与本项目即表示您同意遵守我们的[**贡献者公约及行为准则**](CODE_OF_CONDUCT.md)。请务必阅读并遵守其中的内容，让我们共同维护一个友好、包容的社区环境。

## 💡 我可以做什么贡献

我们接受各种形式的贡献，包括但不限于：

### 📝 文档改进

- 修正文档中的错别字、语法错误、逻辑谬误
- 改进优化语句的表述和可读性
- 翻译文档、国际化推广
- 补充教程内容和示例

### 🖥️ 代码贡献 

- 修复已知的代码BUG
- 实现新功能
- 优化现有代码或注释
- 添加测试用例

### 🐛 问题反馈

- 报告BUG和错误
- 提出新功能建议
- 表达使用反馈

### 🎨 其他贡献

- 改进项目结构
- 优化可视化效果
- 分享学习笔记和心得

## 🚀 开始之前

1. **熟悉项目**：请先阅读 [README.md](README.md) 了解项目的基本信息和目标
2. **查看 Issues**：浏览现有的 [Issues](https://github.com/datawhalechina/code-your-own-llm/issues) 查看是否有您感兴趣的任务
3. **避免重复工作**：在开始工作之前，请先检查是否已有人在处理相同的问题或者在先前的 [PR](https://github.com/datawhalechina/code-your-own-llm/pulls) 中已经被解决
4. **沟通交流**：如果您计划进行较大的改动，建议先开一个 Issue 讨论；如果有一个存在较大潜在代码量的 [Pull Request](https://docs.github.com/pull-requests)，可以考虑将其拆解成一个 [Draft Pull Request](https://github.blog/news-insights/product-news/introducing-draft-pull-requests/) 和多个后续PR的形式，来让项目维护者更容易阅读并进行高效的 [code review](https://github.com/features/code-review)

### 👩‍💻👩‍💻 贡献流程和提交规范

我们采用标准的 [**Fork-and-Pull-Request**](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) 工作流程。以下是详细步骤：

1. 访问 [项目主页](https://github.com/datawhalechina/code-your-own-llm) ，然后点击 ["Fork"](https://github.com/datawhalechina/code-your-own-llm/fork) ，将项目的代码副本保存到您的 GitHub 用户账户下。

2. 将 fork 后的仓库克隆到本地：

   ```bash
   git clone https://github.com/$YOUR_USERNAME/code-your-own-llm.git
   cd code-your-own-llm
   ```

3. 新建虚拟环境并安装项目依赖

   ```bash
   conda create -n env python=3.10 uv
   # install uv (if not already installed)
   command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
   # create a .venv local virtual environment (if it doesn't exist)
   [ -d ".venv" ] || uv venv
   # install the repo dependencies
   uv sync
   # activate venv so that `python` uses the project's venv instead of system python
   source .venv/bin/activate
   ```

4. 进行修改

​	在您的分支上进行修改时请确保：

&emsp;&emsp;a. 代码风格符合项目规范

&emsp;&emsp;b. 添加必要的注释

&emsp;&emsp;c. 如果修改代码，确保代码可以正常运行

&emsp;&emsp;d. 如果修改文档，确保格式正确、表述清晰

5. 提交改动

   ```bash
   # 拉取并同步最新的代码，避免冲突
   git pull
   git add .
   git commit -m "type: description"
   ```

   关于提交信息中的**Type 类型**请参考如下标签：

   - `feat`: 新功能
   - `fix`: 修复 bug
   - `docs`: 文档修改更新
   - `style`: 代码风格修改（不影响代码运行）
   - `refactor`: 代码重构
   - `test`: 测试用例修改
   - `chore`: 构建过程或辅助工具的变动
   - `perf`: 性能优化

6. 推送改动

   ```bash
   git push -u origin main
   ```

## ❓ 获取帮助

如果您在参与贡献过程中遇到任何问题，可以通过以下方式获取帮助：

- 💬 在相关 Issue 或 PR 中评论提问
- 📧 联系项目负责人：[@Tsumugii24](https://github.com/Tsumugii24)
- 🐋 加入 Datawhale 社区交流群 🎉

