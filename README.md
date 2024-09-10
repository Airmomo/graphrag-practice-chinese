# graphrag-practice

# 搭建环境

```bash
mkdir ./graphrag-practice

# 创建 input 目录，用于构建索引的文本文件默认存放于该目录下，可以按需修改 settings.yaml 文件中的 input 部分来指定路径
mkdir ./graphrag-practice/input

git clone https://github.com/zhaoyingjun/graphrag-practice.git ./graphrag-practice

# 这一命令将在 ./ragtest 目录中创建两个文件：.env 和 settings.yaml（这两个文件涉及密钥，所以没有上传，需自行初始化后按需修改。）
python -m graphrag.index --init --root ./graphrag-practice
```

`.env` 包含运行 GraphRAG pipeline 所需的环境变量。该文件定义了一个环境变量 `GRAPHRAG_API_KEY=<API_KEY>` 。
这是 OpenAI API 或 Azure OpenAI endpoint 的 API 密钥。我想要替换掉 OpenAI 模型，所以只需要修改`setting.yaml`，可以不用对该环境变量进行更改。

`settings.yaml` 包含 pipeline 相关的设置。我选择更改了的 LLM 和 Embedding 部分，使用的是：

- 智谱 AI 的 `glm-4-plus` 和 `embedding-3`。
  - 大语言模型：`glm-4-plus`
  - 嵌入模型：`embedding-3`

# 优化策略 — 使模型侧重中文

## 优化 1: 文本切分

官方分块把文档按照 token 数进行切分，对于中文来说容易在 chunk 之间出现乱码，这里参考 Langchain-ChatChat 开源项目，用中文字符数对文本进行切分。

方法 1:

- 用本项目下的 `splitter/chinese_text_splitter.py` 替换掉 python 依赖库中的 `graphrag/index/verbs/text/chunk/strategies/tokens.py` 即可。

方法 2:
要使用 `chinese_text_splitter.py` 中的 `ChineseTextSplitter` 作为文档分割方法，您需要在 `settings.yaml` 文件中添加一个新的 `splitter` 部分。以下是修改后的相关部分：

```yaml
# ... 其他设置保持不变 ...
chunks:
  size: 1200
  overlap: 100
  group_by_columns: [id]

splitter:
  type: custom
  module: splitter.chinese_text_splitter
  class: ChineseTextSplitter
  params:
    pdf: false
    sentence_size: 250
# ... 其他设置保持不变 ...
```

这里的修改说明：

1. 保留原有的 `chunks` 部分，因为它可能仍被用于其他目的。
2. 添加新的 `splitter` 部分：
   - type: `custom` 表示我们使用自定义的分割器。
   - module: `splitter.chinese_text_splitter` 指定了包含 `ChineseTextSplitter` 类的 Python 模块路径。请确保这个路径正确反映了 `chinese_text_splitter.py` 文件在您项目中的位置。
   - class: `ChineseTextSplitter` 指定了要使用的类名。
   - params 部分包含了传递给 `ChineseTextSplitter` 的参数。这里设置了 `pdf: false` 和 `sentence_size: 250`，您可以根据需要调整这些值。

请注意：

1. 确保 `chinese_text_splitter.py` 文件位于正确的位置，使得 GraphRAG 能够找到并导入它。
2. 修改完成后，保存 `settings.yaml` 文件，然后**重新运行 GraphRAG 的索引构建命令**。这应该会使用 `ChineseTextSplitter` 来分割您的文档。

## 优化 2: prompt

在 `prompts` 中可以看到 `GraphRAG` 的四个 prompt 文件的内容都由英文书写，并要求 LLM 使用英文输出。为了更好地处理中文内容，这里使用 `gpt-4o` 模型，将 `prompts` 中的四个 prompt 文件都翻译成中文，并要求 LLM 用中文输出结果。

## 优化 3: 模型调用

`GraphRAG` 默认使用 `openai` 进行模型调用，该模型为国外模型，对中文支持并不友好。为更好地支持中文，这里选择 `bigmodel` 进行模型调用，该模型为国内大模型厂商智谱 AI 提供。

## 优化 4: 模型选择

`GraphRAG` 默认使用 `gpt-4o` 模型，该模型为国外模型，对中文支持并不友好。为更好地支持中文，这里选择 `glm-4-plus` 模型，该模型为国内大模型厂商智谱 AI 提供。

# 索引构建

```bash
python -m graphrag.index --root ./graphrag-practice
```

GraphRAG 会默认为 `input` 路径下的 `txt` 文件构建索引，如果需要指定文本文件的路径或类型，可以修改`settings.yaml`中的`input`部分。

- 注意 GraphRAG 仅支持 `txt 或 csv` 类型的文件，编码格式必须为 `utf-8`。

在本项目中，我将红楼梦原文文本作为样本，在此处将文件路径修改为`input/hongloumeng`，如下:

如果你也想要把红楼梦原文文本作为样本，可以通过我的另一个项目 [hongloumeng-txt](https://github.com/Airmomo/hongloumeng-txt) 获取到符合 GraphRAG 格式要求的文件，获取完成后将文件放在`input/hongloumeng`目录下即可。

```yaml
# ... 其他设置保持不变 ...
input:
  type: file # or blob
  file_type: text # or csv
  base_dir: "input/hongloumeng"
  file_encoding: utf-8
  file_pattern: ".*\\.txt$"
# ... 其他设置保持不变 ...
```

构建过程中会自动创建

- output 目录，用于存放查询结果。
- cache 目录，用于存放缓存数据。

索引构建完成后会提示：`All workflows completed successfully` ，说明即可进行查询。

# 查询测试

## global 全局查询

```python
python -m graphrag.query --root ./ragtest --method global "故事的主旨是什么？"
```

## local 本地查询

```python
python -m graphrag.query --root ./ragtest --method local "贾母对宝玉的态度怎么样？"
```

## Tip：全局查询和本地查询的区别

| 特征         | 本地查询 (Local Search)    | 全局查询 (Global Search)         |
| ------------ | -------------------------- | -------------------------------- |
| 查询范围     | 以特定实体为入口点         | 基于预先计算的实体社区摘要       |
| **查询方法** | **使用实体嵌入和图遍历**   | **向每个社区提问并汇总答案**     |
| **适用场景** | **针对特定实体的精确查询** | **广泛的主题性问题**             |
| 性能         | 对简单直接任务更高效       | 适合处理复杂的多步骤查询         |
| 复杂度       | 相对较低                   | 较高，需要更多计算资源           |
| 响应速度     | 通常更快                   | 可能较慢，取决于查询复杂度       |
| 洞察深度     | 适中                       | 更深入，能更全面理解上下文和关系 |
| Token 使用量 | 较低                       | 较高，due to 多次 LLM 调用       |
| 实现依赖     | 向量搜索和图遍历           | 预计算的社区摘要和多次 LLM 调用  |
| 最佳使用场景 | 需要快速直接答案的情况     | 需要深入洞察和复杂推理的场景     |
