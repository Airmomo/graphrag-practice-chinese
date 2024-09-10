# graphrag-practice-chinese

`graphrag-practice-chinese`是一个 GraphRAG 的应用实例，项目特点在于提供了替换 OpenAI 模型的方法，并通过修改原有提示和切分文档的方法，提高了 GraphRAG 处理中文内容的能力。

# 搭建环境

```bash
mkdir ./graphrag-practice-chinese

# 创建 input 目录，用于构建索引的文本文件默认存放于该目录下，可以按需修改 settings.yaml 文件中的 input 部分来指定路径
mkdir ./graphrag-practice-chinese/input

git clone https://github.com/zhaoyingjun/graphrag-practice-chinese.git ./graphrag-practice-chinese

# 这一命令将在 ./graphrag-practice-chinese 目录中创建两个文件：.env 和 settings.yaml（这两个文件涉及密钥，所以没有上传，需自行初始化后按需修改。）
python -m graphrag.index --init --root ./graphrag-practice-chinese
```

`.env` 包含运行 GraphRAG pipeline 所需的环境变量。该文件定义了一个环境变量 `GRAPHRAG_API_KEY=<API_KEY>` 。
这是 OpenAI API 或 Azure OpenAI endpoint 的 API 密钥。我想要替换掉 OpenAI 模型，所以只需要修改`setting.yaml`，可以不用对该环境变量进行更改。

`settings.yaml` 包含 pipeline 相关的设置。我选择更改了的 LLM 和 Embedding 部分，使用的是：

- 智谱 AI 的 `glm-4-plus` 和 `embedding-3`。
  - 大语言模型：`glm-4-plus`
  - **这里推荐使用大语言模型`glm-4-flash`（首个免费调用的模型）**，因为在推理和总结阶段需要消耗大量的 Tokens。我尝试对完整的《红楼梦》原文构建索引，最终消耗了大约 700W 个 Tokens，个人学习用的话尽力而为吧。
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
python -m graphrag.index --root ./graphrag-practice-chinese
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
python -m graphrag.query --root ./graphrag-practice-chinese --method global "故事的主旨是什么？"
```

查询结果示例：

```markdown
SUCCESS: Global Search Response:
《红楼梦》的主旨在于通过对贾、王、史、薛四大家族的兴衰描写，展现了封建社会的各种矛盾和冲突，揭示了封建社会的腐朽和衰落。故事中的人物关系错综复杂，反映了当时社会的风俗习惯和道德观念。小说通过对宝玉、黛玉、宝钗等主要人物的爱情悲剧，探讨了人性、命运、社会关系等主题，反映了作者对封建礼教和封建制度的批判。

此外，小说还探讨了人生、命运、爱情、婚姻等主题，反映了作者对人生和社会的深刻思考。通过对贾宝玉、林黛玉、薛宝钗等主要人物的塑造，展现了封建社会中人性的复杂性和悲剧性，反映了人性的光辉与阴暗面。故事中的人物命运和家族兴衰反映了当时社会的现实，同时也表达了作者对美好人性的追求和对理想社会的向往。

综上所述，《红楼梦》的主旨不仅揭示了封建社会的腐朽和衰落，还探讨了人性、命运、社会关系等主题，具有深刻的思想内涵和艺术价值。
```

## local 本地查询

```python
python -m graphrag.query --root ./graphrag-practice-chinese --method local "贾母对宝玉的态度怎么样？"
```

查询结果示例：

```markdown
SUCCESS: Local Search Response:
贾母对宝玉的态度可以从多个方面进行总结：

1. **溺爱与关心**：贾母对宝玉有着深厚的溺爱。在《红楼梦》中，贾母多次探望宝玉，甚至亲自到园中看望他，表现出对宝玉的关心和爱护。例如，在贾母探视宝玉的情况中，贾母和王夫人一同探望宝玉，并询问他的病情，显示出贾母对宝玉的关心（Data: Entities (4704, 2929, 3895, 5470, 5868)）。

2. **宠爱与宽容**：贾母对宝玉的宠爱还体现在对宝玉行为的宽容上。宝玉性格顽劣，有时甚至有些荒唐，但贾母却总是以宽容的态度对待他。例如，贾母对宝玉的干妈“老东西”的指责，显示出贾母对宝玉的宠爱（Data: Relationships (528, 2124)）。

3. **期望与教育**：尽管贾母对宝玉宠爱有加，但她也关心宝玉的教育。在贾母房中，贾母关注宝玉的教育，并关心他的成长（Data: Entities (2702, 5524, 5868)）。

4. **情感交流**：贾母与宝玉之间有着深厚的情感交流。在贾母与宝玉的互动中，贾母不仅关心宝玉的身体健康，还关心他的心理状态，体现出两人之间深厚的感情（Data: Sources (607, 314, 481)）。

综上所述，贾母对宝玉的态度是溺爱、关心、宠爱、宽容，同时也有期望和教育。这种复杂的情感关系，体现了贾母对宝玉的深厚感情。
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
