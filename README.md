# graphrag-practice-chinese

`graphrag-practice-chinese`是一个 GraphRAG 的应用实例，项目特点在于提供了替换 OpenAI 模型的方法，并通过修改原有提示和切分文档的方法，提高了 GraphRAG 处理中文内容的能力。

# 搭建环境(非常关键！)

```bash
git clone https://github.com/zhaoyingjun/graphrag-practice-chinese.git

cd ./graphrag-practice-chinese

# 安装项目运行所需的依赖
pip install -r ./requirements.txt

# 创建 input 目录，用于构建索引的文本文件默认存放于该目录下，可以按需修改 settings.yaml 文件中的 input 部分来指定路径
mkdir ./input

# 这一命令将在 graphrag-practice-chinese 目录中创建两个文件：.env 和 settings.yaml
python -m graphrag.index --init --root ./
```

# 修改配置文件

GraphRAG 主要的配置文件有两个：`.env` 和 `settings.yaml`：

- `.env` 包含运行 GraphRAG pipeline 所需的环境变量。该文件默认只定义了一个环境变量 `GRAPHRAG_API_KEY=<API_KEY>` 。

- `settings.yaml` 包含 pipeline 相关的设置。

**在项目根目录你可以找到作为参考的配置文件 [demo.env](./demo.env) 和 [settings.demo.yaml](./settings.demo.yaml)。**

你可以参考配置进行修改，也可以通过重命名覆盖初始化的配置文件。对于更多 settings.yaml 的配置选项，你可以参考官方文档：[Default Configuration Mode (using JSON/YAML)](https://microsoft.github.io/graphrag/config/json_yaml/)和[Fully Custom Config](https://microsoft.github.io/graphrag/config/custom/)

```
这里推荐使用大语言模型 glm-4-flash（首个免费调用的模型），因为在推理和总结阶段需要消耗大量的 Tokens。
我尝试对完整的《红楼梦》原文构建索引，最终消耗了大约 700W 个 Tokens，个人学习用的话尽力而为吧。
```

# 优化策略 — 使模型侧重中文

## 优化 1: 文本切分

官方分块把文档按照 token 数进行切分，对于中文来说容易在 chunk 之间出现乱码，这里参考 `Langchain-ChatChat` 开源项目，用中文字符数对文本进行切分。

复制文件 [splitter/tokens.py](./splitter/tokens.py) 替换掉 python 依赖库中的 `graphrag/index/verbs/text/chunk/strategies/tokens.py` 即可。

## 优化 2: 使用中文提示词(chinese-prompt)

初始化后，在 `prompts` 目录中可以看到 GraphRAG 的四个 prompt 文件的内容都由英文书写，并要求 LLM 使用英文输出。

为了更好地处理中文内容，这里我使用 `gpt-4o` 模型，将 [prompts/](./prompts/) 中的四个 prompt 文件都翻译成中文，并要求 LLM 用中文输出结果。

**如果你有更好的想法，想要自定义提示词，同样可以通过修改这四个 prompt 文件来实现，但注意不要修改提示词的文件名，以及不要修改和遗漏了在原提示词中有关输出的关键字段和格式，以免 GraphRAG 无法正常获取它们。**

## 优化 3: 模型调用

GraphRAG 默认使用 openai 进行模型调用，该模型为国外模型，对中文支持并不友好。为更好地支持中文，这里选择 `bigmodel` 进行模型调用，该模型为国内大模型厂商智谱 AI 提供。

## 优化 4: 模型选择

GraphRAG 默认使用 gpt-4o 模型，该模型为国外模型，对中文支持并不友好。为更好地支持中文，这里选择 `glm-4-plus` 模型，该模型为国内大模型厂商智谱 AI 提供。

# 构建索引

1. 通过运行如下命令， Graphrag 会在指定的文件路径下加载配置文件`.env`和`setting.yaml`，并按照你的配置开始构建索引。

```bash
python -m graphrag.index --root ./graphrag-practice-chinese
```

- 假设你当前的文件路径已经在`graphrag-practice-chinese`下的话，命令指定的构建路径应该为当前目录，则构建索引的命令应该是：

```bash
python -m graphrag.index --root ./
```

**你需要确保指定的文件路径下存在配置文件`.env`和`setting.yaml`，且配置了正确的`api_key`。**

**自定义样本数据**

GraphRAG 会默认为 `input` 路径下的 `txt` 文件构建索引，**如果需要指定文件的路径或类型，可以修改`settings.yaml`中的`input`部分**。

```
注意！GraphRAG 仅支持 `txt 或 csv` 类型的文件，编码格式必须为 `utf-8`。
```

在本项目中，我将红楼梦原文文本作为样本，所以在配置文件`setting.yaml`中将文件路径`base_dir`修改为`input/hongloumeng`，如下:

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

**如果你也想要把红楼梦原文文本作为样本，可以通过我的另一个项目 [hongloumeng-txt](https://github.com/Airmomo/hongloumeng-txt) 获取到符合 GraphRAG 格式要求的文件，获取完成后将文件放在`input/hongloumeng`目录下即可。**

2. 在构建过程中会自动创建两个目录：

- `output` 目录，用于存放查询结果。
- `cache` 目录，用于存放缓存数据。

3. 索引构建完成后会提示：`All workflows completed successfully` ，说明即可构建完成，随时可以进行查询。（如果没有 GPU 加持的话，构建的过程还是比较久的，可以在控制台你看到每一个步骤的进度条。）

# 查询测试

## global 全局查询

```bash
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

```bash
python -m graphrag.query --root ./graphrag-practice-chinese --method local "贾母对宝玉的态度怎么样？"
```

查询结果示例：

```markdown
SUCCESS: Local Search Response:
贾母对宝玉的态度可以从多个方面进行总结：

1. 溺爱与关心：贾母对宝玉有着深厚的溺爱。在《红楼梦》中，贾母多次探望宝玉，甚至亲自到园中看望他，表现出对宝玉的关心和爱护。例如，在贾母探视宝玉的情况中，贾母和王夫人一同探望宝玉，并询问他的病情，显示出贾母对宝玉的关心（Data: Entities (4704, 2929, 3895, 5470, 5868)）。

2. 宠爱与宽容：贾母对宝玉的宠爱还体现在对宝玉行为的宽容上。宝玉性格顽劣，有时甚至有些荒唐，但贾母却总是以宽容的态度对待他。例如，贾母对宝玉的干妈“老东西”的指责，显示出贾母对宝玉的宠爱（Data: Relationships (528, 2124)）。

3. 期望与教育：尽管贾母对宝玉宠爱有加，但她也关心宝玉的教育。在贾母房中，贾母关注宝玉的教育，并关心他的成长（Data: Entities (2702, 5524, 5868)）。

4. 情感交流：贾母与宝玉之间有着深厚的情感交流。在贾母与宝玉的互动中，贾母不仅关心宝玉的身体健康，还关心他的心理状态，体现出两人之间深厚的感情（Data: Sources (607, 314, 481)）。

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
