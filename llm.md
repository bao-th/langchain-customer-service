# LLM

## Agent

### 1. 结构化输出的四种策略 response_format

1. ProviderStrategy 使用模型提供商的“原生结构化输出功能”实现结构化输出
2. ToolStrategy
3. type / AutoStrategy
4. None

#### 1.1多种结构化输出方式：schema参数

1. Pydantic类型（Field）
2. TypedDict类型 (Annotated)
3. JsonSchema类型
4. @dataclass类型
5. 多schema联合模式 （Union[类型1, 类型2, ...]）

#### 1.2 tool_message_content 参数

#### 1.3 handle_errors 参数

1. True/False/固定字符串
2. 设置为指定异常类型
3. 设置为自定义错误处理函数

### 2.流式输出 agent.stream (stream_mode=指定模式)

1. values
2. updates (默认)
3. messages (打字机效果)
4. tasks
5. debug
6. checkpoints
7. custom

### 中间件 (Middleware)

常用的：

1. SummarizationMiddleware - 对历史消息列表进行摘要&总结，达到压缩上下文的效果
2. HumanInTheLoopMiddleware - (人在环中间件、人工审核中间件)
   在工具调用前中断Agent运行，等待用户对工具调用请求决策。可选的决策有：approve (同意执行)、edit (编辑调用配置后执行)、reject
   (拒绝执行)
3. PIIMiddleware - 用于检测和处理对话中的个人身份信息（Personally Identifiable Information, PII），支持自定义处理策略
4. TodoListMiddleware - 赋予了Agent任务规划和追踪进度的能力，可以应对复杂的多步任务

其它：

1. ModelCallLimitMiddleware - 用于限制模型调用次数，避免无限循环，控制调用成本
2. ToolCallLimitMiddleware - 用于限制工具调用次数，也可以限制特定工具的调用次数
3. ModelFallbackMiddleware - 用于在模型调用失败时，切换到备用模型或策略
4. LLMToolSelectorMiddleware - 当工具太多时，用子模型筛选最相关的几个工具
5. ToolRetryMiddleware - 基于指数退避算法，设置工具调用失败时的重试策略
6. ModelRetryMiddleware - 模型调用失败时重试，策略和工具调用的重试一样，都是基于指数退避算法
7. LLMToolEmulator - 某些情况下，工具尚未开发完成，我们希望先测试工具调用，可以用LLM tool emulator模拟工具
8. ContextEditingMiddleware - 上下文编辑中间件，该中间件提供了上下文管理的一种方式，通过更改发送给模型的消息列表来控制成本
9. FilesystemFileSearchMiddleware - 基于系统的Glob (根据文件路径检索)和Grep (根据文件内容检索)检索工具，为Agent赋予本地文件搜索和分析的能力。
10. Shell tool
11. Filesystem
12. Subagent

### checkpointer

短期记忆：

1. checkpointer = InMemorySaver ()
2. checkpointer = PostgresSaver.from_connection_string (DB_URL)
3. checkpointer = SqliteSaver.from_conn_string ()

长期记忆：

1. store = InMemoryStore ()
2. store = PostgresStore.from_connection_string (DB_URL)

## RAG

### 文档加载器 Document Loaders

1. TextLoader (str (path), encoding='utf-8')
2. CSVLoader (path)
3. JSONLoader (path, jq_schema='.data.items[]', text_content=False)
4. PyPDFLoader (path, extraction_mode='plain')
    - 提取模式:控制如何从 PDF 文件中解析和提取文本结构。
        - plain 提取文本,默认值
        - layout 布局感知提取模式,通常会通过插入大量的空格、换行符,来模拟原文档中的多栏、缩进和间距 (适用场景:学术论文 (如
          arXiv 论文)、多栏报刊杂志、带有左右分栏的合同)
5. UnstructuredWordDocumentLoader (path, mode='single')
    - 加载模式:
        - single 返回单个Document对象
        - elements 按标题等元素切分文档
6. UnstructuredMarkdownDocumentLoader (path, mode='single', strategy='fast')
    - 加载模式:
        - single 返回单个Document对象
        - elements 按标题等元素切分文档
    - 解析策略：
        - "fast"(快速模式)，它会以最快的速度提取文本，不进行复杂的版面分析
        - "hi_res"高分辨率模式
7. UnstructuredHTMLDocumentLoader (path, mode='elements', strategy='fast')
    - strategy:
        - “fast"解析加载html文件速度是比较快 (但可能丢失部分结构或元数据)
        - "hi_res":(高分辨率解析)解析精准 (速度慢一些)
        - "ocr_only" 强制使用ocr提取文本,仅仅适用于图像 (对HTML无效)

    - mode : one of {'paged', 'elements', 'single'}
        - 'elements"按语义元素 (标题、段落、列表、表格等)拆分成多个独立的小文档
8. 定义DirectoryLoader对象,指定要加载的文件夹路径、要加载的文件类型和是否使用多线程
   ```python
   directory_loader = DirectoryLoader(
       path="../asset/load",
       glob="*.py",              # 文件匹配模式（过滤器）。使用标准的Unix 路径通配符。
       use_multithreading=True,  # 是否启用多线程。填 True 意味着 LangChain 会同时并发读取多个文件。
       show_progress=True,       # 是否显示进度条。填 True 时，控制台在加载文件时会弹出一个进度条
       loader_cls=PythonLoader   # 指定底层核心加载器
   )

### 文档切分器 Text Splitters

#### Chunking拆分的策略

- 方法1:根据句子切分:这种方法按照自然句子边界进行切分,以保持语义完整性。

- 方法2:按照固定字符数来切分:这种策略根据特定的字符数量来划分文本,但可能
  会在不适当的位置切断句。

- 方法3:按固定字符数来切分,结合重叠窗口 (overlapping windows):此方法与按
  字符数切分相似,但通过重叠窗口技术避免切分关键内容,确保信息连贯性。

- 方法4:递归字符切分方法:通过递归字符方式动态确定切分点,这种方法可以根据文
  档的复杂性和内容密度来调整块的大小。

- 方法5:根据语义内容切分:这种高级策略依据文本的语义内容来划分块,旨在保持
  相关信息的集中和完整,适用于需要高度语义保持的应用场景。

#### TextSplitter的源码剖析

方法1:
split_text (self, text: str) -> list[str]:

- 传入的参数类型:文本内容 (或字符串),返回值类型:字符串列表
- 此方法是抽象方法,具体的实现细节由子类来决定

方法2:
create_documents (self, texts: list[str], ... ) -> list[Document]:

- 传入的参数类型:字符串列表,返回值类型:Document对象列表
- 此方法的底层调用了split_text (),即将参数中的每一个字符串都传入split_text ()中执行,得到的字
  符串列表中,将字符串封装为Document对象,就构成了list[Document]。

方法3:
split_documents (self, documents: Iterable[Document]) -> list[Document]:

- 传入的参数类型:Document对象列表,返回值类型:Document对象列表
- 此方法的底层调用了create_documents (),将参数中的每一个Document对象,提取其page_content字段,则构成了字符串列表,然后调用方法2即可。

#### 具体文档切割器

1. CharacterTextSplitter

```python
splitter = CharacterTextSplitter(
    chunk_size=50,  # 每块大小
    chunk_overlap=5,  # 块与块之间的重复字符数
    # Length_function=len,
    # 设置为空字符串时,表示禁用分隔符优先
    separator=""
)
```

2. RecursiveCharacterTextSplitter: 最常用

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,  # 增加重叠字符
    separators=["\n\n", "\n", "。", "！", "？", "......", "，", ""],  # 添加中文标点
    length_function=len,
    keep_separator=True  # 保留句尾标点(如 ......),避免切割后丢失语气和逻辑
```

3. TokenTextSplitter
4. SemanticChunker：语义分块
5. HTMLHeaderTextSplitter
6. CodeTextSplitter
7. MarkdownTextSplitter

### 文档嵌入模型 Text Embedding Models

1. 句子的向量化 embedding.embed_query (text:str)
2. 文档的向量化 embedding.embed_documents (texts:[str])

### 向量存储 Vector Stores

![img.png](img.png

1. FAISS：一个用于高效相似性搜索和密集向量聚类的库。Meta出品,开源、免费,Facebook Al相似性搜索服务。 (Facebook Al Similarity
   Search,Facebook Al相似性搜索库) /faes/
2. Chroma：开源、免费的轻量级向量数据库,有极简的API
3. Milvus：开源的专为向量搜索设计的云原生数据库。性能强悍,功能丰富。覆盖轻量级的原型开发到十亿级向量的大规模生产系统
4. Pgvector：开源关系型数据库PostgreSQL的扩展,为PostgreSQL增加了向量数据类型和相似性搜索功能
5. Redis：开源内存数据结构存储,现已原生支持向量相似性搜索功能
6. Elasticsearch：开源分布式搜索和分析引擎,提供了一个基于文档的数据库,结构化、非结构化和向量数据通过高效的列式存储统一管理
7. Pinecone：具有广泛功能的向量数据库

![alt text](image.png)