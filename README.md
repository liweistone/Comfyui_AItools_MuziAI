# 插件文档优化版

## 插件概述
本插件集是为 ComfyUI 设计的多功能工具集，包含以下功能：
- 图片处理
- 视频创作与编辑
- AI生成
- 抖音下载

主要服务于内容创作者和AI艺术工作者，专注于：
- 批量生成
- 创建流程化无人值守的工作流
- 商业化变现支持

## 安装说明
1. 手动安装：直接下载插件，解压后将插件文件夹放入ComfyUI的`custom_nodes`目录
2. git安装：在custom_nodes目录下使用命令安装：git clone https://github.com/liweistone/Comfyui_AItools_MuziAI.git
3. 安装依赖：`pip install -r requirements.txt`原则重启comfyui会自动安装依赖文件。

### 特别注意事项：
- Gemini相关节点需要提前准备API密钥（建议存入系统环境变量）
- 抖音下载器需要配置有效的Cookie
- 首次使用图像调节器功能会自动下载模型文件

> 如需API密钥配置帮助，请关注公众号"懂AI的木子"查阅相关教程

## 技术支持
- 联系微信：stone_liwei（加好友请注明来源）
- 更多工具：公众号"懂AI的木子"
- 反馈渠道：[插件帮助页面](https://mp.weixin.qq.com/s/6XGfN18VmhnMU4Xb7KPTcQ)

## 节点分类

### 🎨 公众号懂AI的木子做号工具

#### 批量图片处理
- 😋 批量图片重命名和格式转换 (BatchImageRenamer)
- 🖼️ 懒人做号IMG加载生成器 (ImageLoaderFromFolder)

#### 提示词相关
- 📗 提示词大师 (RandomPromptLoader)

#### 视频处理
- 📺 从数据中随机加载视频 (RandomVideoLoadertwo)
- 🎞️ 图片序列转视频 (FramesToVideoNode)
- 🖼️ 输出视频第一帧 (VideoFirstFrameNode)
- ✂️ 视频裁剪工具 (VideoTrimNode)
- 📺 批次视频加载器 (VideoLoader)
- 🎞️ 视频转图片序列 (VideoToFramesNode)

#### DeepSee剧本生成
- 📖 DeepSeek脚本生成 (DeepSeekScriptNode)
- 🎨 分镜提示词生成 (StoryboardPromptNode)
- ⚙️ DeepSeek配置 (DeepSeekConfigNode)

#### Gemini相关
- 🎬 视频提示生成器 (GeminiImageAnalyzer)
- 🖼️ 图片提示生成器 (GeminiPromptGenerator)
- ✨ 提示词优化器 (GeminiPromptOptimizer)
- 💬 聊天器 (GeminiChatPro)
- 🎨 图像编辑 (GeminiProEditimagemuziai)
- 💬 Gemini聊天 (GeminiProChatmuziai)

#### 抖音下载
- ⬇️ 抖音作品下载器 (DouyinDownloadNode)

#### 人物调节
- 💓 胸部大小调节器 (BreastSizeAdjuster)
- 💓 NSFW版胸部调节 (BreastSizeAdjusternswf)
- ✋ 手部稳定调节器 (HandStabilityAdjuster)
- 🔥 性感风格调节器 (SexyStyleAdjuster)
- 😍 网感调节器 (Influencer_regulator)
- 🎨 滤镜风格调节器 (ESSImageApplyLUT)

## 节点详细说明

### 批量图片处理节点
#### 😋 批量图片重命名和格式转换 (BatchImageRenamer)
**功能**：批量重命名和转换图片格式  
**输入参数**：
- 源文件夹路径
- 目标文件夹路径
- 文件名模板（支持{index}占位符）
- 起始索引
- 输出格式（jpg/png/webp）
- 是否覆盖已存在文件

**输出**：处理状态信息

#### 🖼️ 懒人做号IMG加载生成器 (ImageLoaderFromFolder)
**功能**：从目录加载图片（支持顺序/随机读取）  
**输入参数**：
- 模式（sequential/random）
- 随机种子值（seed）
- 重置计数器（顺序模式可选）

**输出**：
- 图像张量
- 图像路径
- 当前序号

（后续节点说明保持相同优化格式...）

## 使用提示
1. 人物调节器建议从小强度开始尝试
2. 视频处理需要ffmpeg支持
3. 抖音下载器需配置有效Cookie
4. Gemini节点需配置API密钥
