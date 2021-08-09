## 学习后端
后端项目以 PyTorch 为学习框架，利用 ResNet101 网络进行学习。  
### 项目目录结构
```
.  
├─ app.py # 学习系统入口文件  
├─ container.py # 服务容器类文件  
├─ debug.py # DeBug工具文件  
├─ handle.py # 处理代码类文件  
├─ helper.py # 全局辅助函数类文件
├─ requirements.txt # PyThon环境要求  
├─ server.py # 前后端连接类文件  
├─ visualize.py # 可视化工具  
│  
├─config # 配置文件目录  
│  ├─ data.py # 数据源配置  
│  ├─ logs.py # 日志配置  
│  ├─ training.py # 训练配置  
│  └─ __init__.py  
│  
├─Modules # 核心类目录  
│  ├─ Image.py # 图片封装类  
│  ├─ Network.py # 网络模型封装类  
│  └─ __init__.py  
│  
├─public # 前端编译代码  
│  ├─ favicon.ico # 网站图标  
│  ├─ index.html # 网站入口文件  
│  ├─ nginx.example # 网站配置样例文件  
│  │  
│  ├─css # [已编译的]样式表文件  
│  │  └─ *.css  
│  │  
│  ├─img # [已编译的]图标文件  
│  │  └─ *.svg  
│  │  
│  └─js # [已编译的]JavaScript文件  
│     ├─ *.js  
│     └─ *.js.map  
│  
├─Services # 服务类目录  
│  ├─ services.py # 服务提供者注册文件  
│  └─ __init__.py  
│  
├─Source # 数据源目录  
│  ├─ sources.py # 数据源文件  
│  └─ __init__.py  
│  
├─storage # 存储目录  
│  ├─bin # 训练中储存的网络模型  
│  ├─cache # 训练中使用的临时文件  
│  ├─images # 压缩文件解压至此  
│  ├─logs # 训练中输出的日志  
│  │  ├─plots # 预测结果图像  
│  │  ├─print # 标准输出  
│  │  ├─summary # TensorBoard输出  
│  │  └─tex # [未完成]训练报告
│  │    └─ sample.tex
│  │          
│  ├─saves # 保存的数据源加载目录  
│  └─zips # 压缩文件数据源目录  
│  
├─tests # 测试数据目录  
└─Tools # 静态工具类目录  
   ├─ tools.py  
   └─ __init__.py  
```
### 项目说明  
本项目主要可配置部分有：
* 数据源  
配置数据源时，请修改 `config/data.py` 文件中的定义，该文件中使用的数据源均来自 `sources/sources.py` 文件，按照文件中类初始化所需的参数进行配置即可。其中 `FilesSource` 类会将图片数据解压至 `storage/images` 目录中，除 `SavedSource` 类之外的类在加载完毕后会将数据保存至 `storage/cache` 目录中，可直接移动至 `storage/saves` 目录中进行使用。
* 网络结构与训练参数  
网络结构利用预训练网络时不在 `config/training.py` 中指明，需要在 `services/services.py` 中的 `NetworkServiceProvider` 的方法中指明其加载方式。使用自定义网络时，将服务提供者的代码改为自定义封装类后，在 `config/training.py` 中定义即可，详细可见本项目[GitHub页面](https://github.com/XuZhixuan/micro_recognizer)稍早的提交。训练参数则可直接修改配置文件。

使用学习框架进行学习时：  
* 首先配置项目环境，使用 Anaconda 建立虚拟环境 `conda create -n project_name`  
* 启用该虚拟环境 `conda activate project_name`  
* 安装依赖 `conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c conda-forge`、 `pip install -r requirements.txt`  
* 启动框架 `python app.py`

### 启用网页计算服务
本项目计算服务后端与学习后端使用相同的服务容器，启用该模块需要将网络结构定义配置为训练好的模型文件，后端 PyTorch 框架部分配置与前述相同。此外需要配置 HTTP 反向代理服务器，建议使用 nginx 作为代理服务器， nginx 配置文件样例见 `public/nginx.example` ，配置其他反向代理服务器时注意启用 WebSocket 连接和静态资源映射即可。

* 启动服务后端 `python server.py`
