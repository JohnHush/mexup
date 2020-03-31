# 项目说明

```
pip install -r requirements.txt

python server.py
```

## 项目结构 

```
│  README.md                      # 项目说明
│  requirements.txt               # 所需依赖库
│  server.py                      # 运行入口
│  settings.py                    # 配置
│  urls.py                        # 项目路由设置
│
├─apps                            # app应用文件
│  ├─base                         # 处理基类
│  └─restful                      # 具体应用，resful为应用名
│         soccer_handler.py       # 处理逻辑，C控制层
│         urls.py                 # 路由映射
├─configs                         # 项目配置
│         log_config.py           # log配置
├─docs                            # 项目文档/说明
├─quantization                    # 模型层 负责算法模型
│         soccer_poisson.py       #足球模型
│         match_odds.py           #足球所有比赛赔率
│         infer_soccer_model_input.py   #足球反查模型
├─logs                            # 日志文件存放
├─media                           # 一些媒体资源
├─templates                       # 模板，放html页面
├─static                          # 静态文件，存放js 、css 、html、img
└─utils                           # 工具类，比如：验证码生成、IP地址转换
```