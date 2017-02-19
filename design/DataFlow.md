# DataFlow
提供数据ETL平台和管道，实现海量数据的ETL。

## Features
* Distributed, High Speed, High Reliability
* Batch / Flow Mode Support
* Uniform Json Data Format
* Rich Data Augment Methods
* Rich Data Prepropessing Methods
* Graph Programming Interface and Multi-Language APIs

## Data Format

* 用于分类的图像Label:

      {
        "source": "/disk2/image1.jpg",
        "type": "image", //image, video, text ...
        "label":{
          "classification": 1
        }
      }

* 用于分类和检测的图像Label:

      {
        "source": "https://ata.ai/img010.jpg",
        "type": "image", //image, video, text ...
        "label":{
          "classification": 3,
          "detection":{
            "Rois":[
              {
                "cls": 1,  //检测标签
                "roi": [231, 222, 85, 65]  //x,y,w,h
              },
              {
                "cls": 2,
                "roi": [38, 85, 58, 44]
              }
            ]
          }
        }
      }


## 工程进度
### Stage 1: MVP 2017.02.20 - 2017.02.26
#### 分布式处理机制 - 在Kirk上基于X-Spark，实现以下功能
* 任务分割
* 并行处理
* 任务完成度管理
* Failure Restart

#### 工作类型 - 各工作以API的形式提供，分布式平台只需要调用对应的API接口
* 下载 / 上传
* 图像预处理 - 基于Keras，实现Mean-Std，模糊、旋转、镜像、缩放、平移、裁剪等功能
* Q-Pipeline：从日志流抓数据，整理用于重弄新训练的过程
  * 日志采集 - 采集日志，并将日志保存到存储。这一步在qlogview机器上完成
  * Fetch - 读取日志记录，将图片fetch到存储，将url和label保存为[标准格式的json-log](# Data Format)
