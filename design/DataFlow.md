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
