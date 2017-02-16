# Ataraxia Ava Platform
Ava平台是一个分布式机器学习云平台，由DataFlow, Learning Engine, API Server和Basic Firmware 四个模块组成。
* DataFlow - 数据ETL平台
* Learning Engine - 学习引擎，对接DataFlow的数据管道，通过学习得到模型，以及周边服务
* API Server - 推理平台，对接Learning Engine产生的算法模型，提供推理用的API
* Firmware - 底层基础架构，包括数据分布式平台，数据管道，学习分布式平台，Log服务等


## DataFlow
* 分布式数据ETL
* 数据文件接口
* 内置数据预处理函数模块
* 内置数据转换/放大函数模块
* batch/flow模式支持
* 图式编程接口和多语言APIs

## Learning Engine
* 多平台分布式机器学习
* 内置多种训练模式
* 学习过程可视化
* 算法评估子模块Libra
* 图式编程接口和多语言APIs

## API Server
* 模型自动部署API
* 高并发，高可用，高稳定

## Firmware
* 容器编排系统
* 数据存储系统
* 数据库支持
* 日志系统
* 用户管理系统
