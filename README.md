### tf lite + posenet测试

1. 运行前请先安装以下包:
- numpy 
- pillow 
- opencv 
- matplotlib
- tensorflow 或者 [tflite runtime](https://www.tensorflow.org/lite/guide/python)
注意: 如果使用tflite runtime, 需运行较新的操作系统，如ubuntu 18.04等。实测ubuntu 16.04会有glibc版本过低的问题

2. 关于posenet:
官方tflite版本的[posenet](https://www.tensorflow.org/lite/models/pose_estimation/overview)未提供python代码。而tflite的网络模型输出的只是中间数据，需要额外的解码代码来生成最终姿态数据。所以本项目使用了[pysenet](https://github.com/augustye/pysenet), 一个从javascript移植到python的posenet版本. pysenet并不成熟，可能需要进一步代码优化。

