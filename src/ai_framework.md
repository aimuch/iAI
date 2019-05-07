# AI Framework
- [AI 常用的结构框架](#ai框架数据结构)   
- [TensorFlow](ai/tensorflow.md#tensorflow)
  - [Tensorflow to TensorRT Image Classification](ai/tensorflow.md#tensorflow-to-tensorrt-image-classification)
  - [TensorFlow FAQ](ai/tensorflow.md#tensorflow-faq)
- [Pytorch](ai/pytorch.md#pytorch)
    - [将数据转换为Pytorch格式](ai/pytorch.md#将数据转换为pytorch格式)
- [Caffe](ai/caffe.md#caffe)

# AI框架数据结构
| Name | Tensor Format | Channel Order | Format |
| :----: | :------: | :------: | :------: |
| Caffe	| number,channel,height,width | BGR | BBBGGGRRR |
| TensorFlow | number,height,width,channel | RGB | RGBRGBRGB |
| Darknet | number,channel,height,width | RGB | RRRGGGBBB |
| OpenCV | eight,width,channel | BGR | BGRBGRBGR |