# Caffe

[Caffe训练模型可视化](#caffe训练模型可视化)


## caffe模型训练可视化
### 记录训练日志
训练阶段需要加上`-log_dir ./log/`, 其中`./log/`为log文件存放文件文件夹：    
```sh
~/caffe/build/tools/caffe train --solver=~/caffe/examples/mydata/slot_classifier/solver.prototxt -log_dir ./log/
```

### 解析训练日志
将`caffe/tools/extra`文件夹下的`extract_seconds.py`, `parse_log.py`, `parse_log.sh`, `plot_training_log.py.example`拷贝到上述的`./log/`文件夹下.


#### 分步法
1. ~~修改日志文件名删除`caffe.hostname.username.log`之后的`.INFO.XXXX`，保存为`caffe.hostname.username.log`文件~~ 创建软连接`ln -s caffe.hostname.username.log`(`hostname`和`username`具体根据个人电脑, 下面依然)；    
2. 执行: `/parse_log.sh caffe.hostname.username.log` , 这样就会在当前文件夹下生成一个`.train`文件和一个`.test`文件;    
3. 执行:    
   ```shell
   ./plot_training_log.py.example 0  save.png caffe.hostname.username.log
   ```
   就可以生成训练过程中的`Test accuracy  vs. Iters` 曲线,其中`0`代表曲线类型， `save.png` 代表保存的图片名称, caffe中支持很多种曲线绘制，通过指定不同的类型参数即可，具体参数如下:    
    ```vim
    Notes:
       1. Supporting multiple logs.
       2. Log file name must end with the lower-cased ".log".
    Supported chart types:
        0: Test accuracy  vs. Iters
        1: Test accuracy  vs. Seconds
        2: Test loss  vs. Iters
        3: Test loss  vs. Seconds
        4: Train learning rate  vs. Iters
        5: Train learning rate  vs. Seconds
        6: Train loss  vs. Iters
        7: Train loss  vs. Seconds
    ```

#### 一步法

1. 创建软连接 `ln -s caffe.hostname.username.log` ( `hostname` 和 `username` 具体根据个人电脑, 下面依然)；
2. 运行:   
   ```shell
    ./plot_training_log.py  [数字选项] 图片名.png ./caffe.log
   ```
   其中**数字选项如下:**   
   ```vim
    Notes:
       1. Supporting multiple logs.
       2. Log file name must end with the lower-cased ".log".
    Supported chart types:
        0: Test accuracy  vs. Iters
        1: Test accuracy  vs. Seconds
        2: Test loss  vs. Iters
        3: Test loss  vs. Seconds
        4: Train learning rate  vs. Iters
        5: Train learning rate  vs. Seconds
        6: Train loss  vs. Iters
        7: Train loss  vs. Seconds
   ```