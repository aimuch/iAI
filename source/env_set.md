# Ubuntu 系统环境设置问题

- [Ubuntu 系统环境设置问题](#ubuntu-系统环境设置问题)
  - [安装python依赖库](#安装python依赖库)
  - [pip和pip3安装报错](#pip和pip3安装报错)
  - [ubuntu 16下安装spyder3](#ubuntu-16下安装spyder3)

---
## 安装python依赖库
`注意：Python2 的话用pip安装，Python3用pip3安装（总之要知道安装在哪里，有的系统将python软连接到Python3上了）`
```shell
pip install scipy numpy scikit-image scikit-learn jupyter notebook matplotlib pandas
```
**DGX-ONE**服务器下安装：
```shell
apt-get install scipy
apt-get install numpy
apt-get install python-skimage(install skimage)
(pspnet): install matio
```




---
## pip和pip3安装报错
**问题描述**
```shell
$ pip3 --version
Traceback (most recent call last):
  File "/usr/bin/pip3", line 5, in <module>
    from pkg_resources import load_entry_point
  File "/usr/lib/python3/dist-packages/pkg_resources.py", line 2708, in <module>
    working_set.require(__requires__)
  File "/usr/lib/python3/dist-packages/pkg_resources.py", line 686, in require
    needed = self.resolve(parse_requirements(requirements))
  File "/usr/lib/python3/dist-packages/pkg_resources.py", line 584, in resolve
    raise DistributionNotFound(req)
pkg_resources.DistributionNotFound: pip==1.5.6
```
**解决方法**
```shell
sudo python3 get-pip.py
sudo python3 ez_setup.py
```
其中[get-pip.py](./source/fix_pip/get-pip.py)和[ez_setup.py](./source/fix_pip/ez_setup-pip.py)文件在`source/fix_pip/`文件夹中。

---
## ubuntu 16下安装spyder3
安装pip3：
```shell
sudo apt install python3-pip
```
安装 Spyder 最新版本，目前即为 Spyder 3：
```shell
pip3 install -U spyder
```
命令行下运行即可：
```shell
spyder3
```
若运行时发现报错：`qtpy.PythonQtError: No Qt bindings could be found`
那就安装 pyqt5（目前最新为5）……：$ 
```shell
pip3 install -U pyqt5
```