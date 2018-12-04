# Ubuntu 系统环境设置问题

- [Ubuntu 系统环境设置问题](#ubuntu-系统环境设置问题)
  - [安装python依赖库](#安装python依赖库)
  - [安装chrome浏览器](#安装chrome浏览器)
  - [pip和pip3安装报错](#pip和pip3安装报错)
  - [ubuntu 16下安装spyder3](#ubuntu-16下安装spyder3)
  - [安装搜狗输入法](#安装搜狗输入法)
  - [WPS无法输入中文](#wps无法输入中文)
<<<<<<< HEAD
  - [zsh+oh-my-zsh默认shell的最佳替代品](#zsh+oh-my-zsh默认shell的最佳替代品)
=======
  - [安装赛睿霜冻之蓝v2驱动](#安装赛睿霜冻之蓝v2驱动)
>>>>>>> 284db4dbc003eec9e03a652aabcf14d2736fd641

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
## 安装chrome浏览器
***[参考地址](https://blog.csdn.net/qq_30164225/article/details/54632634)***

将下载源加入系统源列表:    
```shell
sudo wget http://www.linuxidc.com/files/repo/google-chrome.list -P /etc/apt/sources.list.d/
```
导入谷歌软件公钥:   
```shell
wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
```
更新源:   
```shell
sudo apt-get update
```
安装chrome
```shell
sudo apt-get install google-chrome-stable
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

---
## 安装搜狗输入法

1. [下载linux版搜狗输入法](https://pinyin.sogou.com/linux/?r=pinyin)    
2. 命令行运行：    
   ```shell
    sudo dpkg -i sogoupinyin_2.2.0.0108_amd64.deb
   ```
3. System Setting -> Language Support -> Keyboard input method system:`fcitx`    
4. 状态栏->输入法->打开Fcitx配置窗口，点击`+`去掉`Only Show Current Language`前面对号，然后搜`sogou`添加好，重启电脑即可。    
5. 有可能重启后会出现两个输入法图标，解决方法：    
   ```shell
   sudo apt-get remove fcitx-ui-qimpanel
   ```
---
## WPS无法输入中文    
**问题**：Ubuntu16.04自带的libre对于office的格式兼容性太差，只好安装了WPS。但是WPS文字、表格、演示均不能输入中文。   
**原因**：环境变量未正确设置。 
**解决办法**:    
#### WPS文字
打开终端输入：
```shell
sudo vim /usr/bin/wps
```    
添加一下文字到打开的文本中（添加到“#!/bin/bash”下面）：   
```shell
export XMODIFIERS="@im=fcitx"
export QT_IM_MODULE="fcitx"    
```    
#### WPS表格
打开终端输入：
```shell
sudo vim /usr/bin/et
```    
添加一下文字到打开的文本中（添加到“#!/bin/bash”下面）：
```shell
export XMODIFIERS="@im=fcitx"
export QT_IM_MODULE="fcitx"
```
####  WPS演示
打开终端输入：
```shell
sudo vim /usr/bin/wpp
``` 
添加一下文字到打开的文本中（添加到“#!/bin/bash”下面）：
```shell
export XMODIFIERS="@im=fcitx"
export QT_IM_MODULE="fcitx"
```
修改完后保存，打开相应的程序切换输入法就可以输入中文了。

---
<<<<<<< HEAD
## zsh+oh-my-zsh默认shell的最佳替代品   
### 项目地址   
**zsh** -----> http://www.zsh.org   
**oh-my-zsh** ----> http://ohmyz.sh   

### 安装zsh   
```shell   
sudo apt-get install zsh
```
### 安装oh-my-zsh   
**via curl**   
```shell
sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
```
**via wget**   
```shell
sh -c "$(wget https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh -O -)"
```

### 设置zsh为系统默认shell   
**为root用户修改默认shell为zsh**   
```shell
chsh -s /bin/zsh root
```
**为当前用户修改默认shell为zsh**   
```shell
chsh -s /bin/zsh
# or
chsh -s `which zsh`
```
**恢复命令**   
```shell
chsh -s /bin/bash
```

**add to ~/.zshrc**   
```shell
export PATH=$PATH:/usr/local/go/bin
#export PATH=$PATH:/Applications/MAMP/bin/php/php5.6.10/bin:/Users/GZM/composer:/Users/GZM/.composer/vendor/bin
#export GOPATH=/Users/GZM/work/go
#export GOPATH=/Volumes/Transcend/git/360/private_cloud_server_code/tools/gowork/
#export GOBIN=$GOPATH/bin
#export GO15VENDOREXPERIMENT=1
LC_CTYPE=en_US.UTF-8
LC_ALL=en_US.UTF-8
```
**插件**   

[Plugins](https://github.com/robbyrussell/oh-my-zsh/wiki/Plugins)   

**升级**   
upgrade_oh_my_zsh
=======
## 安装赛睿霜冻之蓝v2驱动
先安装依赖项：   
```shell
sudo apt-get install build-essential python-dev libusb-1.0-0-dev libudev-dev
```
接着安装驱动：   
```shell
sudo pip install rivalcfg
```
>>>>>>> 284db4dbc003eec9e03a652aabcf14d2736fd641
