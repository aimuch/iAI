# Ubuntu FAQ

[Ubuntu FAQ](#ubuntu-faq)
  - [Linuxbrew安装](#linuxbrew安装)
    - [安装linuxbrew](#安装linuxbrew)
    - [linuxbrew必装包](#linuxbrew必装包)
    - [brew常用命令](#brew常用命令)
    - [linuxbrew注意事项](#linuxbrew注意事项)
  - [Ubuntu每次开机后提示检测到系统程序出现问题的解决方法](#ubuntu每次开机后提示检测到系统程序出现问题的解决方法)
  - [Ubuntu循环登陆问题](#ubuntu循环登陆问题)
  - [安装python依赖库](#安装python依赖库)
  - [安装chrome浏览器](#安装chrome浏览器)
  - [pip和pip3安装报错](#pip和pip3安装报错)
  - [ubuntu 16下安装spyder3](#ubuntu-16下安装spyder3)
  - [安装搜狗输入法](#安装搜狗输入法)
  - [WPS无法输入中文](#wps无法输入中文)
  - [安装赛睿霜冻之蓝v2驱动](#安装赛睿霜冻之蓝v2驱动)
  - [zsh oh-my-zsh默认shell的最佳替代品](#zsh-oh-my-zsh默认shell的最佳替代品)
    - [查看系统shell环境](#查看系统shell环境)
    - [安装zsh](#安装zsh)
    - [安装vimrc](#安装vimrc)
    - [安装oh-my-zsh](#安装oh-my-zsh)
    - [安装zsh-syntax-highlighting](#安装zsh-syntax-highlighting)
  - [vim配置](#vim配置)
    - [YouCompleteMe实现vim自动补全](#youcompleteme实现vim自动补全)
    - [vim最终配置](#vim最终配置)
  - [Sublime Text 3配置问题](#sublime-text-3配置问题)
  - [Visual Studio Code配置问题](#visual-studio-code配置问题)   
  - [Ubuntu查看和关闭进程](#ubuntu查看和关闭进程)   
  - [Ubuntu后台执行命令](#ubuntu后台执行命令)   
  - [查看系统状态](#查看系统状态)
---

## Linuxbrew安装
[*The Homebrew package manager for Linux*](https://linuxbrew.sh/)    

### 安装linuxbrew    

将以下命令粘贴到命令行中运行:    
```bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/Linuxbrew/install/master/install.sh)"
```
The installation script installs Linuxbrew to `/home/linuxbrew/.linuxbrew` using sudo if possible and in your home directory at` ~/.linuxbrew` otherwise. Linuxbrew does not use sudo after installation. Using `/home/linuxbrew/.linuxbrew` allows the use of more binary packages (bottles) than installing in your personal home directory.   

Follow the Next steps instructions to add Linuxbrew to your `PATH` and to your bash shell profile script, either `~/.profile` on Debian/Ubuntu or `~/.bash_profile` on CentOS/Fedora/RedHat.    
```bash
test -d ~/.linuxbrew && eval $(~/.linuxbrew/bin/brew shellenv)
test -d /home/linuxbrew/.linuxbrew && eval $(/home/linuxbrew/.linuxbrew/bin/brew shellenv)
test -r ~/.profile && echo "eval \$($(brew --prefix)/bin/brew shellenv)" >>~/.profile
echo "eval \$($(brew --prefix)/bin/brew shellenv)" >>~/.profile
source ~/.profile
```

You’re done! Try installing a package:
```bash
brew install hello
```
If you’re using an older distribution of Linux, installing your first package will also install a recent version of glibc and gcc. Use brew doctor to troubleshoot common issues.

### linuxbrew必装包   
- git
- wget
- vim

### brew常用命令    
- `brew shellenv`    
  Prints export statements - run them in a shell and this installation of Homebrew will be included into your PATH, MANPATH and INFOPATH.

  HOMEBREW_PREFIX, HOMEBREW_CELLAR and HOMEBREW_REPOSITORY are also exported to save multiple queries of those variables.

  Consider adding evaluating the output in your dotfiles (e.g. ~/.profile) with eval $(brew shellenv)

  ```
  brew shellenv
  ```
  ```bash
  export HOMEBREW_PREFIX="/home/linuxbrew/.linuxbrew"
  export HOMEBREW_CELLAR="/home/linuxbrew/.linuxbrew/Cellar"
  export HOMEBREW_REPOSITORY="/home/linuxbrew/.linuxbrew/Homebrew"
  export PATH="/home/linuxbrew/.linuxbrew/bin:/home/linuxbrew/.linuxbrew/sbin:$PATH"
  export MANPATH="/home/linuxbrew/.linuxbrew/share/man:$MANPATH"
  export INFOPATH="/home/linuxbrew/.linuxbrew/share/info:$INFOPATH"
  ```
- `brew install xxx`    
  安装xxx软件    
- `brew uninstall xxx`    
  卸载xxx软件    
- `brew search xxx`    
  搜索xxx软件     

### linuxbrew注意事项
假如用`linuxbrew`安装的`Python`会替换系统默认的`Python`。若需要还原则需要将`~/.profile`文件中的`eval $(/home/linuxbrew/.linuxbrew/bin/brew shellenv)`这一行屏蔽:    
```shell
# linuxbrew
#eval $(/home/linuxbrew/.linuxbrew/bin/brew shellenv) 
```    
然后重启电脑.   


---
## Ubuntu每次开机后提示检测到系统程序出现问题的解决方法
首先，错误报告存放位置:   
```bash
cd /var/crash/
ls #可以查看错误报告
sudo rm /var/crash/* #删除该目录下的所有文件
```
但是，这只是删除掉的是错误报告，如果系统再有什么崩溃，又会再报错。   

---
## Ubuntu循环登陆问题
### 问题描述
登录Ubuntu的时候在输入密码和登录桌面之间反复循环。   

### 原因
安装软件的时候破坏了NVIDIA驱动导致。   

### 解决方法
1. 进入linux的shell    
在登录界面进入linux的shell（`ctrl + Alt + F1`），输入用户名、密码，进入shell。   
关闭图形界面，命令为:    
```bash
sudo service lightdm stop #或者 sudo /etc/init.d/lightdm stop
#sudo apt-get autoremove #有可能需要
```
2. 卸载NVIDIA驱动:    
```bash
sudo apt-get purge nvidia*
```
或者:   
```bash
sudo PATH_TO_NVIDIA_DRIVE/NVIDIA-Linux-x86_64-xxx.run --uninstall
```
3. 重新安装NVIDIA驱动:    
```bash
sudo apt-get install nvidia-390 nvidia-settings nvidia-prime #nvidia-390可以替换为其他版本
```
或者:   
```bash
sudo PATH_TO_NVIDIA_DRIVER/NVIDIA-Linux-x86_64-xxx.run --no-opengl-files
```
**.run**安装过程选项为:    
```vim
在NVIDIA驱动安装过程中，依次的选项为：
1 accept
2 The distribution-provided pre-install script failed … …
Continue installation
3 Would you like to run the nvidia-xconfig utility to automatically update your X Configuration file so set the NVIDIA X driver will be used when you restart X?
NO
4 Install 32-Bit compatibility libraries?
NO
```
4. 打开图形界面，命令为:   
```bash
sudo service lightdm start #或者sudo /etc/init.d/lightdm start
```

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
![BundleInstall](../img/sougou.png)     
5. 有可能重启后会出现两个输入法图标，解决方法：    
   ```shell
   sudo apt-get remove fcitx-ui-qimpanel
   ```

6. 搜狗拼音输入法候选栏乱码解决方法    
   ```shell
   cd ~/.config
   sudo rm -rf SogouPY* sogou*
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
## 安装赛睿霜冻之蓝v2驱动
先安装依赖项：   
```shell
sudo apt-get install build-essential python-dev libusb-1.0-0-dev libudev-dev
```
接着安装驱动：   
```shell
sudo pip install rivalcfg
```

---
## zsh oh-my-zsh默认shell的最佳替代品
### 项目地址   
- **[zsh](http://www.zsh.org)**     
- **[vimrc](https://github.com/amix/vimrc)**
- **[oh-my-zsh](http://ohmyz.sh)**      
- **[zsh-syntax-highlighting](https://github.com/zsh-users/zsh-syntax-highlighting)**    


### 查看系统shell环境     
查看当前发行版可以使用的`shell`:    
```shell
cat /etc/shells 
```
查看当前使用的`shell`:    
```shell
echo $0
```
查看当前用户(默认)使用的`shell`:    
```shell
echo $SHELL
```

### 安装zsh   
```shell   
sudo apt-get install zsh
```
#### 设置zsh为系统默认shell:   
- **为root用户修改默认shell为zsh**   
```shell
chsh -s /bin/zsh root
```
- **为当前用户修改默认shell为zsh**   
```shell
chsh -s /bin/zsh
# or
chsh -s `which zsh`
```
#### 恢复bash为系统默认：   
```shell
chsh -s /bin/bash
```



### 安装vimrc    
- **Install for your own user only**    
The awesome version includes a lot of great plugins, configurations and color schemes that make Vim a lot better. To install it simply do following from your terminal:
```shell
git clone --depth=1 https://github.com/amix/vimrc.git ~/.vim_runtime 

sh ~/.vim_runtime/install_awesome_vimrc.sh
```
- **Install for multiple users**    
To install for multiple users, the repository needs to be cloned to a location accessible for all the intended users.    
```shell
git clone --depth=1 https://github.com/amix/vimrc.git /opt/vim_runtime
sh ~/.vim_runtime/install_awesome_parameterized.sh /opt/vim_runtime user0 user1 user2
# to install for all users with home directories
sh ~/.vim_runtime/install_awesome_parameterized.sh /opt/vim_runtime --all
```
Naturally, /opt/vim_runtime can be any directory, as long as all the users specified have read access.   

**错误处理**    
在运行vim的时候提示如下错误：    
```vim
vim-go requires Vim 7.4.2009 or Neovim 0.3.1, but you're using an older version.
Please update your Vim for the best vim-go experience.
If you really want to continue you can set this to make the error go away:
    let g:go_version_warning = 0
Note that some features may error out or behave incorrectly.
Please do not report bugs unless you're using Vim 7.4.2009 or newer or Neovim 0.3.1.
```
***解决方法***：    
编辑`.vimrc`    
```shell
vim ~/.vimrc
```
添加一行    
```vim
let g:go_version_warning = 0
```
保存退出 问题解决   
        
    
    
### 安装oh-my-zsh   
- **via curl**   
```shell
sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
```
- **via wget**   
```shell
sh -c "$(wget https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh -O -)"
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
```shell
upgrade_oh_my_zsh
```

**使用oh-my-zsh后导致的卡顿问题**    
现象是每次cd和ll时都会被卡住很长时间根本受不了，最后在官方github查明原因是使用的主题会自动获取git信息，可以使用以下命令禁止zsh自动获取git信息，解决卡顿问题:    
```shell
git config --global oh-my-zsh.hide-status 1
```
若想恢复则：    
```shell
git config --global oh-my-zsh.hide-status 0
```



### 安装zsh-syntax-highlighting    
- **Oh-my-zsh**    
1. Clone this repository in **oh-my-zsh**'s plugins directory:
  ```shell
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
  ```
2. Activate the plugin in `~/.zshrc`:    
  ```shell
    plugins=(
      git
      zsh-syntax-highlighting
    )
  ```
3. Source `~/.zshrc` to take changes into account:   
  ```shell
    source ~/.zshrc
  ```

- **In your ~/.zshrc**    

1. Simply clone this repository and source the script:
```shell
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git
echo "source ${(q-)PWD}/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh" >> ${ZDOTDIR:-$HOME}/.zshrc
```
2. Then, enable syntax highlighting in the current interactive shell:    
```shell
source ./zsh-syntax-highlighting/zsh-syntax-highlighting.zsh
```
If git is not installed, download and extract a snapshot of the latest development tree from:   
```shell
https://github.com/zsh-users/zsh-syntax-highlighting/archive/master.tar.gz
```
Note the `source` command must be **at the end** of `~/.zshrc`.


---
## vim配置
### YouCompleteMe实现vim自动补全    

1 准备条件 

(1) 最新版的`Vim(7.3.584+)`，须支持`python`。    
终端输入命令：`vim –version` 或 打开vim用命令：version 查看版本信息，若python前有‘+’即可。    
然后终端执行命令，安装相关依赖项：   
```shell
sudo apt-get install python-dev
```
装的过程中若遇到问题，依次执行以下命令：    
```shell
sudo apt-get update
sudo apt-get install -f
```
之后重试安装：    
```shell
sudo apt-get install python-dev
```

(2) 安装`cmake`    
```shell
sudo apt-get install cmake
```

(3) 安装`clang`   
```
sudo apt-get install clang
```
**或者跳过这步**，后面编译**YCM**(YouCompleteMe)时，如果没有`clang`会自动安装。 

(4)安装`Vundle`   
这个是用来管理`vim`插件的，安装和卸载都特别方便，各个插件是一个文件夹，放在目录`bunble`下。 


2 安装**vundle**   
(1) 下载`vundle`源码到本地    
```shell
git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim
```

(2) 在 `.vimrc` 的文件起始处，插入以下内容并保存：   
```vim
" >>>>>> vundle
set nocompatible              " 去除VI一致性,必须
filetype off                  " 必须

" 设置包括vundle和初始化相关的runtime path
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
" 另一种选择, 指定一个vundle安装插件的路径
"call vundle#begin('~/some/path/here')

" 让vundle管理插件版本,必须
Plugin 'VundleVim/Vundle.vim'

" 以下范例用来支持不同格式的插件安装.
" 请将安装插件的命令放在vundle#begin和vundle#end之间.
" Github上的插件
" 格式为 Plugin '用户名/插件仓库名'
Plugin 'tpope/vim-fugitive'
" 来自 http://vim-scripts.org/vim/scripts.html 的插件
" Plugin '插件名称' 实际上是 Plugin 'vim-scripts/插件仓库名' 只是此处的用户名可以省略
Plugin 'L9'
" 由Git支持但不再github上的插件仓库 Plugin 'git clone 后面的地址'
Plugin 'git://git.wincent.com/command-t.git'
" 本地的Git仓库(例如自己的插件) Plugin 'file:///+本地插件仓库绝对路径'
"Plugin 'file:///home/andy/path/to/plugin'
" 插件在仓库的子目录中.
" 正确指定路径用以设置runtimepath. 以下范例插件在sparkup/vim目录下
Plugin 'rstacruz/sparkup', {'rtp': 'vim/'}
" 安装L9，如果已经安装过这个插件，可利用以下格式避免命名冲突
"Plugin 'ascenator/L9', {'name': 'newL9'}
" YouCompleteMe
Plugin 'Valloric/YouCompleteMe'
Plugin 'yggdroot/indentline'
Plugin 'jiangmiao/auto-pairs'

" 你的所有插件需要在下面这行之前
call vundle#end()            " 必须
filetype plugin indent on    " 必须 加载vim自带和插件相应的语法和文件类型相关脚本
" 忽视插件改变缩进,可以使用以下替代:
"filetype plugin on
"
" 简要帮助文档
" :PluginList       - 列出所有已配置的插件
" :PluginInstall    - 安装插件,追加 `!` 用以更新或使用 :PluginUpdate
" :PluginSearch foo - 搜索 foo ; 追加 `!` 清除本地缓存
" :PluginClean      - 清除未使用插件,需要确认; 追加 `!` 自动批准移除未使用插件
"
" 查阅 :h vundle 获取更多细节和wiki以及FAQ
" 将你自己对非插件片段放在这行之后
" <<<<<< vundle
```
**注意**：`Bundle ‘插件名或git链接’ `表示要安装的插件     

(3)再次打开vim，在命令行模式中执行：
```vim
BundleInstall
```
![BundleInstall](../img/vim1.png)   
进入安装插件过程：    
![vim插件安装过程](../img/vim2.png)   

Plugin前面有`‘>’`表示该插件正在安装，`YouCompleteMe`插件安装的时间比较长，耐心等待，不要退出，最后会提示有一个错误，这是正常的，因为`YouCompleteMe`需要手工编译出库文件，就像上图中的‘！’，忽略它。    
**注**：若要卸载插件，只需将`.vimrc`中 “Bundle ‘插件’ ”这条语句删掉，然后在vim 命令行模式中执行：`BundleClean`即可。    

3  编译`YouCompleteMe`     

(1) 进入YouCompleteMe文件夹下    
```shell
cd  ~/.vim/bundle/YouCompleteMe/
```
![YouCompleteMe文件夹内容](../img/vim3.png)   

(2) 编译
```shell
./install.py  --clang-completer --go-completer --ts-completer
```
参数`–clang-completer`是为了支持C/C++的补全，不需要可以不加。编译过程比较长，耐心等待。    


4  修改`.vimrc`配置文件
(1) 找到配置文件`.ycm_extra_conf.py`在~/.vim/bundle/YouCompleteMe/third_party/ycmd/下面:    
```shell
cd ~/.vim/bundle/YouCompleteMe/thrid_party/ycmd/
```
`ls -a` 即可看到。    

(2) 自行在`YouCompleteMe/`中创建`cpp/ycm`目录，将 `.ycm_extra_conf.py`拷贝进去:    
```shell
cd ~/.vim/bundle/YouCompleteMe
mkdir cpp
mkdir cpp/ycm
cp ~/.vim/bundle/YouCompleteMe/thrid_party/ycmd/.ycm_extra_conf.py ~/.vim/bundle/YouCompleteMe/cpp/ycm/
```

(3) 修改`.vimrc`配置文件
将下面的内容添加到`.vimrc`里面:    
```vim
" >>>>>> YouCompleteMe
" 寻找全局配置文件
let g:ycm_global_ycm_extra_conf = '~/.vim/bundle/YouCompleteMe/thrid_party/ycmd/.ycm_extra_conf.py'
"let g:ycm_global_ycm_extra_conf = '~/.vim/bundle/YouCompleteMe/cpp/ycm/.ycm_extra_conf.py'
" 禁用syntastic来对python检查
let g:syntastic_ignore_files=[".*\.py$"] 
" 使用ctags生成的tags文件
let g:ycm_collect_identifiers_from_tag_files = 1
" 开启语义补全
" 修改对C语言的补全快捷键，默认是CTRL+space，修改为ALT+;未测出效果
"let g:ycm_key_invoke_completion = '<M-;>'
" 设置转到定义处的快捷键为ALT+G，未测出效果
"nmap <M-g> :YcmCompleter GoToDefinitionElseDeclaration <C-R>=expand("<cword>")<CR><CR> 
"关键字补全
"let g:ycm_seed_identifiers_with_syntax = 1
" 在接受补全后不分裂出一个窗口显示接受的项
set completeopt-=preview
" 让补全行为与一般的IDE一致
set completeopt=longest,menu
" 不显示开启vim时检查ycm_extra_conf文件的信息
let g:ycm_confirm_extra_conf=0
" 每次重新生成匹配项，禁止缓存匹配项
let g:ycm_cache_omnifunc=0
" 在注释中也可以补全
let g:ycm_complete_in_comments=1
" 输入第一个字符就开始补全
let g:ycm_min_num_of_chars_for_completion=1
" 错误标识符
let g:ycm_error_symbol='>>'
" 警告标识符
let g:ycm_warning_symbol='>*'
" 不查询ultisnips提供的代码模板补全，如果需要，设置成1即可
" let g:ycm_use_ultisnips_completer=0
" <<<<<< YouCompleteMe
```
上面的内容中，除了第一句寻找全局配置文件，其他的语句可以根据自己的需要更改、删除或添加。 
**注**：如果没有在第(3)步中自己创建`cpp/ycm`目录拷贝`.ycm_extra_conf.py`文件，则需要将第一句中的路径改为全局配置文件所在的具体路径，如下：   
```vim
let g:ycm_global_ycm_extra_conf = '~/.vim/bundle/YouCompleteMe/thrid_party/ycmd/.ycm_extra_conf.py'
```

5 保存退出`.vimrc` ,打开一个C/C++源程序，体验其自动补全效果。    
![vim提示](../img/vim4.png)   

6 配合上面安装的`syntastic`还可以语法检测     
![vim语法检测](../img/vim5.png)   

`‘>>’`指出有语法错误，但是检测速度太慢，没什么大用。自我感觉这个语法自动检测很烦，可以禁用它：    
进入 `/bundle/YouCompleteMe/plugin`，修改`youcompleteme.vim`中的：    
![syntastic](../img/vim6.png)   
将如上图中的`第141行`的参数改为`0`就可以了。    

7 `YcmDiags`插件可以显示错误或警告信息，可以设置`F9`为打开窗口的快捷键，在`.vimrc`中添加语句：   
![YcmDiags](../img/vim7.png)   
显示效果：   
![YcmDiags效果](../img/vim8.png)   

8 添加头文件     
目前在`include`中，无法补全`stdio.h`等头文件，我们需要将`/usr/include`添加进去。路径添加到 `~/.vim/bundle/YouCompleteMe/cpp/ycm/.ycm_extra_conf.py` 或者`~/.vim/bundle/YouCompleteMe/thrid_party/ycmd/.ycm_extra_conf.py`文件中的`flags` 数组中，每增加一个路径，前面要写`’-isystem’`。     
![添加头文件](../img/vim9.png)   
以后需要boost库等其他的补全，也需要将相应的路径添加进去。

# vim最终配置
```vim
" >>>>>>>> Vundle
set nocompatible              " 去除VI一致性,必须
filetype off                  " 必须

" 设置包括vundle和初始化相关的runtime path
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
" 另一种选择, 指定一个vundle安装插件的路径
"call vundle#begin('~/some/path/here')

" 让vundle管理插件版本,必须
Plugin 'VundleVim/Vundle.vim'

" 以下范例用来支持不同格式的插件安装.
" 请将安装插件的命令放在vundle#begin和vundle#end之间.
" Github上的插件
" 格式为 Plugin '用户名/插件仓库名'
Plugin 'tpope/vim-fugitive'
" 来自 http://vim-scripts.org/vim/scripts.html 的插件
" Plugin '插件名称' 实际上是 Plugin 'vim-scripts/插件仓库名' 只是此处的用户名可以省略
Plugin 'L9'
" 由Git支持但不再github上的插件仓库 Plugin 'git clone 后面的地址'
Plugin 'git://git.wincent.com/command-t.git'
" 本地的Git仓库(例如自己的插件) Plugin 'file:///+本地插件仓库绝对路径'
"Plugin 'file:///home/andy/path/to/plugin'
" 插件在仓库的子目录中.
" 正确指定路径用以设置runtimepath. 以下范例插件在sparkup/vim目录下
Plugin 'rstacruz/sparkup', {'rtp': 'vim/'}
" 安装L9，如果已经安装过这个插件，可利用以下格式避免命名冲突
"Plugin 'ascenator/L9', {'name': 'newL9'}
" YouCompleteMe
Plugin 'Valloric/YouCompleteMe'
Plugin 'yggdroot/indentline'
Plugin 'jiangmiao/auto-pairs'

" 你的所有插件需要在下面这行之前
call vundle#end()            " 必须
filetype plugin indent on    " 必须 加载vim自带和插件相应的语法和文件类型相关脚本
" 忽视插件改变缩进,可以使用以下替代:
"filetype plugin on
"
" 简要帮助文档
" :PluginList       - 列出所有已配置的插件
" :PluginInstall    - 安装插件,追加 `!` 用以更新或使用 :PluginUpdate
" :PluginSearch foo - 搜索 foo ; 追加 `!` 清除本地缓存
" :PluginClean      - 清除未使用插件,需要确认; 追加 `!` 自动批准移除未使用插件
"
" 查阅 :h vundle 获取更多细节和wiki以及FAQ
" 将你自己对非插件片段放在这行之后
" <<<<<<<< Vundle

" >>>>>>>> Vimrc
set runtimepath+=~/.vim_runtime

source ~/.vim_runtime/vimrcs/basic.vim
source ~/.vim_runtime/vimrcs/filetypes.vim
source ~/.vim_runtime/vimrcs/plugins_config.vim
source ~/.vim_runtime/vimrcs/extended.vim

try
source ~/.vim_runtime/my_configs.vim
catch
endtry

let g:go_version_warning = 0
" <<<<<<<< Vimrc


" >>>>>> YouCompleteMe
" 寻找全局配置文件
let g:ycm_global_ycm_extra_conf = '~/.vim/bundle/YouCompleteMe/thrid_party/ycmd/.ycm_extra_conf.py'
" 禁用syntastic来对python检查
let g:syntastic_ignore_files=[".*\.py$"] 
" 使用ctags生成的tags文件
let g:ycm_collect_identifiers_from_tag_files = 1
" 开启语义补全
" 修改对C语言的补全快捷键，默认是CTRL+space，修改为ALT+;未测出效果
"let g:ycm_key_invoke_completion = '<M-;>'
" 设置转到定义处的快捷键为ALT+G，未测出效果
"nmap <M-g> :YcmCompleter GoToDefinitionElseDeclaration <C-R>=expand("<cword>")<CR><CR> 
"关键字补全
"let g:ycm_seed_identifiers_with_syntax = 1
" 在接受补全后不分裂出一个窗口显示接受的项
set completeopt-=preview
" 让补全行为与一般的IDE一致
set completeopt=longest,menu
" 不显示开启vim时检查ycm_extra_conf文件的信息
let g:ycm_confirm_extra_conf=0
" 每次重新生成匹配项，禁止缓存匹配项
let g:ycm_cache_omnifunc=0
" 在注释中也可以补全
let g:ycm_complete_in_comments=1
" 输入第一个字符就开始补全
let g:ycm_min_num_of_chars_for_completion=1
" 错误标识符
let g:ycm_error_symbol='>>'
" 警告标识符
let g:ycm_warning_symbol='>*'
" 不查询ultisnips提供的代码模板补全，如果需要，设置成1即可
" let g:ycm_use_ultisnips_completer=0
" <<<<<< YouCompleteMe

" >>>>>> add by ANDY
" 显示行号
set number
" 显示标尺
set ruler
" 历史纪录
set history=1000
" 输入的命令显示出来，看的清楚些
set showcmd
" 状态行显示的内容
set statusline=%F%m%r%h%w\ [FORMAT=%{&ff}]\ [TYPE=%Y]\ [POS=%l,%v][%p%%]\ %{strftime(\"%d/%m/%y\ -\ %H:%M\")}
" 启动显示状态行1，总是显示状态行2
set laststatus=2
" 语法高亮显示
syntax on
set fileencodings=utf-8,gb2312,gbk,cp936,latin-1
set fileencoding=utf-8
set termencoding=utf-8
set fileformat=unix
set encoding=utf-8
" 配色方案
colorscheme desert
" 指定配色方案是256色
set t_Co=256
set wildmenu
" 去掉有关vi一致性模式，避免以前版本的一些bug和局限，解决backspace不能使用的问题
"set nocompatible
set backspace=indent,eol,start
set backspace=2
" 启用自动对齐功能，把上一行的对齐格式应用到下一行
set autoindent
" 依据上面的格式，智能的选择对齐方式，对于类似C语言编写很有用处
set smartindent
" vim禁用自动备份
set nobackup
set nowritebackup
set noswapfile
" 用空格代替tab
set expandtab
" 设置显示制表符的空格字符个数,改进tab缩进值，默认为8，现改为4
set tabstop=4
" 统一缩进为4，方便在开启了et后使用退格(backspace)键，每次退格将删除X个空格
set softtabstop=4
" 设定自动缩进为4个字符，程序中自动缩进所使用的空白长度
set shiftwidth=4
" 设置帮助文件为中文(需要安装vimcdoc文档)
set helplang=cn
" 显示匹配的括号
set showmatch
" 文件缩进及tab个数
au FileType html,python,vim,javascript setl shiftwidth=4
au FileType html,python,vim,javascript setl tabstop=4
au FileType java,php setl shiftwidth=4
au FileType java,php setl tabstop=4
" 高亮搜索的字符串
set hlsearch
" 检测文件的类型
"filetype off
filetype plugin on
filetype indent on
" C风格缩进
set cindent
set completeopt=longest,menu
" 功能设置
" 去掉输入错误提示声音
set noeb
" 自动保存
set autowrite
" 突出显示当前行
set cursorline
" 突出显示当前列
set cursorcolumn
"设置光标样式为竖线vertical bar
" Change cursor shape between insert and normal mode in iTerm2.app
"if $TERM_PROGRAM =~ "iTerm"
let &t_SI = "\<Esc>]50;CursorShape=1\x7" " Vertical bar in insert mode
let &t_EI = "\<Esc>]50;CursorShape=0\x7" " Block in normal mode
"endif
" 共享剪贴板
set clipboard+=unnamed
" 文件被改动时自动载入
set autoread
" 顶部底部保持3行距离
set scrolloff=3
" <<<<<< add by ANDY
```

---
## Sublime Text 3配置问题
**安装Control Package**:   
通过按下`Ctrl+'` 然后输入以下命令：   
```
import urllib.request,os,hashlib; h = '6f4c264a24d933ce70df5dedcf1dcaee'
+ 'ebe013ee18cced0ef93d5f746d80ef60'; pf = 'Package
Control.sublime-package'; ipp = sublime.installed_packages_path();
urllib.request.install_opener( urllib.request.build_opener(
urllib.request.ProxyHandler()) ); by = urllib.request.urlopen
('http://packagecontrol.io/' + pf.replace(' ', '%20')).read(); dh =
hashlib.sha256(by).hexdigest(); print('Error validating download (got %s
instead of %s), please try manual install' % (dh, h)) if dh != h else
open(os.path.join( ipp, pf), 'wb' ).write(by)
```


**环境配置**:   
```json
{
    "font_face": "Monaco",
    "font_size": 14,
    "translate_tabs_to_spaces": true
}
```

---
## Visual Studio Code配置问题
**推荐插件**:   
- Anaconda Extension Pack
- Chinese (Simplified) Language Pack for Visual Studio Code
- Markdown All in One
- Markdown Preview Github Styling
- Markdown Shortcuts
- prototxt
- vscode-cudacpp   

**环境配置**：   
```json
{
    "python.pythonPath": "/home/andy/anaconda3/envs/tensorflow/bin/python",
    "window.zoomLevel": 0,
    "ai.homepage.openOnStartup": false,
    "vsicons.presets.foldersAllDefaultIcon": true,
    "git.autofetch": true,
    "git.enableSmartCommit": true,
    "editor.fontFamily": "'Monaco', 'Dank Mono', 'Fira Code'",
    "editor.fontLigatures": true,
    "editor.fontSize": 15,
}
```
其中`"editor.fontFamily"`优先使用第一个字体，当第一个字体没有的时候依次使用后面的。


---
# ubuntu查看和关闭进程
根据进程名查看进程PID：   
```shell
ps -aux | grep [Process name]
```
关闭进程：   
```shell
kill -9 [Process_PID]
```

---
# Ubuntu后台执行命令   

*当我们在终端或控制台工作时，可能不希望由于运行一个作业而占住了屏幕，因为可能还有更重要的事情要做，比如阅读电子邮件。对于密集访问磁盘的进程，我们更希望它能够在每天的非负荷高峰时间段运行(例如凌晨)。为了使这些进程能够在后台运行，也就是说不在终端屏幕上运行，有几种选择方法可供使用。*   

## 1、&
当在前台运行某个作业时，终端被该作业占据,可以在命令后面加上`&`实现后台运行。例如：`sh test.sh &`   
适合在后台运行的命令有**find**、费时的排序及一些**shell**脚本。在后台运行作业时要当心：需要用户交互的命令不要放在后台执行，因为这样你的机器就会在那里傻等。不过，作业在后台运行一样会将结果输出到屏幕上，干扰你的工作。如果放在后台运行的作业会产生大量的输出，最好使用下面的方法把它的输出重定向到某个文件中：
```shell
command > out.file 2>&1 &
```
这样，所有的标准输出和错误输出都将被重定向到一个叫做`out.file`的文件中。PS：当你成功地提交进程以后，就会显示出一个进程号，可以用它来监控该进程，或杀死它。(`ps -ef | grep 进程号 `或者 `kill -9 进程号`）   

## 2、nohup
使用&命令后，作业被提交到后台运行，当前控制台没有被占用，但是一但把当前控制台关掉(退出帐户时)，作业就会停止运行。nohup命令可以在你退出帐户之后继续运行相应的进程。nohup就是不挂起的意思( no hang up)。该命令的一般形式为：   
```shell
nohup command &
```   
如果使用nohup命令提交作业，那么在缺省情况下该作业的所有输出都被重定向到一个名为nohup.out的文件中，除非另外指定了输出文件：   
```shell
nohup command > myout.file 2>&1 &
```   
使用了nohup之后，很多人就这样不管了，其实这样有可能在当前账户非正常退出或者结束的时候，命令还是自己结束了。所以在使用nohup命令后台运行命令之后，需要使用exit正常退出当前账户，这样才能保证命令一直在后台运行。   
```shell
ctrl + z #可以将一个正在前台执行的命令放到后台，并且处于暂停状态。
```
```shell
Ctrl+c #终止前台命令。
```
```shell
jobs #查看当前有多少在后台运行的命令。
```
`jobs -l`选项可显示所有任务的PID，jobs的状态可以是running, stopped, Terminated。但是如果任务被终止了（kill），shell 从当前的shell环境已知的列表中删除任务的进程标识。

## 2>&1解析
```shell
command >out.file 2>&1 &
```

1. command>out.file是将command的输出重定向到out.file文件，即输出内容不打印到屏幕上，而是输出到out.file文件中。
2. 2>&1 是将标准出错重定向到标准输出，这里的标准输出已经重定向到了out.file文件，即将标准出错也输出到out.file文件中。最后一个&， 是让该命令在后台执行。
3. 试想2>1代表什么，2与>结合代表错误重定向，而1则代表错误重定向到一个文件1，而不代表标准输出；换成2>&1，&与1结合就代表标准输出了，就变成错误重定向到标准输出.

---
# 查看系统状态

## 1. nvtop　　　
*查看NVIDIA显卡状态信息*　　　
```shell
# Install CMake, ncurses and git
sudo apt install cmake libncurses5-dev libncursesw5-dev git

git clone https://github.com/Syllo/nvtop.git
mkdir -p nvtop/build && cd nvtop/build
cmake ..

# If it errors with "Could NOT find NVML (missing: NVML_INCLUDE_DIRS)"
# try the following command instead, otherwise skip to the build with make.
cmake .. -DNVML_RETRIEVE_HEADER_ONLINE=True

make
make install # You may need sufficient permission for that (root)
```   

运行一下命令：　　　　
```shell
nvtop
```
![png](../img/nvtop.png)   


从上图可以看出**F1**被默认设置为关掉进程的快捷键，会跟系统的帮助快捷键冲突，所里这里需要修改３处源码，重新编译安装:    
```c
case KEY_F(1): //在nvtop.c#L297 
//改为:
case KEY_F(9):
```

```c
case KEY_F(1): //在interface.c#L1661
//改为：
case KEY_F(9):
```

```c
wprintw(win, "F%zu", i + 1);  //在interface.c#L1435
//改为：
if(i==0)
{
    wprintw(win, "F%zu", i + 9); 
}
else
{
    wprintw(win, "F%zu", i + 1);
}
```   
**重新编译安装后的效果：**　　　
![png](../img/nvtop_new.png)   

**注意**：https://github.com/Syllo/nvtop/commit/b126abb63f38d50e8fbb961ad0aedc11b51b3911 之后修复这个问题。    

## 2. htop　　　
*代替传统top命令*　　　

CPU监视可以用自带的`top`命令查看，但是推荐使用`htop`来显示，首先需要安装`htop`:    
```shell
sudo apt-get install htop
```
也可以通过源码安装：　　　
```shell
git clone https://github.com/hishamhm/htop.git
cd htop
./autogen.sh && ./configure && make
```   
然后输入以下命令显示CPU资源利用情况:    
```shell
htop
```
![png](../img/htop.png)   

## 3. glances　　　
*查看系统全部信息*   

```shell
curl -L https://bit.ly/glances | /bin/bash
```   
然后运行：
```shell
glances
```   
![png](../img/glances.png)    
