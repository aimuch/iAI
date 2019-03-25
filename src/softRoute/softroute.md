# 软路由

- [软路由](#%E8%BD%AF%E8%B7%AF%E7%94%B1)
  - [硬件清单](#%E7%A1%AC%E4%BB%B6%E6%B8%85%E5%8D%95)
  - [ESXi](#esxi)
    - [下载ESXi](#%E4%B8%8B%E8%BD%BDesxi)
    - [制作启动U盘](#%E5%88%B6%E4%BD%9C%E5%90%AF%E5%8A%A8u%E7%9B%98)
    - [软路由U盘启动安装ESXi](#%E8%BD%AF%E8%B7%AF%E7%94%B1u%E7%9B%98%E5%90%AF%E5%8A%A8%E5%AE%89%E8%A3%85esxi)
    - [设置ESXi](#%E8%AE%BE%E7%BD%AEesxi)
    - [进入ESXi后台](#%E8%BF%9B%E5%85%A5esxi%E5%90%8E%E5%8F%B0)
  - [iKuai](#ikuai)
  - [LEDE](#lede)
    - [下载LEDE固件并转换格式](#%E4%B8%8B%E8%BD%BDlede%E5%9B%BA%E4%BB%B6%E5%B9%B6%E8%BD%AC%E6%8D%A2%E6%A0%BC%E5%BC%8F)



## 硬件清单
- CPU: I5 7200U
- 网卡: 6个Intel 82583V 10/100/1000以太网
- 内存: 金士顿 DDR4 2400 8G
- SSD: 三星 EVO850
- USB: 3.0x4个
- 电源: 技嘉12V
  
## ESXi
### 下载ESXi
到VMWare官方网站[下载ESXi 6.7版本](https://my.vmware.com/cn/web/vmware/info/slug/datacenter_cloud_infrastructure/vmware_vsphere/6_7)，下载的时候需要注册账号，然后申请试用，序列号请自行购买正版(或Google)。

### 制作启动U盘
到老毛桃官网下载[装机版老毛桃软件](http://www.laomaotao.org/):   
![laomaotao](laomaotao.png)    
然后用老毛桃烧写U盘:    
[老毛桃烧写U盘图占位符]   

在烧写好的U盘根目录中创建一个名字为**MYEXT**的文件夹，将下载的ESXi的ISO镜像放入该文件夹下。 

### 软路由U盘启动安装ESXi
用U盘启动软路由，进入PE，选在老毛桃PE中运行**自定义镜像**选项，选择刚刚放入MYEXT文件夹中的ESXi镜像，按照提示安装ESXi即可。


### 设置ESXi
安装完ESXi后重启，然后按F2进入ESXi设置界面,选择**Configure Management Network**进入设置:   
- **Network Adapters**选项卡，默认选择第一个网卡即可
- **IPv4 Configuration**选项卡，选择**Set Static IPv4 address and network configuration**:    
  - **IPv4 Address**: 192.168.1.100
  - **Subnet Mask**: 255.255.255.0
  - **Default Gateway**: 192.168.1.254

保存退出即可。   

### 进入ESXi后台
浏览器打开192.168.1.100，然后输入设置的密码进入ESXi后台:   

- 网络->虚拟交换机->添加交换机(多少个网口添加多少个)->安全->全部接受
- 网络->端口组->添加端口组(多少个网口添加多少个)->安全->全部接受
- 存储->数据存储浏览器->创建文件夹->LEDE->将转换好的LEDE的ESXi虚拟机镜像上传到该文件夹下->虚拟机->新建虚拟机->客户机操作系统linux->客户机操作系统版本->其他64位->添加硬盘->添加现有硬盘->选择刚刚上传的LEDE虚拟机文件,**删除原来的硬盘**->添加网络适配器(多少个网口添加多少个)->保存->重新编辑配置(若内存栏出现错误，刷新一下浏览器重新编辑)->设置网卡对应的端口
- 存储->数据存储浏览器->创建文件夹->iKuai->将iKuai的ISO镜像上传到该文件夹下->虚拟机->新建虚拟机->客户机操作系统linux->客户机操作系统版本->其他64位->CD/DVD选择刚刚上传的iKuai的ISO文件->添加网络适配器(多少个网口添加多少个)->保存->重新编辑配置(若内存栏出现错误，刷新一下浏览器重新编辑)->设置网卡对应的端口




## iKuai
下载iKuai的ISO文件

## LEDE
### 下载LEDE固件并转换格式
到KoolShare官网下载LEDE的固件:    
![LEDE Download](lede1.png)    
下载完成后解压到本地，然后用**StarWind Converte**将LEDE的.img格式转换为ESXi虚拟机文件。    

