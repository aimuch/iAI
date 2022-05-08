# ESXi下安装iKuai和LEDE双软路由

- [ESXi下安装iKuai和LEDE双软路由](#esxi%e4%b8%8b%e5%ae%89%e8%a3%85ikuai%e5%92%8clede%e5%8f%8c%e8%bd%af%e8%b7%af%e7%94%b1)
  - [硬件清单](#%e7%a1%ac%e4%bb%b6%e6%b8%85%e5%8d%95)
  - [网络拓扑图](#%e7%bd%91%e7%bb%9c%e6%8b%93%e6%89%91%e5%9b%be)
  - [ESXi](#esxi)
    - [下载ESXi](#%e4%b8%8b%e8%bd%bdesxi)
    - [制作启动U盘](#%e5%88%b6%e4%bd%9c%e5%90%af%e5%8a%a8u%e7%9b%98)
    - [软路由U盘启动安装ESXi](#%e8%bd%af%e8%b7%af%e7%94%b1u%e7%9b%98%e5%90%af%e5%8a%a8%e5%ae%89%e8%a3%85esxi)
    - [设置ESXi](#%e8%ae%be%e7%bd%aeesxi)
    - [ESXi的web端设置](#esxi%e7%9a%84web%e7%ab%af%e8%ae%be%e7%bd%ae)
      - [设置笔记本网卡IP](#%e8%ae%be%e7%bd%ae%e7%ac%94%e8%ae%b0%e6%9c%ac%e7%bd%91%e5%8d%a1ip)
      - [激活ESXi](#%e6%bf%80%e6%b4%bbesxi)
      - [打开虚拟机交换机的混杂模式](#%e6%89%93%e5%bc%80%e8%99%9a%e6%8b%9f%e6%9c%ba%e4%ba%a4%e6%8d%a2%e6%9c%ba%e7%9a%84%e6%b7%b7%e6%9d%82%e6%a8%a1%e5%bc%8f)
      - [网卡非直通情况](#%e7%bd%91%e5%8d%a1%e9%9d%9e%e7%9b%b4%e9%80%9a%e6%83%85%e5%86%b5)
      - [网卡直通情况](#%e7%bd%91%e5%8d%a1%e7%9b%b4%e9%80%9a%e6%83%85%e5%86%b5)
  - [iKuai](#ikuai)
    - [安装iKuai](#%e5%ae%89%e8%a3%85ikuai)
      - [网卡非直通情况](#%e7%bd%91%e5%8d%a1%e9%9d%9e%e7%9b%b4%e9%80%9a%e6%83%85%e5%86%b5-1)
      - [网卡直通情况](#%e7%bd%91%e5%8d%a1%e7%9b%b4%e9%80%9a%e6%83%85%e5%86%b5-1)
    - [设置iKuai](#%e8%ae%be%e7%bd%aeikuai)
    - [iKuai的web端设置](#ikuai%e7%9a%84web%e7%ab%af%e8%ae%be%e7%bd%ae)
  - [LEDE](#lede)
    - [安装LEDE](#%e5%ae%89%e8%a3%85lede)
      - [网卡非直通情况](#%e7%bd%91%e5%8d%a1%e9%9d%9e%e7%9b%b4%e9%80%9a%e6%83%85%e5%86%b5-2)
      - [网卡直通情况](#%e7%bd%91%e5%8d%a1%e7%9b%b4%e9%80%9a%e6%83%85%e5%86%b5-2)
    - [设置LEDE](#%e8%ae%be%e7%bd%aelede)
    - [LEDE的web端设置](#lede%e7%9a%84web%e7%ab%af%e8%ae%be%e7%bd%ae)
      - [LEDE作为旁路由](#lede作为旁路由)
      - [LEDE作为二级路由](#lede作为二级路由)
    - [LEDE常见问题](#lede%e5%b8%b8%e8%a7%81%e9%97%ae%e9%a2%98)
  - [设置iKuai和LEDE开机自动启动](#%e8%ae%be%e7%bd%aeikuai%e5%92%8clede%e5%bc%80%e6%9c%ba%e8%87%aa%e5%8a%a8%e5%90%af%e5%8a%a8)
    - [iKuai和LEDE虚拟机里设置](#ikuai%e5%92%8clede%e8%99%9a%e6%8b%9f%e6%9c%ba%e9%87%8c%e8%ae%be%e7%bd%ae)
    - [ESXi中设置自动启动](#esxi%e4%b8%ad%e8%ae%be%e7%bd%ae%e8%87%aa%e5%8a%a8%e5%90%af%e5%8a%a8)
  - [参考资料](#%e5%8f%82%e8%80%83%e8%b5%84%e6%96%99)



## 硬件清单
- CPU: I5 7200U
- 网卡: 6个Intel 82583V 10/100/1000以太网
- 内存: 金士顿 DDR4 2400 8G
- SSD: 三星 EVO850
- USB: 3.0x4个
- 电源: 技嘉12V

## 网络拓扑图
![网络拓扑图](network-topology.png)

- 第1个网口为管理口
- 第6个网口为wan口

## ESXi
### 下载ESXi
到VMWare官方网站[下载ESXi 6.7版本](https://my.vmware.com/cn/web/vmware/info/slug/datacenter_cloud_infrastructure/vmware_vsphere/6_7)，下载的时候需要注册账号，然后申请试用，序列号请自行购买正版(或Google)。

### 制作启动U盘
- **用老毛桃工具**
  到老毛桃官网下载[装机版老毛桃软件](http://www.laomaotao.org/):
  ![下载装机版老毛桃](laomaotao.png)
  然后用老毛桃烧写U盘:
  ![老毛桃烧写U盘](laomaotao1.png)
  在烧写好的U盘根目录中创建一个名字为**MYEXT**的文件夹，将下载的ESXi的ISO镜像放入该文件夹下:
  ![创建MYEXT文件夹](laomaotao2.png)
  ![MYEXT文件夹放入镜像](laomaotao3.png)

- **用软碟通UltraISO**
  到软碟通官网下载UltraISO，然后启动并试用:
  ![启动UltraISO](UltraISO.png)
  ![导入ISO镜像](UltraISO1.png)
  ![写入U盘](UltraISO2.png)
  ![启动烧写](UltraISO3.png)
  ![烧写完成](UltraISO4.png)

### 软路由U盘启动安装ESXi
- **UltraISO**: U盘启动，按照提示安装即可。

- **老毛桃**: 用U盘启动软路由，进入PE，选在老毛桃PE中运行**自定义镜像**选项，选择刚刚放入MYEXT文件夹中的ESXi镜像，按照提示安装ESXi即可。
  ![老毛桃PE启界面](laomaotao4.png)
  ![运行自定义镜像](laomaotao5.png)
  ![选择镜像](laomaotao6.png)

进入安装界面:
![ESXi](ESXi-install1.png)
选择安装位置(我这里为了截图方便，我安装在VMWare虚拟机里)，多块硬盘已经要注意选择安装在哪块硬盘以及别安装在U盘上:
![选择安装ESXi的位置](ESXi-install-Local.png)
设置好密码(默认用户名为:root)，按照提示安装即可，注意安装结束的时候会提示**先移除安装介质/U盘**，然后按下回车重启:
![移除安装介质重启](ESXi-install2.png)
若启动前已经将网线插入则ESXi默认会根据网络动态获取IP，若启动前没有插网线则IP会为0.0.0.0:
![安装完成后重启界面](ESXi-install-finished.png)

### 设置ESXi
安装完ESXi后重启，然后按**F2**进入ESXi设置界面:
![进入设置界面](ESXi-setting1.png)
选择**Configure Management Network**进入设置:
![Configure Management Network](ESXi-setting2.png)

选择**Network Adopters**, 将所有的网卡选中，这样任性一张网卡可以进入ESXi后台:
![Network Adopters](ESXi-setting1_1.png)

选择**IPv4 Configuration**选项卡，然后回车:
![IPv4 Configuration](ESXi-setting3.png)
光标移动到**Set Static IPv4 address and network configuration**，然后用空格键选择，并设置：
![Set Static IPv4 address and network configuration](ESXi-setting4.png)
  - **IPv4 Address**: **10.10.10.100**
  - **Subnet Mask**: **255.255.255.0**
  - **Default Gateway**: **10.10.10.10**
  - 这里设置的默认网关`10.10.10.10`是给`iKuai`的后台地址

按下回车键，然后按ESC键并输入Y保存设置:
![保存设置](ESXi-setting5.png)
设置生效后的界面如下，可以看出IP变为静态的了:
![设置生效后的界面](ESXi-setting6.png)

### ESXi的web端设置
#### 设置笔记本网卡IP
将笔记本的网口用网线连接到软路由的**LAN1**口，并将笔记本有线网卡的IPv4设置为:
- **IPv4**: **10.10.10.111**
- **子网掩码**: **255.255.255.0**
- **默认网关/路由器**: **10.10.10.10**
![笔记本IP设置](ESXi-PC-IP.png)

浏览器打开**10.10.10.100**，然后输入设置的密码进入ESXi后台(用户名为root，密码为安装ESXi时候设置的密码):
![ESXI WEB login](ESXi-web-login.png)

#### 激活ESXi
默认ESXi有试用期限制，解除限制需要用序列号激活ESXi:
![ESXi activate](ESXi-activate.png)

#### 打开虚拟机交换机的混杂模式
网络 -> 虚拟交换机 -> vSwitcho -> 编辑设置 -> 安全 -> 混杂模式 -> 接受
![打开虚拟机的混杂模式](ESXi-NetCard-Setting.png)

#### 网卡非直通情况
- 网络 -> 虚拟交换机 -> 添加交换机(多少个网口添加多少个) -> 安全 -> 全部接受
- 网络 -> 端口组 -> 添加端口组(多少个网口添加多少个) -> 安全 -> 全部接受

#### 网卡直通情况
主机 -> 硬件 -> PCI设备 -> 下拉选择`支持直通`来筛选网卡 -> 选中后5个(共6个)端口。
**这里要注意不要将第一个(第一个好记)端口/LAN1设置为直通，否则导致进不去ESXi后台。**

![网卡直通激活前](ESXi-networkcard.png)
![网卡直通激活后](ESXi-networkcard1.png)

设置好以后要单机**重新引导主机**，否则可能导致进不去ESXi的后台。

## iKuai
这里iKuai作为主路由。
### 安装iKuai
下载iKuai的ISO文件，然后**创建iKuai虚拟机的流程为**:
存储->数据存储浏览器->创建文件夹->iKuai->将iKuai的ISO镜像上传到该文件夹下->虚拟机->新建虚拟机->客户机操作系统linux->客户机操作系统版本->其他64位->CD/DVD选择刚刚上传的iKuai的ISO文件->内存要勾选**预留所有客户机内存(全部锁定)**
![安装iKuai step1](iKuai-install1.png)
![安装iKuai step2](iKuai-install2.png)
![安装iKuai step3](iKuai-install3.png)
![安装iKuai step4](iKuai-install4.png)
![安装iKuai step5](iKuai-install5.png)
![安装iKuai step6](iKuai-install6.png)
![安装iKuai step7](iKuai-install7.png)

**这里要特别注意iKuai虚拟机的内存>=4G** ，否则会出现以下错误:

![iKuai install error](iKuai-install-error.png)

#### 网卡非直通情况
重新编辑iKuai虚拟机设 -> 添加网络适配器并选择第6块虚拟网卡**VM Network5** ，配置好的界面如下:
![网卡非直通](iKuai-feizhitong.png)

#### 网卡直通情况
重新编辑iKuai虚拟机设 -> 添加PCI设备并选择所有的网卡，配置好的界面如下:
![网卡直通](iKuai_zhitong.png)

### 设置iKuai
启动iKuai虚拟机会自动安装iKuai，安装完以后按下回车键，显示:
![iKuai finished](iKuai-lan.png)
输入数字2，然后设置LAN1的IP地址:10.10.10.10:
![iKuai finished](iKuai-lan1.png)
![iKuai finished](iKuai-lan2.png)
![iKuai 设置IP](iKuai-finished.png)


### iKuai的web端设置
电脑浏览器打开**10.10.10.10**，默认用户名`admin`，默认密码`admin`，进入iKuai的web客户端。

#### 绑定LAN口
网络设置 -> 内外网设置 -> 在接口状态会有4个空闲的网口 -> 单击lan1 -> 高级设置 -> 网卡扩展，将eth2、eth3、eth4和eth5绑定到lan1：
![iKuai LAN](iKuai_LAN.png)

#### 绑定WAN口:
网络设置 -> 内外网设置 -> 在接口状态会有一个空闲的wan1口 -> 外网网口单击一下绑定上述空闲的wan1口即可:
![iKuai WAN](iKuai-web1.png)

#### 添加DHCP服务
开启一条DHCP服务: 网络设置 -> DHCP服务端 -> 添加DCHP服务:
- 客户端地址: **10.10.10.101 - 10.10.10.254**
- 子网掩码: **255.255.255.0**
- 网关: **10.10.10.1**   , 这里的网关是LEDE后台的地址，iKuai作为主路由，LEDE作为旁路由
- 首选DNS: **114.114.114.114**
- 备选DNS: **223.5.5.5**
![iKuai DHCP](iKuai-web2.png)

**然后重启DHCP服务**

## LEDE
### 安装LEDE
到KoolShare官网下载LEDE的固件，虚拟机转盘或PE下写盘专用->要选择combined格式，因为uefi格式下LEDE编辑的时候硬盘显示错误:
![LEDE Download](lede1.png)
下载完成后解压到本地，然后用**StarWind Converte**将LEDE的.img格式转换为ESXi虚拟机文件。

**创建LEDE虚拟机的流程为**:
存储->数据存储浏览器->创建文件夹->LEDE->将上述转化后的两个文件上传到该目录下->虚拟机->新建虚拟机->客户机操作系统linux->客户机操作系统版本->其他64位->删除默认硬盘->添加现有硬盘->选择LEDE文件夹下上传的文件->内存要勾选**预留所有客户机内存(全部锁定)**:
![LEDE step1](LEDE-install1.png)
![LEDE step2](LEDE-install2.png)
![LEDE step3](LEDE-install3.png)
![LEDE step4](LEDE-install4.png)
![LEDE step5](LEDE-install5.png)

#### 网卡非直通情况
重新编辑配置LEDE虚拟机->设置添加**除了最后一块虚拟网卡外**的其余虚拟网卡，配置好的界面如下:
![网卡非直通](LEDE-feizhitong.png)

#### 网卡直通情况
重新编辑配置LEDE虚拟机->设置添加**除了最后一块PCI设备以外**的其余PCI设备，并对应其编号(最后一块给iKuai用了，假如错误的将所有PCI设备都加上了同时启动iKuai和LDDE的时候会提示设备错误)，配置好的界面如下:
![网卡直通](LEDE-zhitong.png)


### 设置LEDE
启动iKuai虚拟机会自动安装iKuai，安装好以后按下回车:
![LEDE step6](LEDE-install6.png)
设置IP地址为:**10.10.10.1** ，在虚拟机终端输入以下命令:
```shell
vim /etc/config/network
```
![LEDE step7](LEDE-install7.png)

然后将**option ipaddr 192.168.1.1**改为**option ipaddr 10.10.10.1** :
![LEDE step8](LEDE-install8.png)

然后重启LEDE虚拟机。


### LEDE的web端设置

#### LEDE作为旁路由
LEDE作为二级路由只需要配置好LAN接口中的**IPV4网关指向iKuai的地址**和**DNS服务器**即可，并且**忽略LAN接口的DHCP服务**。
![LEDE LAN Config](LEDE_LAN_config.png)
#### LEDE作为二级路由
电脑浏览器打开**10.10.10.10**，默认密码`koolshare`，进入iKuai的web客户端:
**网络 -> 接口 -> 关闭WAN -> 关闭WAN6 -> 编辑LAN**:
![LEDE WEB1](LEDE-web1.png)

LAN -> 物理接口 -> 接口 -> **选中**:
- bond0
- erspan0
- eth0 - echo4
- ip_vti0
- teql0

![LEDE WEB2](LEDE-web2.png)

所要外接路由器作为AP，则需要将DHCP服务器->高级设置->强制勾选上。

**LAN->基本设置->IPv4网关->10.10.10.10->使用自定义的DNS服务器->114.114.114.114 和 223.5.5.5**:
![LEDE WEB3](LEDE-web3.png)


### LEDE常见问题
安装好LEDE以后会遇到经常掉线的问题，可以尝试一下操作进行修复:
- 禁用启动项里的**mwan3**服务:
  ![禁用mwan3](lede-banwan3.png)
- 防火墙，区域=>转发，WAN口指向LAN口:
  ![防火墙区域转发](lede-firewall.png)
- LAN口强制DHCP打开:
  ![LAN-DHCP](lede-lan-force.png)
- DHCP唯一授权打开，取消DNS仅本地服务:
  ![DNS-local](lede-DHCP.png)


## 设置iKuai和LEDE开机自动启动
### iKuai和LEDE虚拟机里设置
在ESXi的WEB客户端->虚拟机->分别在iKuai和LEDE虚拟机上右键->自动启动 :
![iKuai LEDE开启自动启动](auto-start.png)

### ESXi中设置自动启动
管理->系统->自动启动->编辑设置->是 :
![ESXi 中设置自动启动](auto-start1.png)
![ESXi 中设置自动启动](auto-start2.png)


## 参考资料
> 1. [vedio talk 官网](https://www.vediotalk.com/)
> 2. [vedio talk YouTube频道](https://www.youtube.com/channel/UCaMih5WXqoXq7Hg0S_XJdOg)
> 3. [vedio talk 哔哩哔哩频道](https://space.bilibili.com/28459251)
> 4. [tutu生活志 YouTube频道](https://www.youtube.com/channel/UCuhAUKCdKrjYoMiJQc74ZkQ)
> 5. [JS神技能 YouTube频道](https://www.youtube.com/channel/UC6tPP3jOTKgjqfDgqMsaG4g)
> 6. [七线图 YouTube频道](https://www.youtube.com/channel/UCTVyuHy255movpxD44RuGqA)
> 7. [BIGDONGDONG YouTube频道](https://www.youtube.com/channel/UCpPswAyGzdRwWmiW5oTNnvA)