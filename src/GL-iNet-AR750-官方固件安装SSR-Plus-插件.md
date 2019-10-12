# GL-iNet AR750 官方固件安装SSR Plus 插件


## 简介
GLiNet 他家路由系统采用的是开源 **OpenWrt** 系统并在 *github* 上开源并提供 *sdk* 和很多插件。

GLiNet 有自家刷不死的 uboot 可玩性很高。这里没有直接去刷第三方固件是因为喜欢他们家定制的UI可以比较方便的启用ap模式加入网络。所以就想着自建编译ipk插件在开源的openwrt固件内安装插件。

## 需要准备的环境和工具
* 一台出国留学的Linux(这里用了ubuntu16.04 LTS)
* L大开源的插件 [Git-Lede](https://github.com/coolsnowwolf/lede)
* GLiNet 官方 [SDK](https://github.com/gl-inet/sdk)
* 还有一台 AR750 (官方openwrt-ar750-3.010.bin固件)
* SSH客户端这里用了**MobaXterm**


## ipk 编译过程

**不要用 root 用户 git 和编译！！！**

### 下载源码
分别clone官方 sdk 和 lede 到本地, 把`lede/package`插件目录覆盖至官方`sdk/ar71xx/package`目录中,然后切换到官方sdk主目录下`sdk/ar71xx/`

### 安装依赖
安装依赖命令行输入 `sudo apt-get update` ，然后输入:
```bash
sudo apt-get -y install build-essential asciidoc binutils bzip2 gawk gettext git libncurses5-dev libz-dev patch unzip zlib1g-dev lib32gcc1 libc6-dev-i386 subversion flex uglifyjs git-core gcc-multilib p7zip p7zip-full msmtp libssl-dev texinfo libglib2.0-dev xmlto qemu-utils upx libelf-dev autoconf automake libtool autopoint
```

### 配置
```bash
./scripts/feeds update -a
./scripts/feeds install -a
make menuconfig
```
选中luCI–>Applictions–>luci-app-ssr-plus

### 编译
ssrplus依赖于luci中的一些工具，所以要先编译luci-base组件，不编译此组件直接编译ssrplus会出现类似”bash: po2lmo: command not found“的错误提示
```bash
make package/feeds/luci/luci-base/compile V=99
```

编译ssrplus:
```bash
make package/lean/luci-app-ssr-plus/compile V=99
```
喝杯咖啡去吧，等会编译完bin目录下就会有相应编译完的ipk包了


### 安装
ssh 登陆路由器
复制 ipk 到 `/tmp` 目录下
```bash
opkg install xxx.ipk
```
如果提示缺少某些底层依赖ipk可以通过更新openwrt官方源获取安装`/etc/opkg/customfeeds.conf`
```bash
src/gz glinet_core http://download.gl-inet.com/releases/kmod-3.0/ar71xx/generic
src/gz glinet_base http://download.gl-inet.com/releases/packages-3.x/ar71xx/base
src/gz glinet_gli_pub http://download.gl-inet.com/releases/packages-3.x/ar71xx/gli_pub
src/gz glinet_packages http://download.gl-inet.com/releases/packages-3.x/ar71xx/packages
src/gz glinet_luci http://download.gl-inet.com/releases/packages-3.x/ar71xx/luci
src/gz glinet_routing http://download.gl-inet.com/releases/packages-3.x/ar71xx/routing
src/gz glinet_telephony http://download.gl-inet.com/releases/packages-3.x/ar71xx/telephony
src/gz glinet_glinet http://download.gl-inet.com/releases/packages-3.x/ar71xx/glinet
```
然后更新源:
```bash
opkg update
```

### 芝麻开门
SSR-PLLUS被隐藏了，编译好后装好机，输入以下命令即可出来
```bash
echo 0xDEADBEEF > /etc/config/google_fu_mode
```

## 参考
> 1. http://download.gl-inet.com/firmware/ar750/v1/
> 2. https://gist.github.com/sitsh/4afd4f7d4b18083c9ebad25adef48599
> 2. https://www.qiqisvm.life/archives/102