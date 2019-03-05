注意：本文内容适用于 Tmux 2.3 及以上的版本，但是绝大部分的特性低版本也都适用，鼠标支持、VI 模式、插件管理在低版本可能会与本文不兼容。

# Tmux 快捷键 & 速查表

启动新会话：

    tmux [new -s 会话名 -n 窗口名]

恢复会话：

    tmux at [-t 会话名]

列出所有会话：

    tmux ls

<a name="killSessions"></a>关闭会话：

    tmux kill-session -t 会话名

<a name="killAllSessions"></a>关闭所有会话：

    tmux ls | grep : | cut -d. -f1 | awk '{print substr($1, 0, length($1)-1)}' | xargs kill

# 在 Tmux 中，按下 Tmux 前缀 `ctrl+b`，然后：

## 会话

    :new<回车>  启动新会话
    s           列出所有会话
    $           重命名当前会话

## <a name="WindowsTabs"></a>窗口 (标签页)

    c  创建新窗口
    w  列出所有窗口
    n  后一个窗口
    p  前一个窗口
    f  查找窗口
    ,  重命名当前窗口
    &  关闭当前窗口

## 调整窗口排序

    swap-window -s 3 -t 1  交换 3 号和 1 号窗口
    swap-window -t 1       交换当前和 1 号窗口
    move-window -t 1       移动当前窗口到 1 号

## <a name="PanesSplits"></a>窗格（分割窗口） 

    %  垂直分割
    "  水平分割
    o  交换窗格
    x  关闭窗格
    ⍽  左边这个符号代表空格键 - 切换布局
    q 显示每个窗格是第几个，当数字出现的时候按数字几就选中第几个窗格
    { 与上一个窗格交换位置
    } 与下一个窗格交换位置
    z 切换窗格最大化/最小化

## <a name="syncPanes"></a>同步窗格

这么做可以切换到想要的窗口，输入 Tmux 前缀和一个冒号呼出命令提示行，然后输入：

```
:setw synchronize-panes
```

你可以指定开或关，否则重复执行命令会在两者间切换。
这个选项值针对某个窗口有效，不会影响别的会话和窗口。
完事儿之后再次执行命令来关闭。[帮助](http://blog.sanctum.geek.nz/sync-tmux-panes/)


## 调整窗格尺寸

如果你不喜欢默认布局，可以重调窗格的尺寸。虽然这很容易实现，但一般不需要这么干。这几个命令用来调整窗格：

    PREFIX : resize-pane -D          当前窗格向下扩大 1 格
    PREFIX : resize-pane -U          当前窗格向上扩大 1 格
    PREFIX : resize-pane -L          当前窗格向左扩大 1 格
    PREFIX : resize-pane -R          当前窗格向右扩大 1 格
    PREFIX : resize-pane -D 20       当前窗格向下扩大 20 格
    PREFIX : resize-pane -t 2 -L 20  编号为 2 的窗格向左扩大 20 格
    
    
## 文本复制模式：

按下 `PREFIX-[` 进入文本复制模式。可以使用方向键在屏幕中移动光标。默认情况下，方向键是启用的。在配置文件中启用 Vim 键盘布局来切换窗口、调整窗格大小。Tmux 也支持 Vi 模式。要是想启用 Vi 模式，只需要把下面这一行添加到 .tmux.conf 中：

    setw -g mode-keys vi

启用这条配置后，就可以使用 h、j、k、l 来移动光标了。

想要退出文本复制模式的话，按下回车键就可以了。然后按下 `PREFIX-]` 粘贴刚才复制的文本。

一次移动一格效率低下，在 Vi 模式启用的情况下，可以辅助一些别的快捷键高效工作。

例如，可以使用 w 键逐词移动，使用 b 键逐词回退。使用 f 键加上任意字符跳转到当前行第一次出现该字符的位置，使用 F 键达到相反的效果。

    vi             emacs        功能
    ^              M-m          反缩进
    Escape         C-g          清除选定内容
    Enter          M-w          复制选定内容
    j              Down         光标下移
    h              Left         光标左移
    l              Right        光标右移
    L                           光标移到尾行
    M              M-r          光标移到中间行
    H              M-R          光标移到首行
    k              Up           光标上移
    d              C-u          删除整行
    D              C-k          删除到行末
    $              C-e          移到行尾
    :              g            前往指定行
    C-d            M-Down       向下滚动半屏
    C-u            M-Up         向上滚动半屏
    C-f            Page down    下一页
    w              M-f          下一个词
    p              C-y          粘贴
    C-b            Page up      上一页
    b              M-b          上一个词
    q              Escape       退出
    C-Down or J    C-Down       向下翻
    C-Up or K      C-Up         向下翻
    n              n            继续搜索
    ?              C-r          向前搜索
    /              C-s          向后搜索
    0              C-a          移到行首
    Space          C-Space      开始选中
                   C-t          字符调序

## 杂项：

    d  退出 tmux（tmux 仍在后台运行）
    t  窗口中央显示一个数字时钟
    ?  列出所有快捷键
    :  命令提示符

## 配置选项：

    # 鼠标支持 - 设置为 on 来启用鼠标(与 2.1 之前的版本有区别，请自行查阅 man page)
    * set -g mouse on

    # 设置默认终端模式为 256color
    set -g default-terminal "screen-256color"

    # 启用活动警告
    setw -g monitor-activity on
    set -g visual-activity on

    # 居中窗口列表
    set -g status-justify centre

    # 最大化/恢复窗格
    unbind Up bind Up new-window -d -n tmp \; swap-pane -s tmp.1 \; select-window -t tmp
    unbind Down
    bind Down last-window \; swap-pane -s tmp.1 \; kill-window -t tmp

## 参考配置文件（~/.tmux.conf）：

下面这份配置是我使用 Tmux 几年来逐渐精简后的配置，请自取。

```bash
# -----------------------------------------------------------------------------
# Tmux 基本配置 - 要求 Tmux >= 2.3
# 如果不想使用插件，只需要将此节的内容写入 ~/.tmux.conf 即可
# -----------------------------------------------------------------------------

# C-b 和 VIM 冲突，修改 Prefix 组合键为 Control-Z，按键距离近
set -g prefix C-z

set -g base-index         1     # 窗口编号从 1 开始计数
set -g display-panes-time 10000 # PREFIX-Q 显示编号的驻留时长，单位 ms
set -g mouse              on    # 开启鼠标
set -g pane-base-index    1     # 窗格编号从 1 开始计数
set -g renumber-windows   on    # 关掉某个窗口后，编号重排

setw -g allow-rename      off   # 禁止活动进程修改窗口名
setw -g automatic-rename  off   # 禁止自动命名新窗口
setw -g mode-keys         vi    # 进入复制模式的时候使用 vi 键位（默认是 EMACS）

# -----------------------------------------------------------------------------
# 使用插件 - via tpm
#   1. 执行 git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm
#   2. 执行 bash ~/.tmux/plugins/tpm/bin/install_plugins
# -----------------------------------------------------------------------------

setenv -g TMUX_PLUGIN_MANAGER_PATH '~/.tmux/plugins'

# 推荐的插件（请去每个插件的仓库下读一读使用教程）
set -g @plugin 'seebi/tmux-colors-solarized'
set -g @plugin 'tmux-plugins/tmux-pain-control'
set -g @plugin 'tmux-plugins/tmux-prefix-highlight'
set -g @plugin 'tmux-plugins/tmux-resurrect'
set -g @plugin 'tmux-plugins/tmux-sensible'
set -g @plugin 'tmux-plugins/tmux-yank'
set -g @plugin 'tmux-plugins/tpm'

# tmux-resurrect
set -g @resurrect-dir '~/.tmux/resurrect'

# tmux-prefix-highlight
set -g status-right '#{prefix_highlight} #H | %a %Y-%m-%d %H:%M'
set -g @prefix_highlight_show_copy_mode 'on'
set -g @prefix_highlight_copy_mode_attr 'fg=white,bg=blue'

# 初始化 TPM 插件管理器 (放在配置文件的最后)
run '~/.tmux/plugins/tpm/tpm'

# -----------------------------------------------------------------------------
# 结束
# -----------------------------------------------------------------------------
```