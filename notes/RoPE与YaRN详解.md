# RoPE 与 YaRN 详解

> 配套笔记：[day2_RMSNorm+RoPE(YaRN外推).ipynb](day2_RMSNorm+RoPE(YaRN外推).ipynb)
> 参考实现：[model_minimind.py](../model/model_minimind.py)
> 参数：`head_dim=96  rope_theta=1e6  original_max=2048  factor=16  max_pos=32768`

---

## 目录

- [0. 一页纸速查](#0-一页纸速查)
- [1. 为什么需要位置编码](#1-为什么需要位置编码)
- [2. RoPE 的构造](#2-rope-的构造)
- [3. 完整数据流：一个 8 维 token 走一遍](#3-完整数据流一个-8-维-token-走一遍)
- [4. YaRN：把 2048 外推到 32768](#4-yarn把-2048-外推到-32768)
- [5. 代码逐段讲解](#5-代码逐段讲解)
- [6. 已知问题（4 个坑）](#6-已知问题4-个坑)
- [7. 配置速查与调参](#7-配置速查与调参)
- [8. 常见误解 FAQ](#8-常见误解-faq)
- [9. 符号总表](#9-符号总表)
- [10. 自检清单](#10-自检清单)

---

## 0. 一页纸速查

### 三句话

1. **RoPE**：把一个 token 的 `head_dim` 个数两两配对，每对按"位置 × 角速度"旋转一个角度。这样两个 token 的内积只跟它们的**相对距离**有关。
2. **YaRN**：训练只见过 2048 长度，想用 32768。把"转得慢的那几组"的角速度乘上 0~1 之间的系数，让最远位置的角度落回训练时见过的范围。
3. **两者分工**：RoPE 负责"转"，YaRN 只负责"改转速"。**YaRN 从头到尾没有碰过任何 token。**

### 关键数字（MiniMind 真实参数）

| 量 | 值 |
|---|---|
| 组数 `dim/2` | **48**（96 维两两配对）|
| 原始角速度 `f_0` | 1.0 rad/token |
| 原始角速度 `f_47` | 1.33e-6 rad/token |
| 分界线 `low` | **8** |
| 分界线 `high` | **21** |
| 系数范围 | 组 0~8 → `1.0`；组 14 → `0.5673`；组 21~47 → `0.0625`（=1/16）|
| cos/sin 表大小 | `[32768, 96]`，每张约 **12 MB** |

### 一句话公式

$$
f'_i = f_i \cdot c_i,\qquad
c_i = 1-\rho_i+\frac{\rho_i}{s},\qquad
\rho_i=\mathrm{clamp}\!\left(\frac{i-\text{low}}{\text{high}-\text{low}},0,1\right)
$$

看不懂不要紧，下面全是人话版。

---

## 1. 为什么需要位置编码

Attention 本质上是一个**集合运算**：

$$
\mathrm{Attn}(X) = \mathrm{softmax}(QK^\top)\,V
$$

把所有 token 打乱顺序重排，`QK^T` 的结果完全一样——**模型根本不知道谁在前谁在后**。

所以必须额外注入位置信息。两种思路：

| | 做法 | 缺点 |
|---|---|---|
| **绝对位置嵌入**（老办法）| 给每个位置一个可学习向量，加到输入上 | 训练没见过的位置就完全失效，没法外推 |
| **旋转位置编码 RoPE** | 不动输入，把 `q`、`k` 按位置**旋转** | 天生表达"相对距离"，外推性好得多 |

**RoPE 的关键优势**：它是把位置"乘"进向量，而不是"加"进去，所以能保持"只看相对距离"这个性质。

---

## 2. RoPE 的构造

### 2.1 一个 token 的 96 个数怎么分组

`head_dim = 96`。RoPE 把**前半段和后半段**配对：

```
第 0 维   第 1 维  ...  第 47 维 │ 第 48 维  第 49 维  ...  第 95 维
└─────────── 前半段 ────────────┘ └─────────── 后半段 ────────────┘
      ↑                                  ↑
       └───────── 第 0 组 ────────────────┘
```

$$
\boxed{\text{第 } i \text{ 组} = (\text{第 } i \text{ 维},\ \text{第 } i+48 \text{ 维}),\qquad i = 0,1,\dots,47}
$$

```python
for i in range(48):        # dim // 2 = 48 组
    pair = (i, i + 48)     # 第 i 组 = 这两个维度
```

> **注意**：这种"前后半段配对"是 LLaMA / GPT-NeoX 的写法。原始 RoPE 论文用的是**相邻配对** $(x_0,x_1),(x_2,x_3)$，数学上完全等价，只是排布不同。看博客时别被绕晕。

### 2.2 二维旋转

第 $i$ 组的两个数 $(a,b)$ 被当成平面上的一个点，绕原点转一个角度：

$$
\begin{pmatrix}y_0\\y_1\end{pmatrix}
= \begin{pmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{pmatrix}
\begin{pmatrix}a\\b\end{pmatrix}
= \begin{pmatrix}a\cos\theta - b\sin\theta\\ a\sin\theta + b\cos\theta\end{pmatrix}
$$

**旋转不改变长度**：$y_0^2+y_1^2 = a^2+b^2$。这是能验证实现是否正确的第一条标准。

代码里没有真的构造矩阵，而是用了一个等价写法（见 [5.3](#53-rotate_half)）：

```python
y = x * cos + rotate_half(x) * sin
```

### 2.3 频率（角速度）

48 组转速各不相同，形成**等比数列**：

$$
\theta_i = \text{base}^{-\frac{2i}{d}},\qquad i=0,1,\dots,47
$$

| 组 | $\theta_i$ | 含义 |
|---|---|---|
| 0 | 1.0 | 每个 token 转 1 弧度（飞快）|
| 8 | 0.1 | |
| 16 | 0.01 | |
| 47 | 1.33e-6 | 几乎不动 |

**为什么是几何级数？** 和二进制计数的道理一样：低位变化快（精细）、高位变化慢（粗略）。不同频率组合，让模型同时拥有"毫米刻度"和"公里刻度"。

**`rope_theta = 1e6`（不是经典的 10000）**：base 越大 → 所有 $\theta_i$ 越小 → 转得越慢 → 波长越长 → 原生就支持更长上下文。Qwen 系列也是 `1e6`。

> ⚠️ **命名陷阱**：代码里叫 `freq`，但它**不是频率（Hz）**，而是**角速度**（弧度/token）。HF 里叫 `inv_freq`。真正的频率是 $f_i/2\pi$。

### 2.4 相对位置性质

位置 $m$ 的旋转矩阵记作 $R_m$，则：

$$
(R_m q)^\top (R_n k) = q^\top R_m^\top R_n k = q^\top R_{n-m}\, k
$$

**内积只依赖 $m-n$。** 实测（同一个 q、k 放在不同位置）：

```
同一个 q/k 放在位置 m,n 上的内积矩阵 (行=m, 列=n):
   [ -0.526,   1.394,   1.791,  0.375,  -1.465,  -1.942]
   [ -2.267,  -0.526,   1.394,  1.791,   0.375,  -1.465]
   [ -2.279,  -2.267,  -0.526,  1.394,   1.791,   0.375]
   [ -0.588,  -2.279,  -2.267,  -0.526,  1.394,   1.791]

m-n=0: [-0.5261, -0.5261, ...]  全相同? True
m-n=1: [-2.2667, -2.2667, ...]  全相同? True
m-n=2: [-2.2786, -2.2786, ...]  全相同? True
=> 每条对角线都是常数: True
```

> **验证前提**：必须是**同一份内容**放在不同位置。如果每个位置放不同向量，那当然不满足——这是第一次验证时容易犯的错。

---

## 3. 完整数据流：一个 8 维 token 走一遍

为了能打印，把规模缩小（真实是 96 维）：

```
dim = 8  →  4 组
base = 1e4
L = 2048（训练）, M = 32768（目标）, factor = 16
m = 100（这是第 100 个 token）
```

先算分界线：

```
inv_dim(32) = 1.0080 → floor → low  = 1
inv_dim(1)  = 2.5132 → ceil  → high = 3
```

### 步骤 0：输入 + 配对

```python
x = [1.0, 0.5, -0.5, 0.2,  0.8, -0.3, 0.6, -0.1]
#    └──── 前半段 ────────┘  └──── 后半段 ────────┘
```

| 组 | 配对 |
|---|---|
| 0 | `(1.0, 0.8)` |
| 1 | `(0.5, -0.3)` |
| 2 | `(-0.5, 0.6)` |
| 3 | `(0.2, -0.1)` |

### 步骤 1~3：算每组的角速度（**与 token 无关**）

```
ramp_i  = clamp((i-1)/(3-1), 0, 1)
xishu_i = 1 - ramp + ramp/16
```

| 组 $i$ | $\theta_i$ | ramp | xishu | freq $=\theta\times$ xishu | 态度 |
|---|---|---|---|---|---|
| 0 | 1.000000 | 0.000 | 1.00000 | 1.00000000 | 不动 |
| 1 | 0.100000 | 0.000 | 1.00000 | 0.10000000 | 不动 |
| 2 | 0.010000 | 0.500 | 0.53125 | 0.00531250 | 压一半 |
| 3 | 0.001000 | 1.000 | 0.06250 | 0.00006250 | 压 16 倍 |

### 步骤 4~5：乘位置，求 cos/sin（**token 从这里进来**）

| 组 | freq | angle $=100\times$ freq | 圈数 | 折算 0~360° | cos | sin |
|---|---|---|---|---|---|---|
| 0 | 1.00000000 | 100.0 rad | 15.91 | **329.58°** | 0.86232 | −0.50637 |
| 1 | 0.10000000 | 10.0 rad | 1.59 | **212.96°** | −0.83907 | −0.54402 |
| 2 | 0.00531250 | 0.53125 rad | 0.08 | **30.44°** | 0.86217 | 0.50661 |
| 3 | 0.00006250 | 0.00625 rad | 0.001 | **0.36°** | 0.99998 | 0.00625 |

**四组角度天差地别（330° / 213° / 30° / 0.36°）——因为它们转速差 1000 倍。**

### 步骤 6：4 个旋转同时发生

$$
y_i = x_i\cos - x_{i+4}\sin,\qquad y_{i+4}=x_{i+4}\cos + x_i\sin
$$

- 组 0：`(1.0, 0.8)` 转 329.58° → **`(1.2674, 0.1835)`**
- 组 1：`(0.5, −0.3)` 转 212.96° → **`(−0.5827, −0.0203)`**
- 组 2：`(−0.5, 0.6)` 转 30.44° → **`(−0.7351, 0.2640)`**
- 组 3：`(0.2, −0.1)` 转 0.36° → **`(0.2006, −0.0987)`**（几乎没动）

手算组 0 一遍：

```
y0 = 1.0 × 0.86232 − 0.8  × (−0.50637) = 0.86232 + 0.40510 = 1.2674
y4 = 0.8 × 0.86232 + 1.0  × (−0.50637) = 0.68986 − 0.50637 = 0.1835
```

### 结果

```
输入 x = [ 1.0,    0.5,    -0.5,     0.2,     0.8,     -0.3,    0.6,     -0.1  ]
输出 y = [ 1.2674, -0.5827, -0.7351,  0.2006,  0.1835,  -0.0203,  0.2640, -0.0987 ]
```

**长度验证**（旋转不改变长度）：

| 组 | 输入长度 | 输出长度 |
|---|---|---|
| 0 | 1.2806 | 1.2806 ✅ |
| 1 | 0.5831 | 0.5831 ✅ |
| 2 | 0.7810 | 0.7810 ✅ |
| 3 | 0.2236 | 0.2236 ✅ |
| **整个向量** | **1.624808** | **1.624808** ✅ |

### 一个必须点出来的细节：组 3 转了 0.36°，等于没转

`(0.2, −0.1)` → `(0.2006, −0.0987)`，变化在小数点后第三位。

这不是 bug，是**低频维度的本质**：

- 组 3 的角速度 = 0.0000625 rad/token
- 走 100 个 token 才转 0.36°
- 要走 **10 万个 token** 才转一圈

**在短距离上，低频组本来就是"摆设"，几乎不提供信息。它们只在超长距离上才有意义。** 这也解释了为什么 YaRN 敢放心压它们。

### 推广到 96 维

把上面的表从 4 行扩到 48 行，就是 MiniMind 做的事。**代码里没有任何"跨组"操作**，48 组完全独立。

---

## 4. YaRN：把 2048 外推到 32768

### 4.1 问题：外推为什么会崩

| 组 $i$ | 波长 $\lambda_i$（token/圈）| 训练 2048 转几圈 | 硬推到 32768 转几圈 |
|---|---|---|---|
| 0 | 6.3 | 326 圈 | 5215 圈 |
| 8 | 62.8 | 32.6 圈 | 521 圈 |
| 16 | 628.3 | 3.26 圈 | 52 圈 |
| **20** | **1986.9** | **1.03 圈** | 16.5 圈 |
| **21** | **2649.6** | **0.77 圈** | **12.4 圈** |
| **47** | **4.7e6** | **0.0004 圈** | 0.0072 圈 |

**转得慢的那些维度，训练时连一圈都没转完**（0.77 圈、0.0004 圈）。模型从没见过它们转到 1 圈以上的样子。一外推，直接冲进训练时完全没出现过的角度区间 → 注意力乱套。

转得快的维度（组 0，326 圈）反而没事——它早就把所有相位都遍历过了。

**所以问题只出在低频维度上。** 这是 YaRN 全部设计的出发点。

### 4.2 三种方案

| 方案 | 做法 | 问题 |
|---|---|---|
| **外推** | 什么都不做 | 低频维度越界 → 崩 |
| **插值 PI** | 所有频率**统一**除以 factor | 高频维度也被压，局部精度被削平 |
| **NTK-by-parts**（YaRN 核心）| **分维度处理**：高频不动、低频压缩、中间过渡 | ✅ 正解 |

> 类比：看地图。近处用大比例尺（不动），远处用小比例尺（压缩）。不能用一把尺子量整张地图。

### 4.3 分界线 low / high（不用公式版）

**第一步**：算出 48 个组各转几圈：

$$
\text{圈数}_i = \frac{2048}{2\pi \cdot 10^{6\cdot 2i/96}}
$$

> **小抄（不用背表）**：这批参数下（`base=1e6`, `d=96`），**每往后数 8 个组，波长正好变成 10 倍**：
>
> ```
> i=0  →      6.3 token/圈
> i=8  →     62.8   (×10)
> i=16 →    628.3   (×10)
> i=24 →   6283.2   (×10)
> i=32 →  62831.9
> i=40 → 628318.5
> ```
>
> 原因：每步倍率是 $\text{base}^{2/d}=10^{6\times 2/96}=10^{0.125}$，而 $0.125\times 8=1$。
> 换成别的 `d`/`base` 就不再是整齐的 10 倍，但"等比"这个性质永远成立。

| 组 | 圈数 | 组 | 圈数 | 组 | 圈数 | 组 | 圈数 |
|---|---|---|---|---|---|---|---|
| 0 | 325.95 | 12 | 10.31 | 24 | 0.33 | 36 | 0.01 |
| 1 | 244.43 | 13 | 7.73 | 25 | 0.24 | 37 | 0.01 |
| 2 | 183.29 | 14 | 5.80 | 26 | 0.18 | 38 | 0.01 |
| 3 | 137.45 | 15 | 4.35 | 27 | 0.14 | 39 | 0.00 |
| 4 | 103.07 | 16 | 3.26 | 28 | 0.10 | 40 | 0.00 |
| 5 | 77.29 | 17 | 2.44 | 29 | 0.08 | 41 | 0.00 |
| 6 | 57.96 | 18 | 1.83 | 30 | 0.06 | 42 | 0.00 |
| 7 | 43.47 | 19 | 1.37 | 31 | 0.04 | 43 | 0.00 |
| **8** | **32.59** | **20** | **1.03** | 32 | 0.03 | 44 | 0.00 |
| **9** | **24.44** | **21** | **0.77** | 33 | 0.02 | 45 | 0.00 |
| 10 | 18.33 | 22 | 0.58 | 34 | 0.02 | 46 | 0.00 |
| 11 | 13.75 | 23 | 0.43 | 35 | 0.01 | 47 | 0.00 |

**第二步**：扫一眼，找两个断点：

```
断点 1（32 圈那条线）：
  组 0~7：43 ~ 326 圈  ✅ 都 ≥ 32
  组 8：  32.59 圈     ✅ 刚好够    ← 最后一个够的
  组 9：  24.44 圈     ❌ 不够了     ← 第一个不够
  → low = 8

断点 2（1 圈那条线）：
  组 20： 1.03 圈      ✅ 刚好够    ← 最后一个够的
  组 21： 0.77 圈      ❌ 不够了     ← 第一个不够
  → high = 21
```

**完成。全程没有一个 log。**

> **`beta_fast = 32` 和 `beta_slow = 1` 就是问题里的那两个数字**：
> - `beta_fast`：**"至少转多少圈才算快维度"** → 32 圈
> - `beta_slow`：**"至少转多少圈才可信任"** → 1 圈

### 4.4 反解公式与 log 的唯一用途

代码不想循环 48 次，想一步算出来。但 `i` 卡在指数上：

$$
10^{6\cdot\frac{2i}{96}} = \frac{2048}{2\pi\cdot 32} = 10.1859
$$

**log 的唯一作用：把指数摘下来。**

$$
\log(a^b) = b\log a
$$

手算 4 行：

```
① 10^(6·2i/96) = 10.1859

② 两边取 ln：   (2i/96) × ln(1e6) = ln(10.1859)

③ 代入数字：    (2i/96) × 13.8155 = 2.3210

④ 解出 i：      i = 96 × 2.3210 / (2 × 13.8155) = 8.0640
```

**答案 8.0640 —— 和"数格子"数出来的 8 一致。**

同样的算法求 1 圈：`inv_dim(1) = 20.1052`。

```python
# 完整形式
def inv_dim(q):
    return d * math.log(L / (2 * math.pi * q)) / (2 * math.log(base))
```

**如果看不懂 log，用这个等价版本，结果一模一样：**

```python
i = torch.arange(48)
circles = L / (2 * math.pi * (base ** (2 * i / d)))

low  = (circles >= 32).nonzero().max().item()        # → 8
high = (circles >= 1 ).nonzero().max().item() + 1    # → 21
```

### 4.5 floor / ceil：往下踩，往上踩

```
8.0640  →  floor → 8     组 8 还够，保留
20.1052 →  ceil  → 21    组 20 还够，组 21 开始不够
```

小数尾巴 `.064` `.105` 在告诉你：**分界线不在某一组身上，而在两组之间。**

### 4.6 ramp 与 xishu：两个插值

**`ramp` 是"两点连一条直线"**：

两个已知点 $(8, 0)$ 和 $(21, 1)$，标准两点式：

$$
\text{ramp}_i = \frac{i-8}{21-8} = \frac{i-8}{13}
$$

**为什么要 clamp？** 因为 $i$ 从 0 跑到 47：

| $i$ | $(i-8)/13$ | 问题 |
|---|---|---|
| 0 | −0.615 | 负数，比例不能为负 |
| 47 | 3.000 | 超过 100% |

小于 0 的当 0，大于 1 的当 1。

**`xishu` 是"加权平均"**：

$$
\text{xishu} = \underbrace{(1-\text{ramp})}_{\text{"不动"占几成}}\times 1
+ \underbrace{\text{ramp}}_{\text{"压缩"占几成}}\times\frac{1}{16}
= 1-\text{ramp}+\frac{\text{ramp}}{16}
$$

| ramp | 计算 | 结果 | 含义 |
|---|---|---|---|
| 0.000 | $1-0+0$ | **1.0000** | 100% 不动 |
| 0.250 | $1-0.25+0.0156$ | 0.7656 | 75% 不动 |
| 0.500 | $1-0.5+0.03125$ | 0.5313 | 一半一半 |
| 1.000 | $1-1+0.0625$ | **0.0625** | 100% 压缩 |

**两行连起来就是一个梯形**：

```
xishu
1.00 ┤━━━━━━━━━╮
     │         ╰╮
     │          ╰╮
0.53 ┤           ╰╮  (i=14)
     │            ╰╮
0.13 ┤             ╰╮  (i=20)
0.06 ┤              ╰━━━━━━━━━━━━━━
     └──────┬──────┬──────────────→ 组号 i
            8      21
```

**为什么用过渡而不是一刀切？**

如果直接 `if i<=8: 1 else: 1/16`，组 8 和组 9 的系数差 **16 倍**，而它们是物理上相邻的两个维度（第 8 维 vs 第 9 维）。突变会让模型学到的表示在这两维之间**断裂**。ramp 的作用是把 16 倍的落差摊平在 13 个组上（每组只差约 9%）。

### 4.7 效果验证

| 组 | 训练时最大角度 | 硬外推到 32768 | YaRN 后 | 状态 |
|---|---|---|---|---|
| 0 | 2048 rad（326 圈）| 32768（5215 圈）| 32768 | 不变（本来就安全）|
| 8 | 204.8（32.6 圈）| 3276.8（521 圈）| 3276.8 | 不变（本来就安全）|
| 12 | 64.76（10.3 圈）| 1036（165 圈）| **737（117 圈）** | 拉回一部分 |
| 16 | 20.48（3.26 圈）| 327.7（52 圈）| **138.6（22 圈）** | 拉回一部分 |
| **21** | 4.857（0.77 圈）| 77.7（12.4 圈）| **4.857（0.77 圈）** | ✅ **精确复原** |
| **47** | 0.0027（0.0004 圈）| 0.044 | **0.0027（0.0004 圈）** | ✅ **精确复原** |

看组 21 和组 47：**YaRN 之后的角度和训练时最大角度一模一样**。这是插值的本质——把 32768 个位置压进 2048 个位置的角度空间。

过渡区的组（12、16）只被拉回一部分，这是有意的：它们是"半信半疑"的中间地带。

### 4.8 完整数值链（$i=14$）

```
(1) f_14  = 1e6^(-2×14/96)              = 0.01778279 rad/token
(2) λ_14  = 2π / f_14                   = 353.3 token
(3) r_14  = 2048 / λ_14                 = 5.80 圈

(4) G(32) = 8.0640      G(1) = 20.1052

(5) low = 8,  high = 21

(6) ρ_14 = (14-8)/(21-8)                = 0.4615
(7) c_14 = 1 - 0.4615 + 0.4615/16       = 0.5673
(8) f'_14 = 0.01778279 × 0.5673         = 0.01008832 rad/token   ← YaRN 结束

(9) m=100: θ = 100 × 0.01008832         = 1.008832 rad = 57.80°
(10) cos = 0.532850,  sin = 0.846210

(11) 若 q_14=0.8, q_62=-0.3：
     q'_14 = 0.8×0.53285 - (-0.3)×0.84621 = 0.68014
     q'_62 = (-0.3)×0.53285 + 0.8×0.84621 = 0.51711
     (长度 0.854400 → 0.854400 不变 ✅)
```

### 4.9 关于 attn_factor

```python
cos_freqs = ... * attn_factor
sin_freqs = ... * attn_factor
```

上下文一长，attention 分布会被摊平（熵变大）。YaRN 用一个温度系数把分布拉尖锐。

**乘在 cos/sin 上 = q、k 各放大 $a$ 倍 = $q^\top k$ 放大 $a^2$ 倍。**

YaRN 论文的标准公式：

$$
\text{mscale} = 0.1\ln(s)+1
$$

$s=16$ 时 ≈ **1.277**。但 **MiniMind 配置写的是 `1.0`，也就是没启用**。

---

## 5. 代码逐段讲解

### 5.1 RMSNorm

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # 可学习缩放参数 γ

    def _norm(self, x):
        '''x: [B, S, dim]  →  返回: [B, S, 1]'''
        return torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self._norm(x.float())).type_as(x)   # 广播
```

**形状实测**：

```
输入 x                       (2, 5, 96)
x.pow(2)                     (2, 5, 96)
mean(dim=-1, keepdim=True)   (2, 5, 1)   ← 最后一维变成 1
+eps -> rsqrt                (2, 5, 1)
weight.shape                 (96,)       ← 一维，靠广播对齐到最后一维
输出                         (2, 5, 96)
```

**每一步在干嘛**：

| 代码 | 数学 | 作用 |
|---|---|---|
| `x.pow(2)` | $x_i^2$ | 每个数平方 |
| `.mean(dim=-1, keepdim=True)` | $\frac{1}{d}\sum x_i^2$ | 最后一维求平均 |
| `+ self.eps` | $+\varepsilon$ | 防除以 0 |
| `torch.rsqrt(...)` | $1/\sqrt{\cdot}$ | 开根号取倒数 |
| `* self.weight` | $\times\gamma$ | 每维一个可学习缩放 |

**两个 dtype 操作**：

- `x.float()`：转 fp32 再算，bf16 尾数只有 8 位，平方后精度掉得厉害
- `.type_as(x)`：算完转回原精度

**广播对齐**：

```
   weight  :            [  96]
   _norm   : [2, 5, 1]
   ─────────────────────────────
   补 1 后  : [1, 1, 96]
   结果     : [2, 5, 96]
```

**手算例子**（`x = [1,-2,3,-4]`）：

```
平方 [1, 4, 9, 16] → 和 30 → 平均 7.5 → +eps → 7.500001
rsqrt → 0.365148
输出 [0.3651, -0.7303, 1.0954, -1.4606]
验算：输出的平方平均 = 1 ✅
```

> ⚠️ `keepdim=True` **绝对不能省**。省了形状变 `[2,5]`，和 `[96]` 相乘时会对齐到 `S` 维，结果全错，**而且不报错**。

**在 MiniMind 里的 5 处使用**：

| 位置 | 作用 |
|---|---|
| `input_norm` | Attention 之前（Pre-Norm）|
| `post_attention_norm` | FFN/MoE 之前（Pre-Norm）|
| `q_norm` / `k_norm` | 对每个 head 的 q、k 归一化（Qwen3 风格 QK-Norm），`dim=head_dim` |
| `self.norm` | 全部层结束后、进 lm_head 之前 |

---

### 5.2 permute_freq_cls（预计算 cos/sin 表）

```python
def precompute_freqs_cis(dim, rope_base=1e6, end=32*1024, rope_scaling=None):
    # (1) 算 48 个角速度
    freq, attn_factor = 1 / torch.pow(rope_base, (torch.arange(0, dim, 2).float() / dim)), 1.0

    if rope_scaling is not None:
        origin_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling["original_max_position_embeddings"],
            rope_scaling["factor"],
            rope_scaling["beta_fast"],
            rope_scaling["beta_slow"],
            rope_scaling["attention_factor"],       # 注意 key 名
        )
        # (4) 只有需要外推时才启用
        if end / origin_max > 1.0:
            inv_dim = lambda q: dim * math.log(origin_max / (2*math.pi*q)) / (2*math.log(rope_base))
            low, high = (max(math.floor(inv_dim(beta_fast)), 0),
                         min(math.ceil (inv_dim(beta_slow)), dim//2))
            ramp  = torch.clamp((torch.arange(dim//2) - low).float()
                                / max(high - low, 0.001), 0, 1)
            xishu = 1 - ramp + ramp / factor
            freq  = xishu * freq

    # (9) 位置 × 角速度 → 角度矩阵
    freqs = torch.outer(torch.arange(end, device=freq.device), freq).float()

    # (10) cos/sin，复制一份拼成 96 宽
    cos_freqs = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    sin_freqs = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

    return cos_freqs, sin_freqs
```

**形状实测**：

```
arange(0, dim, 2)     (48,)          值: [0, 2, 4, 6, 8, 10] ... 共 48 个
arange/dim            (48,)          值: [0.0, 0.0208, 0.0417, 0.0625, ...]
base ** 上面          (48,)          值: [1.00e+00, 1.33e+00, 1.78e+00, 2.37e+00]
取倒数 = freq         (48,)          值: [1.0, 0.749894, 0.562341, 0.421697]
low=8  high=21
ramp                  (48,)          值(每6取1): [0.0, 0.0, 0.308, 0.769, 1.0, 1.0, 1.0, 1.0]
xishu                 (48,)          值: [1.0, 1.0, 0.7115, 0.2788, 0.0625, ...]
outer(32768, freq)    (32768, 48)    ← [位置, 组]
cos(freqs)            (32768, 48)
cat([x,x], -1)        (32768, 96)    ← 复制一份拼后面
```

**实测验证**：

```
cos[0,:4] = [1.0, 1.0, 1.0, 1.0]     ← 位置 0，角度 0，全是 1
sin[0,:4] = [0.0, 0.0, 0.0, 0.0]     ← 位置 0，转了等于没转
cos[1,:4] = [0.5403, 0.7318, 0.846, 0.9124]
```

**内存**：`32768 × 96 × 4 字节 ≈ 12.0 MB` 每张，两张 24 MB。

**命名注意**：

| 名字 | 说明 |
|---|---|
| notebook 里叫 `permute_freq_cls` | **叫错了**，它不做 permute，只是预计算 |
| model 文件里叫 `precompute_freqs_cis` | ✅ 正确 |
| `cis` | $\text{cis}(\theta)=\cos\theta+i\sin\theta$ |
| `freq` | 角速度（rad/token），**不是频率** |

**`register_buffer`**（在 `MiniMindModel.__init__` 里）：

```python
self.register_buffer("freqs_cos", freqs_cos, persistent=False)
self.register_buffer("freqs_sin", freqs_sin, persistent=False)
```

- 是 buffer → `.to(device)` 会自动跟着搬
- `persistent=False` → **不写进 checkpoint**（能重算，省 24 MB）

---

### 5.3 rotate_half

```python
def rotate_half(x):
    return torch.cat([-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]], dim=-1)
```

**实测**：

```
输入 [0,1,2,3]         → [-2.0, -3.0,  0.0,  1.0]
输入 [0,1,2,3,4,5,6,7] → [-4.0, -5.0, -6.0, -7.0,  0.0, 1.0, 2.0, 3.0]
```

**两个切片**：

| 代码 | 取到 | 做什么 |
|---|---|---|
| `x[..., h:]` | 后半段 | 加负号 |
| `x[..., :h]` | 前半段 | 原样 |

`cat` 的顺序是 **`[-后半, 前半]`**。

**这个顺序必须和 cos 的复制顺序（`[cos, cos]`）配合**：

$$
\text{out}_i = x_i\cdot c_i + (\text{rot}_i)\cdot s_i
$$

- $i < h$：$\text{rot}_i = -x_{i+h}$ ✅
- $i \ge h$：$\text{rot}_i = x_{i-h}$ ✅

**顺序搞反，整个 RoPE 就变成错的旋转，而且不报错。**

---

### 5.4 apply_rope

```python
def apply_rope(q, k, cos, sin, unsqueeze_dim=1):
    # q, k 形状为 [B, seq_len, num_heads, head_dim]
    # cos, sin 形状为 [seq_len, head_dim]
    def rotate_half(x):
        return torch.cat([-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]], dim=-1)

    q_rope = (q * cos.unsqueeze(unsqueeze_dim)
              + rotate_half(q) * sin.unsqueeze(unsqueeze_dim)).to(q.dtype)
    k_rope = (k * cos.unsqueeze(unsqueeze_dim)
              + rotate_half(k) * sin.unsqueeze(unsqueeze_dim)).to(k.dtype)
    return q_rope, k_rope
```

**形状实测**：

```
q                    (2, 5, 8, 96)
cos[:S]              (5, 96)
cos[:S].unsqueeze(1) (5, 1, 96)     ← 插入的 1 占住 heads 维
q * cos.unsqueeze(1) (2, 5, 8, 96)
q_rope               (2, 5, 8, 96)
k_rope               (2, 5, 4, 96)  ← 省了 4 个头（GQA）
```

**广播对齐**：

```
q   : [2,  5,  8,  96]
cos :      [5,  1,  96]
             ↑   ↑
          seq   heads（1 广播成 8）
```

**为什么 `unsqueeze_dim=1`**：`cos` 没有 batch 维，需要从右往左数第 3 个位置插进去才对上 `seq`：

```
从右数      -1    -2    -3    -4
q   :       96     8     5     2
cos :       96     1     5    (无)
```

**换布局时这是第一个会错的地方**（HF 的 `[B,H,S,D]` 需不同写法）。

**`.to(q.dtype)`**：表是 fp32 建的，q 可能是 bf16，fp32×bf16 → fp32，得转回来。

> ⚠️ **notebook 这版少了 `return q_rope, k_rope`**，见 [6.1](#6-已知问题4-个坑)。

---

### 5.5 调用链

**初始化时调一次**（[model_minimind.py:274](../model/model_minimind.py)）：

```python
freqs_cos, freqs_sin = precompute_freqs_cis(
    dim=config.head_dim, rope_base=config.rope_theta,
    end=config.max_position_embeddings, rope_scaling=config.rope_scaling)
```

**每次 forward 切出这一段对应的行**（第 303 行）：

```python
pos_embedding = (self.freqs_cos[start_pos : start_pos+seq_length],
                 self.freqs_sin[start_pos : start_pos+seq_length])
```

**每层 attention 里施加**（第 142 行）：

```python
cos, sin = position_embeddings
xq, xk = apply_rope(xq, xk, cos, sin)
```

`start_pos` 的定义（第 291 行）：

```python
start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
```

- 训练 / 首次 prefill：`past_key_values` 空 → `start_pos = 0`
- 生成第 k 个 token：缓存已有 `P` 个 → `start_pos = P`

---

## 6. 已知问题（4 个坑）

### 6.1 `apply_rope` 少了 `return`

**位置**：notebook 最后一格。

```python
def apply_rope(q, k, cos, sin, unsqueeze_dim=1):
    ...
    k_rope = (...).to(k.dtype)
    # ← 到此结束，函数返回 None
```

**实际报错**：

```
TypeError: cannot unpack non-iterable NoneType object
```

**修法**：

```python
    return q_rope, k_rope
```

### 6.2 `attn_factor` 与 `attention_factor` 名字不一致

- `MiniMindConfig` 定义的是 `"attention_factor"`（[第 36 行](../model/model_minimind.py)）
- `precompute_freqs_cis` 读的是 `rope_scaling["attn_factor"]`（[第 73 行](../model/model_minimind.py)）

**实际报错**（当 `inference_rope_scaling=True`）：

```
KeyError: 'attn_factor'
```

默认 `False` 所以平时不触发。**修法**：两边统一成 `"attention_factor"`（HF 的叫法）。

### 6.3 RMSNorm 把 dtype 泄漏成 fp32

**差分只在一处括号**：

```python
# notebook（有泄漏）
return self.weight * self._norm(x.float()).type_as(x)

# model_minimind.py 第 61 行（正确）
return (self.weight * self._norm(x.float())).type_as(x)
```

**实测**：

```
输入 dtype      : torch.bfloat16
notebook 版输出 : torch.float32    ← 变回 fp32 了！
model 文件版输出: torch.bfloat16    ← 保持 bf16
weight dtype    : torch.float32
```

**原因**：`self.weight` 是 fp32，`bf16 × fp32` 会按类型提升规则升回 fp32。在 bf16 训练下会让残差流混进 fp32 张量。

### 6.4 注释位置写错

```python
# 这里 freq 大小为 [end, dim // 2]      ← 错
freq, attn_factor = 1 / torch.pow(...)
```

`freq` 实际是 `[48]` 一维。`[end, dim//2]` 是几行之后 `freqs` 的形状。容易让人把两个变量搞混。

---

## 7. 配置速查与调参

### 7.1 MiniMind 的实际配置

```python
self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)   # M
self.rope_theta = kwargs.get("rope_theta", 1e6)                               # base

self.rope_scaling = {
    "beta_fast": 32,                            # 至少转 32 圈才算"快"
    "beta_slow": 1,                             # 至少转 1 圈才可信任
    "factor": 16,                               # 压缩倍数
    "original_max_position_embeddings": 2048,   # L
    "attention_factor": 1.0,                    # 温度，1.0 = 不启用
    "type": "yarn",
} if self.inference_rope_scaling else None
```

### 7.2 最重要的约束：`factor == M / L`

$$
s = \frac{M}{L}\quad\Longleftrightarrow\quad M\cdot\frac{f}{s}=L\cdot f
$$

左边是"目标长度下的最大角度"，右边是"训练长度下的最大角度"。**两者必须相等。**

MiniMind：$\frac{32768}{2048} = 16 = s$ ✅

| 情况 | 现象 |
|---|---|
| `factor` 太大 | 压过头，长文感知不到区别（只用到训练角度范围的 1/(s·L/M)）|
| `factor` 太小 | 压不够，长文直接越界崩掉 |

**代码只检查了 $M/L>1$，没有检查 $s=M/L$，得自己保证。**

### 7.3 `beta_fast` / `beta_slow` 的敏感度

固定 `beta_slow=1`，调 `beta_fast`：

| `beta_fast` | 门槛 | `low` | 不动区 | 过渡带 | 倾向 |
|---|---|---|---|---|---|
| 4 | 至少 4 圈 | 15 | 0~15（16 组）| 6 组 | 接近纯外推 |
| 8 | 至少 8 圈 | 12 | 0~12（13 组）| 9 组 | 保守 |
| 16 | 至少 16 圈 | 10 | 0~10（11 组）| 11 组 | |
| **32** | **至少 32 圈** | **8** | **0~8（9 组）** | **13 组** | **默认** |
| 64 | 至少 64 圈 | 5 | 0~5（6 组）| 16 组 | 激进 |
| 128 | 至少 128 圈 | 3 | 0~3（4 组）| 18 组 | 很激进 |

固定 `beta_fast=32`，调 `beta_slow`：

| `beta_slow` | 门槛 | `high` | 全压区 | 过渡带 |
|---|---|---|---|---|
| 0.5 | 不足 0.5 圈 | 23 | 23~47（25 组）| 15 组 |
| **1.0** | **不足 1.0 圈** | **21** | **21~47（27 组）** | **13 组** |
| 2.0 | 不足 2 圈 | 18 | 18~47（30 组）| 10 组 |
| 4.0 | 不足 4 圈 | 16 | 16~47（32 组）| 8 组 |
| 8.0 | 不足 8 圈 | 13 | 13~47（35 组）| 5 组 |

**两个都调大 = 更激进地压缩。**

### 7.4 这两个值是哪来的

- **`beta_slow = 1` 有硬道理**：转不满一圈的维度，训练中相位没走遍，模型学不到可靠位置信息 → 必须压。1 是"一个完整周期"的自然门槛。
- **`beta_fast = 32` 是量级判断**：32 圈 ⇒ 波长 64 token ⇒ 每 token 相位差 5.7°，完全能分辨相邻位置 → 这在管局部细节，动不得。
- 其余部分：**论文直接给的经验值**，HF 设成默认，Qwen / Llama 3.1 / DeepSeek 照抄，成了事实标准。

**不是推导出来的，是调出来的。** 改它要重做长文本评测，成本高，所以没人动。

---

## 8. 常见误解 FAQ

### Q1：`i` 是"维度"还是"组"？

**组号**。范围 **0~47**（`dim/2`），不是 0~95。

| 名词 | 范围 |
|---|---|
| 维度 | 0 ~ 95（96 个）|
| 组 = 维度对 $i$ | 0 ~ 47（**48 个**）|

所有带下标 $i$ 的量（$\theta_i$、$\lambda_i$、`ramp[i]`）都是**组号**。

### Q2：波长 $\lambda_i$ 的单位是 token 还是组？

**token**。因为角度是 $m\cdot f_i$，而 $m$ 每加 1 就走过一个 token。转一圈需要 $\frac{2\pi}{f_i}$ **个 token**。

组号决定的是**转速**，不是"转了多久"。用钟表类比：**组 = 哪根针，token = 过了多久**。

### Q3：`inv_dim` 里为什么没有 `i`？

因为它就是**解出来的 `i`**。原式是 $2\pi\cdot\text{base}^{2i/d}=L/q$，$i$ 卡在指数上；解完之后 $i$ 单独跑到等号左边，右边就只剩已知量了。

就像 $3x=12$ 解成 $x=4$，右边没有 $x$。

### Q4：解出来 8.064 是小数，`i` 不是整数吗？

物理上 $i$ 只有 48 个整数值，这是硬约束。但解方程时先当连续变量解，再取整：

- 8.064 说明分界**落在组 8 和组 9 之间**
- `floor` → 8：最后一个还够快的
- 20.105 → `ceil` → 21：第一个不够的

### Q5：YaRN 是"超过 2048 才启用"的开关吗？

**不是。** 它是**静态替换频率表**，从第 0 个 token 到最后一个 token 全程生效，比例恒定。

```
             组0~8      组9~20     组21~47
             ×1         ×0.53...   ×1/16
m=0          ×1         ×b         ×1/16
m=1          ×1         ×b         ×1/16
...          ×1         ×b         ×1/16
m=32767      ×1         ×b         ×1/16
```

- **横着看**（同一 token，48 组）：分三段 ← **这才是 YaRN 的分段**
- **竖着看**（同一组，所有 token）：系数恒定

**分段发生在"维度轴"上，不发生在"位置轴"上。**

而且它**不能**做成运行时开关：如果组 8 在第 2047 个 token 用原频率、第 2048 个突然换成压缩频率，位置编码在边界上会断掉，模型立刻崩。

### Q6：位置 0 的 token 转了等于没转？

**是的**，因为 $\text{angle}=0\times f=0$，$\cos 0=1$，$\sin 0=0$，所以输出 = 输入。

但**流程上照样走了**：`cos[0]` 全是 1，`sin[0]` 全是 0，乘出来就是原样。

第一个 token（通常是 BOS）也走一整套，只是转 0°。

### Q7：每个 token 都要转吗？还是从某个 token 之后才开始？

**每个都转，从位置 0 开始，一个不漏。**

而且是**同一个 token 内容放在不同位置，转出来完全不同**：

```
m=0: [1.0, 0.5, -0.5, 0.2,  0.8, -0.3,  0.6, -0.1]   ← 原样
m=1: [-0.1329, 0.5275, -0.5032, 0.2, 1.2737, -0.2486, 0.5973, -0.1]
m=2: [-1.1436, 0.5496, -0.5063, 0.2, 0.5764, -0.1947, 0.5947, -0.1]
```

**转的不是"token 本身"，是"token 出现在这个位置上的这一次"。**

KV cache 里存的 `k` **已经是转好的**，生成时不会重转历史 token。

### Q8：`freq` 是频率吗？

**不是，是角速度**（弧度/token）。真正的频率是 $f_i/2\pi$。HF 里叫 `inv_freq`。

### Q9：为什么只给 q、k 转，不给 v 转？

因为 attention 分数是 $q^\top k$ 算出来的——位置只需要影响"谁该关注谁"。`v` 是被加权求和的内容，转它反而破坏语义。

### Q10：为什么 cos/sin 要复制一份拼成 96 宽？

因为第 $i$ 组和第 $i+48$ 组用**同一个角度**，而 `x` 是 96 宽。复制让一次元素乘法就能覆盖全部 48 组，不用循环。

---

## 9. 符号总表

| 符号 | 代码名 | 单位 | 含义 | MiniMind 取值 |
|---|---|---|---|---|
| $d$ | `dim` / `head_dim` | — | 每个头的向量长度 | 96 |
| $i$ | `arange(dim//2)` | — | **组号**，0~47 | 0~47 |
| $b$ | `rope_base` / `rope_theta` | — | RoPE 底数，越大转越慢 | 1e6 |
| $L$ | `origin_max` | token | 训练时最长序列 | 2048 |
| $M$ | `end` | token | 目标长度 | 32768 |
| $s$ | `factor` | 倍 | 压缩倍数，**必须** $=M/L$ | 16 |
| $\beta_{\text{fast}}$ | `beta_fast` | 圈 | 快维度门槛 | 32 |
| $\beta_{\text{slow}}$ | `beta_slow` | 圈 | 可信门槛 | 1 |
| $f_i$ | `freq` | **rad/token** | **角速度** | 如 $f_{14}=0.01778$ |
| $\lambda_i$ | — | **token** | 转一圈要几个 token | 如 $\lambda_{14}=353.3$ |
| $r_i$ | — | 圈 | 训练内转几圈 | 如 $r_{14}=5.80$ |
| $G(q)$ | `inv_dim` | — | 给圈数求组号（反解）| $G(32)=8.06$ |
| low | `low` | — | 左分界线 | 8 |
| high | `high` | — | 右分界线 | 21 |
| $\rho_i$ | `ramp` | 0~1 | 斜坡 | 如 $\rho_{14}=0.4615$ |
| $c_i$ | `xishu` | — | **YaRN 系数** | 如 $c_{14}=0.5673$ |
| $a$ | `attn_factor` | — | 温度系数 | 1.0 |
| $m$ | `start_pos + ...` | — | token 位置 | — |
| $\theta_{m,i}$ | `freqs` | rad | 位置 $m$ 第 $i$ 组的角度 | — |
| $C,S$ | `cos_freqs`/`sin_freqs` | — | `[M, d]` 查表 | `[32768, 96]` |

---

## 10. 自检清单

读完这份笔记，应该能回答这些问题：

- [ ] 96 维为什么分成 48 组？第 $i$ 组配的是哪两个数？
- [ ] 旋转为什么不改变向量长度？怎么用这一点验证实现？
- [ ] 为什么 $q_m^\top k_n$ 只依赖 $m-n$？验证时需要什么前提？
- [ ] `rope_theta` 从 1e4 换成 1e6，转速变快还是变慢？
- [ ] 波长、圈数、角速度三者的关系？
- [ ] `low=8` 和 `high=21` 是怎么从"32 圈"和"1 圈"数出来的？
- [ ] 为什么 `inv_dim` 右边没有 $i$？
- [ ] 为什么一个用 `floor` 一个用 `ceil`？
- [ ] `ramp` 和 `xishu` 分别是什么操作？（提示：都是插值）
- [ ] 为什么必须有过渡带，不能一刀切？
- [ ] `factor` 为什么必须等于 `M/L`？
- [ ] YaRN 是开关还是静态替换？为什么不能做成开关？
- [ ] 为什么只给 q、k 加 RoPE，v 不加？
- [ ] 位置 0 的 token 发生了什么？
- [ ] notebook 里的 4 个坑分别是什么？怎么修？

---

## 跑一遍验证

```python
import torch, math

# ---------- RoPE 基本正确性 ----------
def rotate_half(x):
    return torch.cat([-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]], dim=-1)

def apply_rope(q, k, cos, sin, unsqueeze_dim=1):
    q = (q * cos.unsqueeze(unsqueeze_dim) + rotate_half(q) * sin.unsqueeze(unsqueeze_dim)).to(q.dtype)
    k = (k * cos.unsqueeze(unsqueeze_dim) + rotate_half(k) * sin.unsqueeze(unsqueeze_dim)).to(k.dtype)
    return q, k

dim, base, end = 8, 1e4, 10
freq = 1 / torch.pow(base, torch.arange(0, dim, 2).float() / dim)
freqs = torch.outer(torch.arange(end), freq).float()
cos_f = torch.cat([torch.cos(freqs)]*2, -1)
sin_f = torch.cat([torch.sin(freqs)]*2, -1)

# 同一个向量放在不同位置（必须是同一份内容！）
qv, kv = torch.randn(dim), torch.randn(dim)
qs = qv.view(1,1,1,dim).repeat(1, end, 1, 1)
ks = kv.view(1,1,1,dim).repeat(1, end, 1, 1)
qr, kr = apply_rope(qs, ks, cos_f, sin_f)
M = qr[0,:,0,:] @ kr[0,:,0,:].T

for d in range(5):
    vals = [round(M[i, i-d].item(), 4) for i in range(d, end)]
    assert len(set(vals)) == 1, f"m-n={d} 不是常数: {vals}"
print("✅ 相对位置性质成立")

# 长度不变
assert torch.allclose(qs[0,1,0].norm(), qr[0,1,0].norm(), atol=1e-6)
print("✅ 旋转保持长度")

# ---------- YaRN 分界线 ----------
d96, b, L = 96, 1e6, 2048
circles = L / (2 * math.pi * (b ** (2 * torch.arange(d96//2) / d96)))
low  = (circles >= 32).nonzero().max().item()
high = (circles >= 1 ).nonzero().max().item() + 1
print(f"✅ low={low}, high={high}")

# 和 log 公式对照
inv_dim = lambda q: d96 * math.log(L / (2*math.pi*q)) / (2*math.log(b))
assert low == math.floor(inv_dim(32))
assert high == math.ceil(inv_dim(1))
print("✅ 无公式版与 log 公式结果一致")
```

---

**最后一句**：

> **RoPE = 48 个独立的小旋转，转向由"位置 × 角速度"决定。**
> **YaRN = 出厂前把那 48 个角速度各乘一个 0~1 的系数，让慢的组在最长文本上也不越界。**
