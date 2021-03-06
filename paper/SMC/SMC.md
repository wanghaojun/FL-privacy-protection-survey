## SMC

#### Privacy preserving regression modelling via distributed computation

Ashish P Sanil, Alan F Karr, Xiaodong Lin, and Jerome P Reiter. Privacy preserving regression modelling via distributed computation. In Proceedings of the tenth ACM SIGKDD international conference on Knowledge discovery and data mining, pages 677–682. ACM, 2004.

https://dl.acm.org/doi/10.1145/1014052.1014139

##### 摘要

数据所有者不愿与拥有相关数据库的其他人共享其可能的机密或专有数据，这是进行互利数据挖掘分析的严重障碍。 我们解决了垂直分区数据的情况-多个数据所有者/代理商各自拥有每个数据记录的一些属性。 我们关注的情况是，代理商希望对完整的记录进行线性回归分析，而又不透露其自身属性的值。 本文介绍了一种算法，使此类机构能够计算全局回归方程式的确切回归系数，并在保护其数据机密性的同时执行一些基本的拟合优度诊断。 在隐私场景之外的更一般的设置中，此算法也可以看作是用于回归分析的分布式计算的方法。

##### 内容

从多个分布式数据库上进行统计分析（线性回归），同时保证数据的机密性。

考虑两种情况，纵向分割的数据和横向分割的数据（联邦学习中的横向和纵向数据划分），这篇文章只考虑了如何对垂直数据分区进行安全的线性回归。

我们认为，一组寻求对其数据进行综合分析的政府机构可以很好地反映我们所处理的半诚实数据共享方案。因此，即使在某些情况下他们可能是公司或其他数据持有人，我们也将其称为参与者“代理商”。在第二部分中，我们概述了隐私保护回归问题。在第3节中，对Powell的数值最小化方法和安全求和协议进行了简要说明，它们共同构成了我们程序的组成部分。第4节描述了主要算法，第5节讨论了使用该程序揭示的内容以及各机构共同学习的内容（包括进行回归分析的可能途径）。最后，我们在第6节作总结。

##### 场景

有$k$个参与方，每个参与方拥有数目为$d_i$的属性，样本数为$n$，目的是训练模型

<img src="http://img.wanghaojun.cn//img/20201222144946.png" style="zoom: 80%;" />

使用平方误差和来计算$\beta$的最小二乘法估计（最大似然估计）
$$
E(\beta)=(\mathbf{y}-X \beta)^{T}(\mathbf{y}-X \beta)
\tag{6}
$$
上式是关于$\beta$的二次平方公式，其最小值计算为：
$$
\hat{\beta}=\left(X^{T} X\right)^{-1} X^{T} y
\tag{7}
$$
参与方还被假定为不愿意共享将其数据与其他代理商的数据相关联的摘要统计信息，例如属性之间的相关性。

Remark1：与其他数据共享协议一样，我们要求一个机构在启动和协调流程中担当主要角色。这纯粹是管理角色，并不意味着任何信息优势或劣势。我们将假定代理商1为指定负责人。

Remark2：数据库需要有一个公共主键，使代理机构能够按相同顺序正确对齐记录（可能在代理机构1的指导下）。

Remark3：属性集合不重合，诸如（1）之类的回归模型通常会包含一个常数或“截距”项。这等效于X中一个为1的列。在不失一般性的前提下，我们将假设其中一个属性为$$
\mathbf{x}^{T}=(1,1, \ldots, 1)$$，并且它是机构1所“拥有”的。

##### 技术

优化算法：powell方法

安全求和：$K$个客户端，首先选择一个大数$m$，所有的$v_j$都位于$[0,m)$之间，客户端1选择一个随机数$R$位于$[0,m)$之前，客户端1计算
$$
s_{1}=\left(R+v_{1}\right) \bmod m
$$
发给客户端2，然后客户端2继续加上自己的$v_2$，知道$s_k$，然后发送给客户端1，客户端1减去$R$，然后得到和$v$，并发送给所有客户端。（不能防止共谋问题）

我们的算法本质上是鲍威尔（Powell）算法，其实现方式是，每个代理Aj根据其拥有的数据属性和所有代理共有的一个n维向量，使用安全求和来更新其自身的β分量和其自身的搜索方向分量。



#### SMSS: Secure Member Selection Strategy in Federated Learning

##### 摘要

数据安全性和用户隐私问题已成为重要领域。由于联邦学习（FL）可以解决数据安全和隐私问题带来的问题，因此它开始被应用于许多不同的应用机器学习任务中。但是，FL不会验证来自系统中不同方面的数据的质量。因此，具有较少公共实体的劣质数据集可以与其他实体一起训练。这可能会导致大量的计算资源浪费，以及来自作为联邦成员的恶意客户端对FL模型的攻击。为了解决这个问题，本文提出了一种安全的成员选择策略（SMSS），可以在训练之前评估成员的数据质量。使用SMSS，数据集只会共享更多的公共实体，而不能选择某个阈值进行学习，而具有较少公共对象的恶意客户端无法获取有关模型的任何信息。本文实现了SMSS，并通过几个广泛的实验来评估其性能。实验结果表明，SMS是安全，有效和高效的。

##### 介绍

问题：FL不验证联邦成员或合作者的数据质量，这将导致严重的安全风险。纵向联邦 学习三个步骤：加密实体对齐；加密模型训练和模型更新。FL仅在“加密实体一致性”部分中确认联邦成员之间的公共实体，而未提供成员验证。更糟糕的是，恶意成员可以通过加密的实体对齐方式获取联邦成员的实体信息，并伪造这些实体的标签以降低模型性能。现有的FL方法（例如安全的多方计算，差分隐私（DP），同态加密和秘密共享）永远不会将数据排除在联邦成员之外，以保证安全性和隐私性。

为了解决此问题，我们提出了一种安全成员选择策略（SMSS），联邦成员可以相互验证并仅使用公共渠道来确认其公共实体ID。当且仅当成员通过验证测试时，他们才能使用组合密钥加入FL，其他成员将被排除。通过这种方式。  SMSS实现安全的成员选择，而不会造成信息泄漏。如图1所示，如果它们中的任何两个共享足够大的实体ID（不需要相同的ID），并且大于某个阈值u，则它们将通过验证。否则，数据集将较少的公共实体将从FL处理中排除（请参见虚线框）。

使用一种Jaccard系数阈值（JCT）的方式$J_{i j}=\frac{\left|P_{i} \cap {P}_{j}\right|}{\left|P_{i} \cup P_{j}\right|}$来验证是否是合法参与者。适用于横向、纵向和迁移。实施基于JCT的安全FL的主要挑战是如何以隐式方式计算设置的Jaccard系数？  SMSS集成了专用集交叉点（private set intersection，PSI）和Shamir的秘密共享，以实现基于实体的验证。

PSI无需事先了解即可在公共渠道上实现对称密钥交付。 Shamir的秘密共享方案可确保包含超过u个公共元素的数据集Pi可以获得密钥Si，而少于u个公共元素的数据集Pi无法恢复Si。  每个联邦成员贡献一个密钥Si，并利用总和$\mathbb{S}=\sum S_{i}$构造一个秘密通道，这样，恶意客户端Eve无法参与通信，因为她无法获取其他人的密钥来计算S。训练信息由$\mathbb{S}$和联邦学习框架进行加密，不需要找到可信的第三方作为协作者。

贡献：

- 我们建议在FL框架中使用SMSS，这样可以避免为密钥预分发建立秘密渠道。该解决方案可以轻松扩展到其他基于相似性的关键协议工作。
- SMSS结合了PSI和Shamir的方案，是零信息泄漏，可以或至少可以部分防御试错攻击，共谋攻击和重播攻击。
- SMSS是有效的。我们的实验表明，时间开销通常不超过0.2 s，当恶意客户端加入协同训练时，可以弥补74％的准确性损失。

##### 协议

在本文中，我们将重点放在对原始协议不诚实的恶意对手上。因此，我们将对手模式定义如下。他们可以执行以下操作：1）侦听公共频道上传递的所有消息，但不对其进行修改；  2）感知信道环境，注入新流量并重播数据包；和3）完全了解所提出的方法和算法。

SMSS主要有三个过程：联邦成员验证，加密实体对齐，秘密通道建立。即使每个参与者都是窃听者，纵向联邦学习可以使用SMSS安全的完成任务。

##### 分析

- 不同FL类别的阈值配置；
- SMSS的效率；
- RANSAC迭代轮的分布分析；
- 信息泄漏问题；
- 对于共谋攻击的抵抗；

##### 评价

SMSS的时间开销：

- 总体时间开销

  平均多花费0.2s

- Shamir方案的时间开销

- RANSAC算法的时间开销

SMSS对FL模型的影响：

- SMSS的准确性评估

  原始0.97，恶意参与者0.9，SMSS0.95，弥补71.4%

- SMSS的安全性评估

  恶意参与者在每个回合中几乎都无法获得任何实体信息。

##### 总结

在大数据和人工智能领域，我们面临着越来越多的隐私和安全挑战。提议将FL用作数据安全性的关键角色。我们提出SMSS，一种混合的安全成员选择解决方案。  SMSS利用Shamir的方案共享对称密钥，这可以避免恶意客户端或不正确的数据集在不进行数据交换的情况下对模型进行训练。另外，SMS利用PSI解决了实时机密块分发问题，并引入了RANSAC算法来从并非所有正确的机密块中恢复正确的密钥。我们进行了严格的分析，以显示我们协议的可行性和安全性。实验结果证明了SMSS的高效性和鲁棒性。我们相信SMSS的想法可以在更多应用场景中扩展。

#### Secure Byzantine-Robust Machine Learning

##### 摘要

越来越多的机器学习系统被部署到边缘服务器和设备（例如移动电话）并以协作的方式进行培训。这样的分布式/联合/分散式培训引起了对程序的鲁棒性，隐私性和安全性的许多关注。尽管在解决鲁棒性，隐私或安全性方面已经进行了广泛的工作，但很少研究它们的组合。在本文中，我们提出了一种安全的两服务器协议，该协议可提供输入隐私和拜占庭的鲁棒性。另外，该协议具有通信效率高，容错能力强并且可以享受本地差分隐私。

##### 介绍

恶意参与者可以故意向错误的数据提供数据，从而破坏培训系统，这被称为数据中毒。在本文中，我们解决了这个问题，并提出了一种新颖的分布式培训框架，该框架既提供隐私又具有鲁棒性。

在本文中，我们借助两个无冲突的诚实但好奇的服务器，提出了一个安全的聚合框架。该框架还可以容忍服务器工人的串通行为。此外，我们将健壮性和隐私性结合在一起，但仅以泄漏工人相似性信息为代价，这对于高维神经网络而言是微不足道的。请注意，我们的重点不是开发针对最新攻击的新防御措施，例如[3,40]。取而代之的是，我们专注于使当前和未来的基于距离的鲁棒聚合规则（例如，Krum  [26]，RFA [30]）与安全聚合兼容。

贡献：提出了一个新的分布式训练框架，具有以下特点：

隐私保护，我们的方法可确保每个用户的输入数据不受任何其他用户以及我们诚实但好奇的服务器的侵害。

拜占庭鲁棒性，我们的方法提供了拜占庭的鲁棒性，并允许合并现有的鲁棒聚合规则。结果是准确的，即与非私有鲁棒方法相同。

容错和易于使用：我们的方法本身就支持工人退出或重新加入培训过程。对于用户而言，它也易于实现和理解。

高效且可扩展：与非私有方法相比，我们方法的计算和通信开销可忽略不计（不到2倍）。就成本而言，包括设置和通信在内的可伸缩性在工人人数方面是线性的。

##### 协议

安全聚合协议：两个服务器的模型

不考虑鲁棒性的安全聚合：

![](http://img.wanghaojun.cn//img/20210113192851.png)

考虑拜占庭鲁棒性的安全聚合：

![](http://img.wanghaojun.cn//img/20210114095558.png)

![](http://img.wanghaojun.cn//img/20210114095633.png)

总结

在本文中，我们提出了一种新颖的安全且拜占庭鲁棒的聚合框架。据我们所知，这是共同解决这两个关键特性的第一个工作。语言算法简单且具有容错能力，并且可以随着工人数量的增加而很好地扩展。此外，我们的框架适用于任何现有的基于距离的鲁棒规则。此外，我们算法的通信开销大致由2来限制，并且如算法3所示，计算开销很小，甚至可以在训练之前进行计算。

#### HybridAlpha: An Efficient Approach for Privacy-Preserving Federated Learning

##### 摘要

联合学习已成为一种有希望的协作和隐私保护学习方法。联邦学习过程中的参与者通过交换模型参数而不是他们可能希望保持私有的实际训练数据来协作训练模型。但是，参数交互和生成的模型仍可能会泄露有关所用训练数据的信息。为了解决这些隐私问题，已经提出了几种基于差分隐私和安全多方计算（SMC）的方法。它们通常会导致大量的通信开销和缓慢的培训时间。在本文中，我们提出了HybridAlpha，一种使用基于函数加密的SMC协议保护隐私的联合学习的方法。该协议简单，有效且对退出参与者具有弹性。我们评估有关使用联合学习过程在MNIST数据集上训练CNN的训练时间和数据量交换的方法。对现有基于密码的SMC解决方案进行的评估表明，HybridAlpha可以将训练时间平均减少68％，将数据传输量平均减少92％，同时提供与现有解决方案相同的模型性能和隐私保证。

##### 贡献

我们提出了HybridAlpha，这是一种高效的隐私保护FL方法，该方法采用差分隐私机制并根据多输入函数加密方案定义了SMC协议。我们对这种方案进行了调整，并加入了其他条款，以减少好奇的聚合者和串通参与者推断私人信息的风险。

我们在理论上和实验上都实现了函数加密方案，并将其与常见的传统加密方案（例如加法同态加密及其变体，通常用于SMC）进行比较。我们的基准测试结果将指导未来在为FL选择合适的SMC时采用这些密码系统。

我们描述了混合Alpha方法的实现，并将其应用于卷积神经网络。在MNIST数据集上的实验结果表明，我们的混合Alpha框架在训练时间和通信成本上都有效率上的提高，同时提供了与其他方法相同的模型性能和隐私保证。

同时，我们演示了针对动态参与者组问题的解决方案，这表明我们提出的框架对于参与者的退出或增加是强大的。我们还将在可信任的TPA，诚实但好奇的聚合者和部分不诚实的参与者中，在我们定义的威胁模型下分析HybridAlpha框架的安全性和隐私保证。

据我们所知，这是第一个用于保护隐私的联合学习的方法，该方法演示了如何利用函数加密来防止某些幼稚地应用此密码系统可能发生的推理攻击。

##### 背景知识

函数加密（Functional Encryption）

HybridAlpha依赖于函数加密（FE），它是允许各方加密其数据的公共密钥密码系统，同时，外部实体可以在密文上计算特定的函数，而无需从基础明文数据中学习任何其他信息。

Multi-Input Functional Encryption(MIFF)，本文使用的MIFF依赖于DDH假设，主要包含了五个算法：Setup，PKDistribute， SKGenerate，Encrypt，Decrypt，三个角色：一个第三方认证（TPA），参与方和聚合者。算法设置、公钥分发、私钥生成是由TPA运行的。每个参与者运行加密算法以加密其模型参数，聚合器运行解密算法以获取加密的模型参数的平均和。每个客户端都有不同的公钥进行加密。

假设内积函数如下图所示：
$$
f\left(\left(\mathbf{x}_{1}, \mathbf{x}_{2}, \ldots, \mathbf{x}_{n}\right), \mathbf{y}\right)=\sum_{i=1}^{n} \sum_{j=1}^{\eta_{i}}\left(x_{i j} y_{\sum_{k=1}^{i-1} \eta_{k}+j}\right) s . t \cdot|\mathbf{y}|=\sum_{i=1}^{n} \eta_{i}
$$
$n$表示输出源的数量，$\eta_i$表示每个输入向量$\mathbf{x}_{i}$的长度，dimension $(\mathbf{y})=\sum_{i=1}^{n}$ dimension $\left(\mathbf{x}_{i}\right)$，MIFF过程如下：

<img src="http://img.wanghaojun.cn//other/20210118103229.png" style="zoom:150%;" />

<img src="http://img.wanghaojun.cn//other/20210118103251.png" style="zoom:150%;" />

在联邦学习环境下，聚合者拥有$\mathbf{y}$，每个$\eta_i$的值是1.

##### 框架

在本节中，我们介绍了用于保护隐私的联合学习的HybridAlpha框架的特定结构。我们的框架可防止来自好奇的聚合者的推理攻击，并限制共谋参与者的推理能力，这将在威胁模型中稍后详细介绍。

图1展示了混合Alpha的概述。参与者希望在不与系统中任何其他实体共享其本地数据的情况下，共同学习机器学习模型。他们同意仅与聚合器共享模型更新。该实体负责接收来自多个参与者的模型更新，以构建通用的机器学习模型。

![](http://img.wanghaojun.cn//other/20210114192618.png)

参与者希望保护自己的数据免受FL过程中以及最终模型的任何推理攻击。为此，他们加入了具有第三方授权机构（Third Party Authority，TPA）的混合Alpha。该实体提供了一个密钥管理服务，该服务可启动密码系统并向所有各方提供函数加密密钥。为了防止潜在的信息泄漏，HybridAlpha还包括一个推理阻止模块，该模块可限制提供哪种类型的函数加密密钥。该模块旨在确保好奇的聚合者无法获得解密密钥，并限制潜在的串通攻击。我们将在第3.2.2节中详细介绍此模块。

3.1 威胁模型

诚实但好奇的聚合器：我们假设聚合器正确地遵循了算法和协议，但可能会尝试检查参与者在过程中发送的模型更新，以学习私有信息。这是一个普遍的假设

好奇和串通的参与者：我们认为参与者可能会合谋通过检查与聚合器或最终模型交换的消息来尝试从其他参与者获取私人信息

受信任的TPA：该实体是一个独立的机构，受到参与者和聚合器的广泛信任。在实际情况下，经济的不同部门已经拥有可以担当这种角色的实体。例如，在银行业中，中央银行通常扮演着完全可信赖的角色，而在其他部门中，服务或咨询公司等第三家公司可以体现TPA。我们还注意到，在采用TPA作为底层基础结构的现有密码系统中，假设这种受信任和独立的代理机构是一个普遍的假设。  TPA负责掌握主私钥和公钥。也可以信任TPA来执行公共密钥分发和功能派生的秘密密钥生成。同样，推理预防模块是完全受信任的。

我们假定在所有通信中都使用安全通道，因此可以防止中间人攻击和琐碎的监听攻击。我们还假设有一个安全的密钥提供程序，例如Diffie-Hellman，以保护密钥机密性。最后，旨在创建拒绝服务攻击或注入恶意模型更新的攻击不在本文讨论范围之内。

基于上述威胁模型，我们提出的隐私保护框架可以确保（i）半诚实的聚合者无法学习其他信息，只输出使用差分隐私机制的预定输出；以及（ii）恶意勾结的参与者无法学习其他诚实参与者的参数。第5节介绍了特定的安全性和隐私分析。

3.2.1 非恶意设置

算法1：

![](http://img.wanghaojun.cn//other/20210114192642.png)

3.2.2 模型的推理预防

一个好奇的聚合器可能通过以下方式来获取特定用户的模型：

- 构造权重向量$\mathbf{w}_p$时，一个为1，其余为0
- 缩小加权向量以包括单一的参与者
- 与恶意的参与者合谋，聚合中仅包含目标参与者，因此可以轻松地重构目标参与者的模型更新

为了预防此类推理攻击，我们提出了一个额外的组件叫做推理预测模块来与TPA结合。该模块拦截并检查给定加权矢量对私钥的请求，以防止好奇的聚合者获得允许推理内积的密钥。

![](http://img.wanghaojun.cn//other/20210118154454.png)

HybridAlpha适用于使用SGD进行训练的ML算法，比如说SVMs，logistic回归，线性回归，Lasso和神经网络。还可以通过考虑将发送给聚合器的计数视为向量来训练其他模型，例如决策树和随机森林，这些模型需要每个参与者的聚合计数。

#### ARIANN: Low-Interaction Privacy-Preserving Deep Learning via Function Secret Sharing