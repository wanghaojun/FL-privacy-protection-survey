## 论文整理-wanghaojun

参考文献

《Federated Learning: Challenges, Methods, and Future Directions》

《A Survey on Federated Learning Systems: Vision, Hype and Reality for Data Privacy and Protection》

## HE

#### Federated Forest

隐私保护机制不太确定 大概是同态 没有细看

Yang Liu, Yingting Liu, Zhijie Liu, Junbo Zhang, Chuishi Meng, and Yu Zheng. Federated forest. arXiv preprint arXiv:1905.10053, 2019.

https://arxiv.org/abs/1905.10053

大多数真实世界的数据分散在不同的公司或政府组织中，并且无法根据数据隐私和相关法规（例如欧盟的《通用数据保护法规》（GDPR）和中国的《网络安全法》）轻松整合。这样的数据孤岛状况以及数据隐私和安全性是人工智能应用的两个主要挑战。在本文中，我们解决了这些挑战，并提出了一种称为Federated Forest的隐私保护机器学习模型，该模型是传统随机森林方法的无损学习模型，即达到与非隐私保护相同的准确性水平方法。在此基础上，我们开发了一个安全的跨区域机器学习系统，该系统允许在具有相同用户样本但属性集不同的不同区域的客户上共同训练学习过程，从而无需交换原始数据即可处理每个区域中存储的数据数据。还提出了一种新颖的预测算法，可以大大减少通信开销。在真实世界和UCI数据集上进行的实验表明，联合林的性能与非联合版本一样准确。我们提出的系统的效率和鲁棒性已得到验证。总体而言，我们的模型适用于实际任务，具有实用性，可扩展性和可扩展性。

#### SecureBoost: A Lossless Federated Learning Framework

Kewei Cheng, Tao Fan, Yilun Jin, Yang Liu, Tianjian Chen, and Qiang Yang. Secureboost: A lossless federated learning framework. arXiv preprint arXiv:1901.08755, 2019.

https://arxiv.org/abs/1901.08755

保护用户隐私是机器学习中的一个重要问题，这一点在2018年5月欧盟（EU）推出的《通用数据保护条例》（GDPR）中得到了证明。GDPR旨在为用户提供对他们的更多控制权个人数据，这促使我们探索具有数据共享功能而又不侵犯用户隐私的机器学习框架。为了实现此目标，在本文中，我们提出了一种在联邦学习环境中称为SecureBoost的新型无损隐私保护树提升系统。该联合学习系统允许使用部分相同的用户样本但具有不同功能集（对应于垂直划分的虚拟数据集）在多方上共同进行学习过程。 SecureBoost的一个优点是，它提供与非隐私保护方法相同的准确性，同时不泄露每个私有数据提供者的信息。从理论上讲，我们证明SecureBoost框架与将数据置于一处的其他非联合梯度树增强算法一样准确。此外，连同安全性证明一起，我们讨论了使协议完全安全所需的条件。



#### Privacy-preserving ridge regression on distributed data

Yi-Ruei Chen, Amir Rezapour, and Wen-Guey Tzeng. Privacy-preserving ridge regression on distributed data. Information Sciences, 451:34–49, 2018

https://www.sciencedirect.com/science/article/abs/pii/S0020025518302500

岭回归是一种统计方法，用于建模因变量和某些解释值之间的线性关系。它是构建模块，在许多学习算法（例如推荐系统）中起着重要作用。但是，在许多应用程序中，例如电子医疗，解释值包含不愿共享这些信息的不同患者拥有的私人信息，除非保证了数据隐私。在本文中，我们提出了一种在高维数据上进行隐私保护脊回归（PPRR）的协议。在我们的协议中，每个用户以加密的形式将其数据提交给评估者，评估者在不了解其内容的情况下计算所有用户数据的线性模型。核心加密方法具有同态属性，以使评估人员能够对加密数据执行岭回归。我们实现了我们的协议，并证明了它适用于处理数百万用户之间分布的高维数据。在计算和通信成本方面，我们还将我们的协议与最新解决方案进行比较。结果表明，我们的协议优于基于安全多方计算，乱码，完全同态加密，秘密共享和混合方法的大多数现有方法。

#### Private federated learning on vertically partitioned data via entity resolution and additively homomorphic encryption

Stephen Hardy, Wilko Henecka, Hamish Ivey-Law, Richard Nock, Giorgio Patrini, Guillaume Smith, and Brian Thorne. Private federated learning on vertically partitioned data via entity resolution and additively homomorphic encryption. arXiv preprint arXiv:1711.10677, 2017.

https://arxiv.org/abs/1711.10677

考虑两个数据提供者，每个数据提供者维护有关公共实体的不同功能集的私有记录。他们旨在联合学习联合模型中的线性模型，即数据是本地的，共享模型是从本地计算的更新中训练出来的。与大多数有关分布式学习的工作相反，在这种情况下（i）数据是按垂直方向拆分的，即按功能划分的；（ii）只有一个数据提供者知道目标变量，并且（iii）实体未跨数据提供者链接。因此，对于私人学习的挑战，我们在实体解析中增加了错误可能带来的负面影响。我们的贡献是双重的。首先，我们在两个阶段中描述了一个由三方组成的端到端解决方案：-隐私保护实体解析和对通过加法同态方案加密的消息进行的联合逻辑回归--防止诚实但又好奇的对手。该系统允许学习而无需明文暴露数据或共享数据提供者共有的实体。我们的实施与将所有数据集中到一处的天真的非私有解决方案一样准确，并且可以解决具有数以百计的功能的数百万个实体的问题。其次，我们首先提供关于实体解析错误对学习的影响的正式分析，并提供关于如何影响最佳分类器，经验损失，边际和泛化能力的结果。我们的结果为联合学习提供了明确而有力的支持：在对实体解析错误的数量和严重程度进行合理假设的情况下，在每个对等方的数据彼此之间都有显着提升的情况下进行联合学习可能非常有益。

#### BatchCrypt: Efficient Homomorphic Encryption for Cross-Silo Federated Learning

Zhang C, Li S, Xia J, et al. Batchcrypt: Efficient homomorphic encryption for cross-silo federated learning[C]//2020 {USENIX} Annual Technical Conference ({USENIX}{ATC} 20). 2020: 493-506.

https://www.usenix.org/conference/atc20/presentation/zhang-chengliang

跨孤岛联合学习（FL）使组织（例如金融或医疗）能够通过汇总来自每个客户端的本地梯度更新来协作训练机器学习模型，而无需共享对隐私敏感的数据。为了确保在聚合期间不会显示任何更新，工业FL框架允许客户端使用加性同态加密（HE）来掩盖本地梯度更新。但是，这导致计算和通信成本很高。在我们的表征中，HE操作控制着训练时间，同时使数据传输量增加了两个数量级。在本文中，我们介绍了BatchCrypt，这是一种用于跨孤岛FL的系统解决方案，可大大减少HE造成的加密和通信开销。而不是完全精确地加密各个梯度，我们将一批量化的梯度编码为一个长整数，并一次性对其进行加密。为了允许对编码批次的密文进行梯度方式的聚合，我们开发了新的量化和编码方案以及一种新颖的梯度裁剪技术。我们将BatchCrypt实施为FA  TE（工业跨仓库FL框架）中的插件模块。在地理位置分布式数据中心中使用EC2客户端进行的评估表明，BatchCrypt可以实现23×-93×的训练速度，同时将通信开销减少了66×-101×。由于量化误差导致的精度损失小于1％。

#### A Practical Privacy-preserving Method in Federated Deep Learning

尽管联邦学习通过交换模型更新而不是原始数据来提高训练数据的隐私性，但许多研究结果表明，共享模型更新可能仍然存在风险。为了减轻这个问题，许多隐私保护技术已被引入联合学习中。然而，考虑到联合学习中的深度学习模型，所得方案要么不能很好地实现非线性激活函数，要么不能保持与原始训练相同的模型精度，否则将承受无法承受的成本。在本文中，我们为联合深度学习定制了一种实用的隐私保护方法，该方法通用且适用于大多数最新模型，例如ResNet和DenseNet。特别地，该方法可以在加密域上很好地支持非线性激活功能，因此支持半信任客户端在加密模型迭代上有效地本地训练深度神经网络（即，保护服务器端模型的隐私）。同时，可以将其与秘密共享技术相结合，以进一步确保半受信任的服务器无法获得每个客户端的本地梯度（即，为客户端保护训练数据的隐私）。详细的安全性分析和大量的实验表明，该方法可以在不牺牲模型准确性和引入过多额外成本的情况下实现隐私保护。

#### PrivFL: Practical Privacy-preserving Federated Regressions on High-dimensional Data over Mobile Networks

联合学习（FL）使大量用户可以共同学习由集中式服务器协调的共享机器学习（ML）模型，该模型中的数据分布在多个设备上。这种方法使服务器或用户可以使用梯度下降来训练和学习ML模型，同时将所有训练数据保留在用户的设备上。我们考虑通过移动网络训练ML模型，而用户掉队是一种普遍现象。尽管联合学习旨在减少数据隐私风险，但ML模型的隐私并未受到太多关注。在这项工作中，我们介绍了PrivFL，这是一个隐私保护系统，用于在联合设置中训练（预测）线性和逻辑回归模型以及遗忘的预测，同时保证数据和模型的隐私性，并确保对网络中退出的用户的鲁棒性。我们设计了两个隐私保护协议，用于基于加性同态加密（HE）方案和聚合协议来训练线性和逻辑回归模型。利用联合学习的训练算法，我们训练协议的核心是对活着的用户数据进行安全的多方全局梯度计算。我们分析针对半诚实的对手的培训协议的安全性。只要聚合协议在聚合隐私博弈下是安全的，并且附加HE方案在语义上是安全的，PrivFL就可以保证用户针对服务器的数据隐私，以及服务器针对用户的回归模型的隐私。我们演示了PrivFL在真实数据集上的性能，并展示了其在联合学习系统中的适用性。

## SMC

#### Practical secure aggregation for privacy-preserving machine learning

#### Privacy preserving regression modelling via distributed computation

Ashish P Sanil, Alan F Karr, Xiaodong Lin, and Jerome P Reiter. Privacy preserving regression modelling via distributed computation. In Proceedings of the tenth ACM SIGKDD international conference on Knowledge discovery and data mining, pages 677–682. ACM, 2004.

https://dl.acm.org/doi/10.1145/1014052.1014139

数据所有者不愿与拥有相关数据库的其他人共享其可能的机密或专有数据，这是进行互利数据挖掘分析的严重障碍。 我们解决了垂直分区数据的情况-多个数据所有者/代理商各自拥有每个数据记录的一些属性。 我们关注的情况是，代理商希望对完整的记录进行线性回归分析，而又不透露其自身属性的值。 本文介绍了一种算法，使此类机构能够计算全局回归方程式的确切回归系数，并在保护其数据机密性的同时执行一些基本的拟合优度诊断。 在隐私场景之外的更一般的设置中，此算法也可以看作是用于回归分析的分布式计算的方法。

ps：看摘要，有点像跨组织的纵向联邦学习，所以尽管2004年的 我也放上了

从多个分布式数据库上进行统计分析（线性回归），同时保证数据的机密性。

考虑两种情况，纵向分割的数据和横向分割的数据（联邦学习中的横向和纵向数据划分），这篇文章只考虑了如何对垂直数据分区进行安全的线性回归。

我们认为，一组寻求对其数据进行综合分析的政府机构可以很好地反映我们所处理的半诚实数据共享方案。因此，即使在某些情况下他们可能是公司或其他数据持有人，我们也将其称为参与者“代理商”。在第二部分中，我们概述了隐私保护回归问题。在第3节中，对Powell的数值最小化方法和安全求和协议进行了简要说明，它们共同构成了我们程序的组成部分。第4节描述了主要算法，第5节讨论了使用该程序揭示的内容以及各机构共同学习的内容（包括进行回归分析的可能途径）。最后，我们在第6节作总结。

#### SMSS: Secure Member Selection Strategy in Federated Learning

https://ieeexplore.ieee.org/document/9136885

数据安全和用户隐私问题已成为重要领域。由于联邦学习（FL）可以解决数据安全和隐私问题带来的问题，因此它开始被应用于许多不同的应用机器学习任务中。但是，FL不会验证来自系统中不同方面的数据的质量。因此，具有较少公共实体的低质量数据集可以与其他实体一起训练。这可能导致大量的计算资源浪费，以及来自作为联邦成员的恶意客户端对FL模型的攻击。为了解决这个问题，本文提出了一种安全的成员选择策略（SMSS），可以在训练之前评估成员的数据质量。使用SMSS，只有数据集共享的公共实体多于某个阈值才能选择用于学习，而公共对象较少的恶意客户端无法获取有关该模型的任何信息。本文实现了SMSS，并通过多个广泛的经验来评估其性能。实验结果表明，SMSS是安全，有效和有效的。

ps：一篇用于联邦学习中选择参与成员的论文，混合方案，我看到了sham

#### HybridAlpha: An Efficient Approach for Privacy-Preserving Federated Learning

联合学习已成为一种有希望的协作和隐私保护学习方法。联邦学习过程中的参与者通过交换模型参数而不是他们可能希望保持私密的实际训练数据来协作训练模型。但是，参数交互和生成的模型仍可能会泄露有关所用训练数据的信息。为了解决这些隐私问题，已经提出了几种基于差分隐私和安全多方计算（SMC）的方法。它们通常会导致大量的通信开销和缓慢的培训时间。在本文中，我们提出了HybridAlpha，这是一种使用基于功能加密的SMC协议保护隐私的联合学习的方法。该协议简单，有效且对退出参与者具有弹性。我们评估有关使用联合学习过程在MNIST数据集上训练CNN的训练时间和数据量交换的方法。对现有基于密码的SMC解决方案进行的评估表明，HybridAlpha可以平均减少68％的训练时间并将数据传输量减少92％，同时提供与现有解决方案相同的模型性能和隐私保证。

#### ARIANN: Low-Interaction Privacy-Preserving Deep Learning via Function Secret Sharing

我们提出了ARIANN，这是一种低交互性的框架，用于对敏感数据进行私人培训和标准深度神经网络的推理。该框架实现了半诚实的两方计算并利用了功能秘密共享，这是一种最新的加密协议，仅使用轻量级原语通过输入大小的单个消息即可实现高效的在线阶段，用于比较和乘法等操作。神经网络的基础。它建立在PyTorch之上，提供了包括ReLU，MaxPool和BatchNorm在内的多种功能，并允许使用AlexNet或ResNet18之类的模型。我们报告用于远程服务器推理和训练的实验结果。最后，我们提议扩展以支持n-party私有联合学习。

## DP

#### Scalable and Differentially Private Distributed Aggregation in the Shuffled Model

B. Ghazi, R. Pagh, and A. Velingker. Scalable and differentially private distributed aggregation in the shuffled model. arXiv preprint arXiv:1906.08320, 2019.

https://arxiv.org/abs/1906.08320

联合学习通过使用安全聚合方法实现梯度下降，有望使机器学习在分布式私有数据集上可行。这个想法是在不透露单个用户贡献的情况下计算全局权重更新。当前用于安全聚合的实用协议在“诚实但好奇”的环境中工作，其中假设服务器诚实并遵循该协议，观察到服务器的所有来往的好奇对手无法学习任何私人信息。用于隐私保护协议的更具可伸缩性和鲁棒性的原语是用户数据的改组，以便隐藏每个数据项的来源。 Bittau等人在Encode-Shuffle-Analyze框架中提出了一种用于混洗的高度可扩展且安全的协议，即所谓的混合网，作为隐私保护分析的原始方法，后来由Erlingsson等人进行了分析研究。 Cheu等人和Balle等人最近的论文。给出了用于安全聚合的协议，该协议在此“混洗模型”中实现了不同的隐私保证。但是，它们的协议要付出一定的代价：预期的聚合错误或每位用户的通信量会随着用户数$ n $的多项式$ n ^ {\ Omega（1）} $扩展。在本文中，我们提出了一种简单有效的协议，用于在改组后的模型中进行聚合，其中通信和错误仅在$ n $中对数增加。我们的新技术是概念上的“隐形斗篷”，它使用户的数据与随机噪声几乎无法区分，同时对总和引入零失真。

#### Protection Against Reconstruction and Its Applications in Private Federated Learning

A. Bhowmick, J. Duchi, J. Freudiger, G. Kapoor, and R. Rogers. Protection against reconstruction and its applications in private federated learning. arXiv preprint arXiv:1812.00984, 2018.

https://arxiv.org/abs/1812.00984

在大规模的统计学习中，数据收集和模型拟合正逐渐从外围数据收集转向外围设备（如电话，手表，健身追踪器）。随着分散数据的增加，在保持隐私同时允许足够的信息适合准确，有用的统计模型的挑战也越来越大。这激发了本地隐私概念-最重要的是，本地差异隐私提供了针对敏感数据泄露的强大保护-在统计人员或学习者甚至没有观察到数据之前就对其进行了模糊处理，从而为个人数据提供了强有力的保护。然而，传统上采用的本地隐私可能证明对实际使用而言过于严格，尤其是在现代高维统计和机器学习问题中。因此，我们重新审视我们提供保护的披露和对手的类型，考虑事先具有有限信息的对手，并确保其具有很高的可能性，以确保他们无法在有用的公差范围内重建个人数据。通过重新概念化这些保护，我们允许发布更多有用的数据-本地差异隐私中的较大隐私参数-，并且我们针对\ emph {all}隐私级别的统计学习问题设计了新的（minimax）最佳本地差异私有机制。因此，我们提出了以前不可能进行的大规模本地私有模型训练的可行方法，从理论和经验上证明了我们可以在不降低实用性的情况下适应大规模图像分类和语言模型。

#### Differentially Private Federated Learning: A Client Level Perspective

R. C. Geyer, T. Klein, and M. Nabi. Differentially private federated learning: A client level perspective. arXiv preprint arXiv:1712.07557, 2017.

https://arxiv.org/abs/1712.07557

联合学习是隐私保护方面的最新进展。在这种情况下，受信任的策展人聚合由多个客户端以分散方式优化的参数。然后将生成的模型分发回所有客户端，最终收敛到联合代表模型，而无需显式共享数据。但是，该协议容易受到差分攻击，这种攻击可能源自在联合优化过程中做出贡献的任何一方。在这种攻击中，将通过分析分布式模型来揭示客户在培训期间的贡献以及有关其数据集的信息。我们解决了这个问题，并提出了一种用于客户端差分隐私保护联合优化的算法。目的是在培训期间隐藏客户的贡献，平衡隐私损失和模型性能之间的权衡。实证研究表明，在有足够数量的参与客户的情况下，我们提出的程序可以在模型性能上以较小的成本维持客户级的差异隐私。

#### Differentially Private Meta-Learning

 J. Li, M. Khodak, S. Caldas, and A. Talwalkar. Differentially-private gradient-based meta-learning. Technical Report, 2019.

https://arxiv.org/abs/1909.05830

参数传递是元学习的一种众所周知的通用方法，其应用包括少拍学习，联合学习和强化学习。但是，参数传递算法通常需要共享已对来自特定任务的样本进行过训练的模型，从而使任务所有者容易受到隐私的侵犯。我们在这种情况下进行了首次正式的隐私研究，并将任务全局差异隐私的概念形式化为对更常见的威胁模型的实际放宽。然后，我们针对基于梯度的参数传递提出了一种新的差分私有算法，该算法不仅满足此隐私要求，而且在凸设置中保留了可证明的传递学习保证。从经验上讲，我们将分析应用于具有个性化和少量快照分类的联合学习问题，这表明允许从较普遍研究的局部隐私概念放宽到任务全局隐私会导致递归神经语言建模和复制的性能大大提高。图像分类。

#### Learning Differentially Private Recurrent Language Models

H. B. McMahan, D. Ramage, K. Talwar, and L. Zhang. Learning differentially private recurrent language models. In International Conference on Learning Representations, 2018.

https://arxiv.org/abs/1710.06963

我们证明，可以用用户级别的差异性隐私保证来训练大型递归语言模型，而预测准确性只需很少的成本。 我们的工作基于对用户分区数据和用于随机梯度下降的隐私权进行深度网络训练的最新进展。 特别是，我们将用户级别的隐私保护添加到联合平均算法中，该算法从用户级别的数据中进行“大步”更新。 我们的工作表明，给定一个具有足够多用户的数据集（即使是很小的Internet规模的数据集也很容易满足这一要求），实现差异隐私是以增加计算为代价的，而不是像大多数以前的工作那样以减少效用为代价。 我们发现，在大型数据集上进行训练时，我们的私有LSTM语言模型在数量和质量上与无噪声模型相似。

#### Differentially Private Learning with Adaptive Clipping

O. Thakkar, G. Andrew, and H. B. McMahan. Differentially private learning with adaptive clipping. arXiv preprint arXiv:1905.03871, 2019.

https://arxiv.org/abs/1905.03871

我们引入了一种新的自适应裁剪技术，用于训练具有用户级别差异隐私的学习模型，从而无需进行大量参数调整。 解决该问题的先前方法使用联合随机梯度下降或具有噪声更新的联合平均算法，并使用Moments Accountant计算差异隐私保证。 这些方法依赖于为每个用户的模型更新选择一个规范界限，需要对其进行仔细调整。 最佳值取决于学习率，模型体系结构，对每个用户的数据进行的通过次数以及可能的其他各种参数。 我们显示，基于未裁剪的规范分布的目标分位数的差分私有估计，自适应地设置应用于每个用户更新的裁剪规范就足以消除对此类广泛参数调整的需求。

#### cpSGD: Communication-efficient and differentially-private distributed SGD

N. Agarwal, A. T. Suresh, F. X. X. Yu, S. Kumar, and B. McMahan. cpSGD: Communication-efficient and differentially-private distributed sgd. In Advances in Neural Information Processing Systems, 2018.

https://arxiv.org/abs/1805.10559

分布式随机梯度下降是分布式学习中的重要子程序。当客户端是移动设备时，一个特别令人感兴趣的设置是两个重要的关注点，即通信效率和客户端的隐私。近来的一些工作集中在降低通信成本或引入隐私保证上，但是，已知的所有提议的通信有效方法均未实现隐私保护，已知的隐私机制均未达到通信效率。为此，我们研究了实现通信效率和差异隐私的算法。对于$ d $个变量和  $n \approx d$ 个客户端，建议的方法使用每个客户端每个坐标的$ O（\ log \ log（nd））$位通信，并确保恒定的隐私。

我们还扩展和改进了对Binomial mechanism的先前分析，结果表明，它可以实现与高斯机制几乎相同的效用，同时所需的表示位数更少，这可以是独立利益。

#### Federated Learning with Bayesian Differential Privacy

我们考虑了通过正式的隐私保证来加强联合学习的问题。我们建议采用贝叶斯差分隐私，这是对类似分布的数据的差分隐私的放松，以提供更清晰的隐私丢失范围。我们将贝叶斯隐私会计方法调整为适用于联合设置，并建议进行多项改进以提高不同级别的隐私预算效率。我们的实验表明，对于图像分类任务（包括医疗应用）的联合学习而言，相对于最新的差分隐私界限而言，它具有显着优势，这使客户端级别的隐私预算低于ε=  1，而在客户端级别则低于ε= 0.1实例级别。较低的噪声量也有助于提高模型的准确性，并减少通信回合的次数。

#### Policy-Based Federated Learning

应用，连接的设备，服务和智能环境日益包围着我们，这些环境要求对各种个人数据进行细粒度的访问。我们的个人和专业政策的固有复杂性以及与这些分析服务进行交互时的偏好提出了重要的隐私挑战。此外，由于数据的敏感性以及监管和技术障碍，以集中方式进行这些政策谈判并不总是可行的。在本文中，我们介绍了PoliFL1，这是一个分散的，基于边缘的，用于基于策略的个人数据分析的框架。  PoliFL汇集了许多现有的已建立组件，以在分布式环境中提供保护隐私的分析。我们使用流行的私有分析示例Federated  Learning评估我们的框架，并证明对于不同的模型大小和使用案例，PoliFL能够在非常合理的资源和时间预算内执行准确的模型训练和推理。

#### FedSel: Federated SGD under Local Differential Privacy with Top-k Dimension Selection

随着小工具产生大量数据，在移动设备上进行联合学习已成为一种新兴趋势。在联邦环境中，随机梯度下降（SGD）已广泛用于各种机器学习模型的联邦学习中。为了防止根据用户的敏感数据计算出的梯度导致的隐私泄漏，最近在联合SGD中将本地差分隐私（LDP）视为隐私保证。然而，现有的解决方案具有尺寸依赖性问题：注入的噪声基本上与尺寸d成比例。在这项工作中，我们为LDP下的联合SGD提出了一个两阶段框架FedSel，以缓解此问题。我们的关键思想是，并非所有维度都同样重要，因此我们根据联盟SGD每次迭代中的贡献私下选择Top-k维度。具体来说，我们提出了三种私有维度选择机制，并采用了梯度累积技术来稳定学习过程中的噪声更新。我们还从理论上分析了FedSel的私密性，准确性和时间复杂性，这优于最新的解决方案。在现实世界和综合数据集上的实验证明了我们框架的有效性和效率。

#### LDP-Fed: Federated Learning with Local Differential Privacy

本文介绍LDP-Fed，这是一种新颖的联合学习系统，具有使用本地差分隐私（LDP）的正式隐私保证。开发现有的LDP协议主要是为了确保单个数字或类别值（例如Web访问日志中的点击计数）集合中的数据隐私。但是，在联合学习模型中，参数更新是从每个参与者中反复收集的，并且由高精度的高维连续值（小数点后10位数）组成，这使得现有的LDP协议不适用。为了应对LDP馈送中的这一挑战，我们设计和开发了两种新颖的方法。首先，LDP-Fed的LDP模块为正式的差分隐私保证提供了保证，该模型在多个个体参与者的私有数据集的大规模神经网络的联合训练中重复收集模型训练参数。其次，LDP-Fed实现了一系列选择和过滤技术，用于与参数服务器干扰和共享选择的参数更新。我们验证了使用压缩LDP协议部署的系统在针对公共数据训练深度神经网络时的有效性。在模型准确性，隐私保护和系统功能方面，我们将这种版本的LDP-Fed（造币CLDP-Fed）与其他最新方法进行了比较。

#### Exploring Private Federated Learning with Laplacian Smoothing

联合学习旨在通过协作学习模型而不在用户之间共享私有数据来保护数据隐私。但是，攻击者仍然可以通过攻击已发布的模型来推断私人训练数据。差异性隐私（DP）提供了针对此类攻击的统计保证，其隐私性可能会降低训练模型的准确性或实用性。在本文中，我们将基于拉普拉斯平滑的效用增强方案应用于差分私有联合学习（DP-Fed-LS），其中采用高斯噪声注入的参数聚合在统计精度上得到了改善。我们为统一抽样和Poisson二次抽样提供紧密的封闭形式隐私边界，并为有差别的私人联合学习（有无Laplacian平滑）提供了相应的DP保证。在MNIST，SVHN和Shakespeare数据集上进行的实验表明，在两种二次采样机制下，该方法均可以通过DP保证提高模型的准确性。

#### Secure Byzantine-Robust Machine Learning

越来越多的机器学习系统被部署到边缘服务器和设备（例如移动电话）并以协作的方式进行培训。这种分布式/联合/分散式培训引起了对程序的鲁棒性，隐私性和安全性的许多担忧。尽管在解决鲁棒性，隐私或安全性方面已经进行了广泛的工作，但很少研究它们的组合。在本文中，我们提出了一种安全的两服务器协议，该协议可提供输入隐私和拜占庭的鲁棒性。此外，该协议具有通信效率高，容错能力强，并享有本地差分隐私。

#### Distributed Differentially Private Averaging with Improved Utility and Robustness to Malicious Parties

与联合学习中一样，从几方拥有的数据中学习，给提供给参与者的隐私保证和在存在恶意方的情况下计算的正确性提出了挑战。我们在分布式平均的背景下应对这些挑战，分布式平均是分布式和联合学习的基本组成部分。我们的第一个贡献是一种新颖的分布式差分专用协议，该协议自然会随着参与方数量的增长而扩展。我们协议的基本思想是沿着网络图的边缘交换相关的高斯噪声，并由各方添加的独立噪声进行补充。我们分析了协议的差异性隐私保证以及图拓扑的影响，表明即使双方仅与随机选择的其他对数通信，我们也可以匹配可信策展人模型的准确性。这与本地隐私模型（精度较低）或基于安全聚合（所有用户对都需要交换消息）的协议形成对比。我们的第二个贡献是使用户能够证明他们计算的正确性，而又不影响协议的效率和隐私保证。我们的构建依赖于标准的加密原语，例如承诺方案和零知识证明。

#### Federated Learning with Differential Privacy: Algorithms and Performance Analysis

联合学习（FL）作为分布式机器学习的一种方式，能够显着地保护客户的私人数据免遭外部对手的攻击。尽管如此，仍然可以通过分析客户端上传参数的差异来泄露私人信息，例如在深度神经网络中训练的权重。在本文中，为了有效防止信息泄漏，我们提出了一种基于差分隐私（DP）概念的新颖框架，在该框架中，在进行聚合之前将人工噪声添加到客户端的参数上，即在模型聚合FL（  NbAFL）。首先，我们证明了NbAFL通过适当地适应不同的人工噪声方差可以满足不同保护级别下的DP。然后，我们在NbAFL中建立了经过训练的FL模型的损失函数的理论收敛界。具体而言，理论界限揭示了以下三个关键属性：1）融合性能和隐私保护级别之间存在折衷，即，较好的融合性能导致较低的保护级别；  2）给定固定的隐私保护级别，增加参与FL的总客户数N可以提高融合性能；  3）对于给定的保护级别，在收敛性能方面有一个最佳的最大聚合次数（通信回合）。此外，我们提出了一种K随机调度策略，其中从N个总体客户端中随机选择K（1 

## CC

#### A training-integrity privacy-preserving federated learning scheme with trusted execution environment

 Yu Chen, Fang Luo, Tong Li, Tao Xiang, Zheli Liu, and Jin Li. A training-integrity privacypreserving federated learning scheme with trusted execution environment. Information Sciences, 522:69–79, 2020.

https://www.sciencedirect.com/science/article/abs/pii/S0020025520301201

在敏感的真实世界数据上训练的机器学习模型有望改善从医学筛查到疾病暴发发现的一切。在许多应用领域中，学习参与者将受益于合并他们的私有数据集，在汇总数据上训练精确的机器学习模型以及共享使用这些模型的收益。考虑到隐私和安全问题通常会阻止参与者提供敏感数据进行培训，因此研究人员提出了几种在联合学习系统中实现数据隐私的技术。但是，这样的技术容易受到因果攻击的攻击，从而恶意参与者可能注入虚假的训练结果，从而破坏了良好学习的模型。为此，在本文中，我们提出了一种新的保护隐私的联合学习方案，该方案可以保证深度学习过程的完整性。基于可信执行环境（TEE），我们为此方案设计了一个训练完整性协议，在其中可以检测到因果攻击。因此，迫使每个参与者正确地执行该方案的隐私保护学习算法。我们通过原型实现来评估我们的方案的性能。实验结果表明，该方案具有训练完整性和实用性。

#### Glimmers: Resolving the Privacy/Trust Quagmire

来自李锐

## Other & Hybrid

#### Practical Federated Gradient Boosting Decision Trees

hashing

Qinbin Li, Zeyi Wen, and Bingsheng He. Practical federated gradient boosting decision trees. arXiv preprint arXiv:1911.04206, 2019.

https://arxiv.org/abs/1911.04206

梯度提升决策树（GBDT）近年来非常成功，在机器学习和数据挖掘竞赛中获得了许多奖项。最近有一些关于如何在联合学习环境中训练GBDT的研究。在本文中，我们专注于水平联合学习，其中具有相同特征的数据样本分布在多方之间。但是，现有研究不够有效或不够实用。由于使用了昂贵的数据转换（例如秘密共享和同态加密）而导致效率低下，或者由于差异性隐私设计而导致的模型准确性低，使他们受苦。在本文中，我们研究了一个具有宽松隐私限制的实用联邦环境。在这种环境下，不诚实的一方可能会获得有关另一方数据的某些信息，但是对于不诚实的一方而言，仍然不可能获得其他方的实际原始数据。具体来说，每一方都通过利用基于位置敏感的哈希值的相似性信息来增加树的数量。我们证明了我们的框架是安全的，并且不会将原始记录暴露给其他方，同时训练过程中的计算开销保持较低。我们的实验研究表明，与使用双方的本地数据进行常规训练相比，我们的方法可以显着提高预测准确性，并获得与来自各方的数据原始GBDT相当的准确性。

#### A hybrid approach to privacy-preserving federated learning

SMC;DP

Stacey Truex, Nathalie Baracaldo, Ali Anwar, Thomas Steinke, Heiko Ludwig, Rui Zhang, and Yi Zhou. A hybrid approach to privacy-preserving federated learning. In Proceedings of the 12th ACM Workshop on Artificial Intelligence and Security, pages 1–11. ACM, 2019.

https://arxiv.org/abs/1812.03224

联合学习促进了模型的协作训练，而无需共享原始数据。但是，最近的攻击表明，在培训过程中仅维护数据局部性并不能提供足够的隐私保证。而是，我们需要一种联邦学习系统，该系统能够防止对训练过程中交换的消息和最终训练模型之间的推论，同时确保所得模型也具有可接受的预测准确性。现有的联合学习方法要么使用容易受到推论的安全多方计算（SMC），要么使用差分隐私，这在给定大量参与者且每个参与者都具有相对少量数据的情况下会导致准确性降低。在本文中，我们提出了一种利用差异隐私和SMC来平衡这些权衡的替代方法。将差分隐私与安全的多方计算相结合，使我们能够在不牺牲隐私的情况下，同时保持预先定义的信任率的前提下，随着方数的增加而减少噪声注入的增长。因此，我们的系统是一种可扩展的方法，可防止推理威胁并产生高精度的模型。此外，我们的系统可用于训练各种机器学习模型，并通过3种不同的机器学习算法的实验结果对其进行验证。我们的实验表明，我们的方法优于最新的解决方案。

#### Boosting privately: Privacy-preserving federated extreme boosting for mobile crowdsensing

HE;SMC

Yang Liu, Zhuo Ma, Ximeng Liu, Siqi Ma, Surya Nepal, and Robert Deng. Boosting privately: Privacy-preserving federated extreme boosting for mobile crowdsensing. arXiv preprint arXiv:1907.10218, 2019.

https://arxiv.org/abs/1907.10218

最近，Google和其他24个机构对联邦学习（FL）提出了一系列公开挑战，其中包括应用程序扩展和同态加密（HE）。前者旨在扩展FL的适用机器学习模型。后者侧重于将HE应用于FL时谁掌握了密钥。对于朴素的HE方案，将服务器设置为主密钥。这样的设置会导致严重的问题，如果服务器在解密之前不进行聚合，则服务器将有机会访问用户的更新。受这两个挑战的启发，我们提出了FedXGB，这是一种支持强制聚合的联邦极端梯度增强（XGBoost）方案。 FedXGB主要实现以下两个突破。首先，FedXGB涉及用于FL的基于HE的新安全聚合方案。通过结合秘密共享和同态加密的优点，该算法可以解决上述第二个挑战，并且对用户退出具有鲁棒性。然后，FedXGB通过将安全聚合方案应用于XGBoost的分类和回归树构建，将FL扩展到新的机器学习模型。此外，我们进行了全面的理论分析和广泛的实验，以评估FedXGB的安全性，有效性和效率。结果表明，与原始XGBoost相比，FedXGB的准确度损失小于1％，并且可以为基于FL的HE模型更新聚合提供约23.9％的运行时间和33.3％的通信减少。

#### Privacy-Preserving Ridge Regression on Hundreds of Millions of Records

HE;SMC(Yao)

Valeria Nikolaenko, Udi Weinsberg, Stratis Ioannidis, Marc Joye, Dan Boneh, and Nina Taft. Privacy-preserving ridge regression on hundreds of millions of records. In 2013 IEEE Symposium on Security and Privacy, pages 334–348. IEEE, 2013.

https://ieeexplore.ieee.org/document/6547119

Ridge回归是一种将大量数据点作为输入并通过这些点找到最适合的线性曲线的算法。 该算法是许多机器学习操作的基础。 我们提出了一种用于保护隐私的岭回归的系统。 系统以明文形式输出最佳拟合曲线，但不公开有关输入数据的其他信息。 我们的方法结合了同态加密和Yao乱码电路，在算法的不同部分都使用了后者，以获得最佳性能。 我们实现了完整的系统并在真实数据集上进行了实验，结果表明，该系统明显优于仅基于同态加密或Yao电路的纯实现。

ps：13年的论文，有点老，稍微看一下，可以不加



#### VerifyNet: Secure and Verifiable Federated Learning

伪随机技术；同态哈希函数

Xu G, Li H, Liu S, et al. Verifynet: Secure and verifiable federated learning[J]. IEEE Transactions on Information Forensics and Security, 2019, 15: 911-926.

https://ieeexplore.ieee.org/abstract/document/8765347

联邦学习作为一种新兴的神经网络训练模型，由于能够在不收集用户原始数据的情况下更新参数，因此受到了广泛的关注。但是，由于对手可以从共享渐变中跟踪并获取参与者的隐私，因此联合学习仍然面临各种安全和隐私威胁。在本文中，我们考虑了深度神经网络（DNN）训练过程中的两个主要问题：1）在训练过程中如何保护用户的隐私（即局部梯度），以及2）如何验证完整性（或正确性）服务器返回的汇总结果中。为了解决上述问题，已经提出了几种关注安全或隐私保护的联合学习的方法，并将其应用于各种情况。但是，这仍然是一个悬而未决的问题，使客户可以验证云服务器是否正常运行，同时在培训过程中保证用户的隐私。在本文中，我们提出了VerifyNet，这是第一个保护隐私和可验证的联合学习框架。具体来说，我们首先提出一种双重屏蔽协议，以确保在联合学习期间用户本地梯度的机密性。然后，需要云服务器向每个用户提供有关其聚合结果正确性的证明。我们声称，除非对手能够解决我们模型中采用的NP难题，否则对手不可能通过伪造Proof来欺骗用户。此外，VerifyNet还支持培训过程中退出的用户。在真实数据上进行的大量实验也证明了我们提出的方案的实际性能。

在本文中，我们提出了VerifyNet，它支持向每个用户验证服务器的计算结果。此外，VerifyNet还支持在培训过程中辍学的用户。安全分析表明，在真实但有趣的安全设置下，我们的VerifyNet具有很高的安全性。此外，对真实数据进行的实验也证明了我们提出的方案的实际性能。作为未来研究工作的一部分，我们将专注于减少整个协议的通信开销。

#### Efficient and Privacy-Enhanced Federated Learning for Industrial Artificial Intelligence

DP；HE

Hao M, Li H, Luo X, et al. Efficient and privacy-enhanced federated learning for industrial artificial intelligence[J]. IEEE Transactions on Industrial Informatics, 2019, 16(10): 6532-6542.

https://ieeexplore.ieee.org/abstract/document/8859260/

最新的隐私保护FL方案主要基于三种加密方法：SMC，差分隐私（DP）和HE。首先，SMC显然不适合我们的解决方案，因为我们需要非交互协议来执行安全聚合。其次，DP是防止培训过程中隐私泄露的有效方法。但是，传统的DP 可能不适合FL，因为需要可信的第三方将噪声添加到统计结果中。这与CS通常被认为是不诚实的事实相反。第三，基于HE的方案是一种潜在的解决方案，如果为所有参与者分配了相同的密钥，则可用于支持非交互式FL。但是，如果在训练期间有多个实体勾结在一起，则这很容易受到攻击。另一变体，例如阈值HE，迫使CS仅在多个参与者的协助下才能正确地解密聚合的密文。但是，交互式解密严重增加了通信开销，尤其是对于神经网络中的高维数据而言。

为了解决上述挑战，我们提出了隐私增强的联合学习（PEFL）以实现IAI的高效和PEFL。我们的贡献总结如下：

1）PEFL在每个安全聚合中都是非交互的。为了实现我们的安全聚合协议，私有梯度的同构密文嵌入了增强错误学习（A-LWE）术语。特别是，我们提供了一个具体的实例，其中我们利用了改进的BGV  HE方案，该方案消除了键切换操作并增加了明文空间。  

2）即使对手与多个实体勾结，PEFL也可以防止隐私从局部梯度以及共享参数中泄漏出去。具体来说，为了进一步保护训练数据的隐私，我们使用分布式高斯机制实现了示例级DP。  3）详细的安全性分析表明，PEFL提供了量子后的安全性，并保证了聚合器的遗忘安全性。绩效评估证明了实际的培训准确性，以及有效的计算和通信开销。

#### A generic framework for privacy preserving deep learning

MPC；DP；2018

我们详细介绍了用于保护深度学习的隐私的新框架，并讨论了其资产。该框架非常重视所有权和数据的安全处理，并基于命令和张量链引入了有价值的表示形式。这种抽象允许人们实施复杂的隐私保护结构，例如联合学习，安全多方计算和差异性隐私，同时仍然向最终用户展示熟悉的深度学习API。我们在Boston  Housing和Pima Indian  Diabetes数据集上报告了早期结果。尽管除“差异性隐私”之外的隐私功能不会影响预测准确性，但该框架的当前实现会引入大量的性能开销，这将在开发的后续阶段解决。我们相信这项工作是一个重要的里程碑，它引入了第一个可靠的通用框架来保护深度学习的隐私。

ps：感觉挺偏应用的

#### Federated Generative Privacy

GAN;2019

在本文中，我们提出了FedGP，这是在联合学习环境中用于保护隐私的数据发布的框架。我们使用生成对抗网络，通过FedAvg算法训练其生成器组件，以绘制保存隐私的人工数据样本并凭经验评估信息泄露的风险。我们的实验表明，FedGP能够生成高质量的标记数据，从而成功地训练和验证监督模型。最后，我们证明了我们的方法显着降低了此类模型对反演攻击进行建模的漏洞。

#### Enhancing the Privacy of Federated Learning with Sketching

sketching algorithms

为响应对用户隐私日益增长的担忧，联合学习已成为一种有前途的工具，可以在设备网络上训练统计模型，同时保持数据本地化。联合学习方法直接在用户设备上运行培训任务，并且不与第三方共享原始用户数据。但是，当前的方法仍然在训练过程中共享模型更新，其中可能包含私人信息（例如，一个人的体重和身高）。旨在改善联合学习的隐私的现有工作在以下一个或多个关键领域做出了折衷：性能（特别是通信成本），准确性或隐私。为了更好地优化这些折衷，我们建议草绘算法具有一个独特的优势，因为它们可以在保持准确性的同时提供隐私和性能优势。我们使用三个具有代表性的学习模型的原型来评估基于草图的联合学习的可行性。我们的初步发现表明，有可能在不牺牲性能或准确性的情况下为联邦学习提供强有力的隐私保证。我们的工作着重指出，在分布式环境中，隐私和通信之间存在着根本的联系，并提出了一些重要的开放性问题，这些问题涉及理论，实践方法和系统的设计，私人联合学习的系统设计。

#### Anonymizing Data for Privacy-Preserving Federated Learning

联合学习使您能够从分布在多个站点的数据中训练全局机器学习模型，而无需移动数据。这在医疗应用中尤其重要，在医疗应用中，数据中充斥着高度敏感的个人信息，并且数据分析方法必须可证明符合监管准则。尽管联合学习阻止共享原始数据，但仍然有可能对训练过程中暴露的模型参数或生成的机器学习模型发起隐私攻击。在本文中，我们提出了在联合学习的上下文中提供隐私的第一种句法方法。与最新的基于差异的基于隐私的框架不同，我们的方法旨在最大化效用或模型性能，同时支持GDPR和HIPAA要求的可辩护的隐私级别。我们使用100万患者的真实电子健康数据，对医疗保健领域中的两个重要问题进行了全面的实证评估。结果证明了我们的方法在实现较高模型性能的同时提供了所需的隐私级别的有效性。通过比较研究，我们还表明，对于各种数据集，实验设置和隐私预算，与联合学习中基于隐私权的不同技术相比，我们的方法具有更高的模型性能。

#### Learn to Forget: User-Level Memorization Elimination in Federated Learning

摘要—联合学习是一种去中心化的机器学习技术，在研究领域和现实世界中都引起了广泛的关注。但是，当前的保护隐私的联合学习方案仅为用户提供了一种安全的方式来贡献其私有数据，而从未留下撤回对模型更新的贡献的方式。这种不可逆的设置可能会破坏有关数据保护的法规，并增加数据提取的风险。为了解决该问题，本文介绍了一种用于联合学习的新概念，称为记忆消除。基于此概念，我们提出了联合学习框架Forsaken，该框架允许用户消除在训练模型中存储其私人数据的麻烦。具体来说，被遗忘者中的每个用户都部署有可训练的虚拟渐变生成器。经过训练步骤后，生成器可以产生虚拟梯度来刺激机器学习模型的神经元，从而消除特定数据的记忆。此外，我们证明被遗忘者的额外记忆消除服务不会破坏联合学习的通用过程或降低其安全性

#### Enhancing Privacy via Hierarchical Federated Learning

联合学习遭受一些与隐私相关的问题，这些问题使参与者面临各种威胁。联合学习的集中式体系结构加剧了许多这些问题。在本文中，我们讨论将联邦学习应用于分层体系结构作为一种潜在的解决方案。我们介绍了对培训过程进行更灵活的分散控制的机会，以及对参与者隐私的影响。此外，我们研究了增强防御和验证方法的效率和有效性的可能性。

#### Privacy Preserving Distributed Machine Learning with Federated Learning

边缘计算和分布式机器学习已发展到可以变革特定组织的水平。诸如物联网（IoT）之类的分布式设备通常会产生大量数据，最终产生大数据，这对于揭示隐藏模式以及医疗，银行和警务等众多领域的其他见解至关重要。与医疗保健和银行业等领域相关的数据可能包含潜在的敏感数据，如果未适当清理，这些数据可能会公开。联合学习（FedML）是最近开发的分布式机器学习（DML）方法，该方法试图通过将ML模型的学习引入数据所有者的手中来保护隐私。但是，文献显示了不同的攻击方法，例如利用ML模型以及协调服务器来检索私有数据的漏洞的成员资格推断。因此，FedML需要采取其他措施来保证数据隐私。此外，大数据通常比标准计算机需要更多的资源。本文通过提出一种名为DISTPAB的分布式扰动算法来解决这些问题，以保护水平分区数据的隐私。  DISTPAB通过利用分布式环境中资源的不对称性来分配隐私保护任务来缓解计算瓶颈，分布式环境可以具有资源受限的设备以及高性能计算机。实验表明，DISTPAB具有高精度，高效率，高可扩展性和高抗攻击性。保留隐私的FedML的进一步实验表明，DISTPAB是一种出色的解决方案，可以在保留高数据实用性的同时阻止DML中的隐私泄漏。