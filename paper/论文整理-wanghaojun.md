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

## SMC

#### Practical Secure Aggregation for Federated Learning on User-Held Data 

#### Practical secure aggregation for privacy-preserving machine learning

#### Scalable and Differentially Private Distributed Aggregation in the Shuffled Model

B. Ghazi, R. Pagh, and A. Velingker. Scalable and differentially private distributed aggregation in the shuffled model. arXiv preprint arXiv:1906.08320, 2019.

https://arxiv.org/abs/1906.08320

联合学习通过使用安全聚合方法实现梯度下降，有望使机器学习在分布式私有数据集上可行。这个想法是在不透露单个用户贡献的情况下计算全局权重更新。当前用于安全聚合的实用协议在“诚实但好奇”的环境中工作，其中假设服务器诚实并遵循该协议，观察到服务器的所有来往的好奇对手无法学习任何私人信息。用于隐私保护协议的更具可伸缩性和鲁棒性的原语是用户数据的改组，以便隐藏每个数据项的来源。 Bittau等人在Encode-Shuffle-Analyze框架中提出了一种用于混洗的高度可扩展且安全的协议，即所谓的混合网，作为隐私保护分析的原始方法，后来由Erlingsson等人进行了分析研究。 Cheu等人和Balle等人最近的论文。给出了用于安全聚合的协议，该协议在此“混洗模型”中实现了不同的隐私保证。但是，它们的协议要付出一定的代价：预期的聚合错误或每位用户的通信量会随着用户数$ n $的多项式$ n ^ {\ Omega（1）} $扩展。在本文中，我们提出了一种简单有效的协议，用于在改组后的模型中进行聚合，其中通信和错误仅在$ n $中对数增加。我们的新技术是概念上的“隐形斗篷”，它使用户的数据与随机噪声几乎无法区分，同时对总和引入零失真。

#### Privacy preserving regression modelling via distributed computation

Ashish P Sanil, Alan F Karr, Xiaodong Lin, and Jerome P Reiter. Privacy preserving regression modelling via distributed computation. In Proceedings of the tenth ACM SIGKDD international conference on Knowledge discovery and data mining, pages 677–682. ACM, 2004.

https://dl.acm.org/doi/10.1145/1014052.1014139

数据所有者不愿与拥有相关数据库的其他人共享其可能的机密或专有数据，这是进行互利数据挖掘分析的严重障碍。 我们解决了垂直分区数据的情况-多个数据所有者/代理商各自拥有每个数据记录的一些属性。 我们关注的情况是，代理商希望对完整的记录进行线性回归分析，而又不透露其自身属性的值。 本文介绍了一种算法，使此类机构能够计算全局回归方程式的确切回归系数，并在保护其数据机密性的同时执行一些基本的拟合优度诊断。 在隐私场景之外的更一般的设置中，此算法也可以看作是用于回归分析的分布式计算的方法。

ps：看摘要，有点像跨组织的纵向联邦学习，所以尽管2004年的 我也放上了



#### SMSS: Secure Member Selection Strategy in Federated Learning

https://ieeexplore.ieee.org/document/9136885

数据安全和用户隐私问题已成为重要领域。由于联邦学习（FL）可以解决数据安全和隐私问题带来的问题，因此它开始被应用于许多不同的应用机器学习任务中。但是，FL不会验证来自系统中不同方面的数据的质量。因此，具有较少公共实体的低质量数据集可以与其他实体一起训练。这可能导致大量的计算资源浪费，以及来自作为联邦成员的恶意客户端对FL模型的攻击。为了解决这个问题，本文提出了一种安全的成员选择策略（SMSS），可以在训练之前评估成员的数据质量。使用SMSS，只有数据集共享的公共实体多于某个阈值才能选择用于学习，而公共对象较少的恶意客户端无法获取有关该模型的任何信息。本文实现了SMSS，并通过多个广泛的经验来评估其性能。实验结果表明，SMSS是安全，有效和有效的。

ps：一篇用于联邦学习中选择参与成员的论文，混合方案，我看到了sham

## DP

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

## CC

#### A training-integrity privacy-preserving federated learning scheme with trusted execution environment

 Yu Chen, Fang Luo, Tong Li, Tao Xiang, Zheli Liu, and Jin Li. A training-integrity privacypreserving federated learning scheme with trusted execution environment. Information Sciences, 522:69–79, 2020.

https://www.sciencedirect.com/science/article/abs/pii/S0020025520301201

在敏感的真实世界数据上训练的机器学习模型有望改善从医学筛查到疾病暴发发现的一切。在许多应用领域中，学习参与者将受益于合并他们的私有数据集，在汇总数据上训练精确的机器学习模型以及共享使用这些模型的收益。考虑到隐私和安全问题通常会阻止参与者提供敏感数据进行培训，因此研究人员提出了几种在联合学习系统中实现数据隐私的技术。但是，这样的技术容易受到因果攻击的攻击，从而恶意参与者可能注入虚假的训练结果，从而破坏了良好学习的模型。为此，在本文中，我们提出了一种新的保护隐私的联合学习方案，该方案可以保证深度学习过程的完整性。基于可信执行环境（TEE），我们为此方案设计了一个训练完整性协议，在其中可以检测到因果攻击。因此，迫使每个参与者正确地执行该方案的隐私保护学习算法。我们通过原型实现来评估我们的方案的性能。实验结果表明，该方案具有训练完整性和实用性。

ps：好不容易找到篇可信执行环境的论文 下不下来 没有免费的 在线看一下摘要和介绍吧

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



