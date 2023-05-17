Parallelism Training 主要在做计算、管道，数据，模型并行化和通信。
大模型并行化最近五年的顶会文章总结， 一共32篇，其中Parallelism & Distributed Systems 重点文章阅读14篇，在后面用粗体标出。
1. [OSDI '22] Unity: Accelerating DNN Training Through Joint Optimization of Algebraic Transformations and Parallelization
2. [EuroSys '22] Varuna: Scalable, Low-cost Training of Massive Deep Learning Models
3. [SC '21'] Chimera: Efficiently Training Large-Scale Neural Networks with Bidirectional Pipelines
4. [ICML '21] PipeTransformer: Automated Elastic Pipelining for Distributed Training of Large-scale Models
5. [OSDI '20] A Unified Architecture for Accelerating Distributed DNN Training in Heterogeneous GPU/CPU Clusters
6. [ATC '20] HetPipe: Enabling Large DNN Training on (Whimpy) Heterogeneous GPU Clusters through Integration of Pipelined Model Parallelism and Data Parallelism
7. [NeurIPS '19] GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism
8. [SOSP '19] A Generic Communication Scheduler for Distributed DNN Training Acceleration
9. [SOSP '19] PipeDream: Generalized Pipeline Parallelism for DNN Training
10. [EuroSys '19] Parallax: Sparsity-aware Data Parallel Training of Deep Neural Networks
11. [arXiv '18] Horovod: fast and easy distributed deep learning in TensorFlow
12. [ATC '17] Poseidon: An Efficient Communication Architecture for Distributed Deep Learning on GPU Clusters
13. [EuroSys '16] STRADS: A Distributed Framework for Scheduled Model Parallel Machine Learning
14. [EuroSys '16] GeePS: Scalable Deep Learning on Distributed GPUs with a GPU-specialized Parameter Server



1. Bamboo: Making Preemptible Instances Resilient for Affordable Training of Large DNNs NSDI'23
1. nsdi23-bamboo.pdf
2. Bamboo通过有效使用可抢占实例（即在闲置时可以以更便宜的价格获得，但可能在被优先用户请求时被抢占），显著降低训练成本。然而，这样做需要新形式的弹性和效率来应对频繁的抢占可能性，这是一种与现有检查点技术针对的正常集群设置中偶尔出现的故障模型截然不同的失败模式。
3. Bamboo这是一个分布式系统，通过将冗余计算引入训练流水线中来解决这些挑战，即一个节点不仅计算自己的层，还计算其相邻节点的某些层。训练大型模型通常需要管道并行性，其中“管道泡泡”自然存在。Bamboo将冗余计算谨慎地填充到这些泡泡中，以低成本提供弹性。在各种广泛使用的DNN模型中，Bamboo的训练吞吐量比传统检查点技术提高了3.7倍，并且与使用按需实例的设置相比，成本降低了2.4倍。

2. MPress: Democratizing Billion-Scale Model Training on Multi-GPU Servers via Memory-Saving Inter-Operator Parallelism HPCA'23
1. 现有的节省内存的技术，如GPU-CPU交换、重计算和ZeRO系列，都存在额外的计算、通信开销或内存减少受限等问题。MPress是个单服务器多GPU系统，打破了亿级模型训练的GPU内存墙，交替选择低交叉GPU通信流量的操作间并行性，并结合重计算和交换，以平衡训练性能和维持的模型大小。此外，MPress采用了一种新颖的快速D2D交换技术，它同时利用多个高带宽NVLink将张量交换到轻负载GPU上，基于这样一个关键观察：操作间并行训练可能导致GPU内存利用不平衡，最少使用的设备可以释放出多余的内存空间，而它们之间的高端互连可以支持低开销的交换。最后，我们将MPress与两个代表性的操作间并行训练系统PipeDream和DAPPLE进行了集成。在DGX-1和DGX-2两代现代GPU服务器上，分别配备8个V100或A100卡，使用两个流行的DNN模型Bert和GPT进行实验，结果表明，MPress相比于具有相同内存减少的ZeRO系列能够显著提高训练吞吐量，同时能够训练比重计算基线更大的模型。

3. TopoOpt: Optimizing the Network Topology for Distributed DNN Training NSDI'23
1. TopoOpt是一种针对深度神经网络（DNN）训练工作负载的新型直连网络架构。TopoOpt通过三个维度的分布式训练过程协同优化：计算、通信和网络拓扑。它展示了AllReduce流量的可变性，并利用这一特性构建了用于DNN训练作业的高效网络拓扑。TopoOpt然后使用交替优化技术和一个名为TotientPerms的群论算法，找到最佳的网络拓扑和路由计划，以及并行化策略。TopoOpt构建了一个具有远程直接内存访问（RDMA）转发功能的全功能12节点直连原型，传输速率为100 Gbps。对真实分布式训练模型的大规模模拟表明，与类似成本的Fat-Tree互连相比，TopoOpt将DNN训练时间缩短了最多3.4倍。

4. Optimus-CC: Efficient Large NLP Model Training with 3D Parallelism Aware Communication Compression ASPLOS'23
1. Optimus-CC是一种用于大规模NLP模型 （3D并行处理的NLP模型）的快速、可扩展的分布式训练框架，采用了积极的通信压缩技术。Optimus-CC与现有的通信压缩框架不同，具有以下特点：首先，压缩了流水线并行（阶段间）流量。具体而言，对阶段间的反向传播和嵌入同步进行压缩，以及采用现有的数据并行处理流量压缩方法，避免由压缩引起的模型质量下降。我们进一步提供了数学和经验分析，以表明Optimus-CC可以成功地抑制压缩误差。最后，我们分析了流水线，并选择性地压缩了位于关键路径上的流量。这进一步有助于减少压缩误差。我们在GPU集群上展示了我们的解决方案，并实现了比分布式训练现有最先进解决方案更高的加速，同时不牺牲模型质量。

5. Slapo: A Schedule Language for Progressive Optimization of Large Deep Learning Model Training arxiv
1. DL框架（如PyTorch）使用动态图来方便模型开发人员，从静态图优化（例如XLA）到针对大规模分布式训练进行优化（例如DeepSpeed和Megatron-LM）的各种方法都被实践者提出来以提高训练效率，但这些方法都需要在灵活性和可用性之间做出取舍。
2. 调度语言Slapo 将一个张量级运算符的平台特定优化与它的算术定义分离，以将模型的执行与定义分离。Slapo 可以用一组调度基元对 PyTorch 模型进行转换，以实现常见的模型训练优化，例如高性能内核、有效的3D并行和高效的激活检查点。与现有的优化解决方案相比，Slapo 可以通过高级基元“按需”逐步优化模型，从而在很大程度上保留了可编程性和可调试性。我们的评估结果显示，通过使用 Slapo 对现有的手工优化进行系统化调度，我们能够在单台拥有8个 NVIDIA V100 GPU 的计算机上将训练吞吐量提高最多3.35倍，在多台拥有多达64个GPU的计算机上将训练吞吐量提高最多1.32倍，与 DeepSpeed 和 Megatron-LM 的开箱即用性能相比。

6. Zeus: Understanding and Optimizing {GPU} Energy Consumption of {DNN} Training NSDI'23
1. Zeus，一种优化框架，通过自动查找重复的DNN训练作业的最佳作业和GPU级别配置来解决能源问题。Zeus使用在线探索-开发方法，结合及时的能源分析，避免了昂贵的离线测量，并适应时间上的数据漂移。我们的评估结果显示，对于各种工作负载，Zeus可以将DNN训练的能源效率提高15.3％-75.8％。

7. ModelKeeper: Accelerating DNN Training via Automated Training Warmup NSDI'23
1. ModelKeeper是第一个自动训练预热系统，通过重新利用共享集群中以前训练的模型来加速DNN训练。我们的主要见解是，通过转换已经训练过的模型的权重来初始化训练作业的模型可以启动它并减少总训练量。然而，随着时间的推移，提交的模型在其结构和精度上可能会有所不同。给定要训练的新模型，ModelKeeper能够可扩展地识别它与先前训练的模型之间的结构相似性，选择具有高相似性和良好模型精度的父模型，并对权重进行结构感知的转换，以在新模型权重的预热过程中保留来自父模型的最大信息。我们在数千个CV和NLP模型上的评估显示，ModelKeeper以很小的开销和不降低模型精度的情况下，能够实现1.3×–4.3×的更快训练完成。

8. HET: Scaling out Huge Embedding Model Training via Cache-enabled Distributed Framework VLDB'22
1. 嵌入模型一直是高维数据学习范式中的有效方法。然而，嵌入模型的一个开放问题是其表示（潜在因素）通常会导致大量参数空间。我们观察到，现有的分布式训练框架在嵌入模型方面存在可扩展性问题，因为从服务器更新和检索共享嵌入参数通常占据训练周期的主导地位。在本文中，我们提出了HET，一种新的系统框架，它显著提高了大型嵌入模型训练的可扩展性。我们将嵌入数据的偏斜流行度分布视为性能机会，并利用它来通过嵌入缓存解决通信瓶颈。为了确保缓存之间的一致性，我们将一种新的一致性模型纳入HET设计中，它针对每个嵌入提供了细粒度的一致性保证。与仅允许读操作陈旧度的先前工作相比，HET还利用了写操作的陈旧度。在六个代表性任务的评估中，HET实现了高达88%的嵌入通信减少和高达20.68倍的性能提升，超过了最先进的基线模型。

9. Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning OSDI'22
1. Alpa是一种自动化大规模深度学习（DL）模型并行训练的系统，通过生成统一的数据、运算符和管道并行执行计划来实现。现有的模型并行训练系统要么要求用户手动创建并行化计划，要么从有限的模型并行性配置空间中自动生成计划，不足以在分布式计算设备上扩展复杂的DL模型。Alpa通过将并行性视为两个层次：运算符内部并行性和运算符之间并行性，来实现大型DL模型的训练分发。基于此，Alpa构建了一个新的层次空间，用于执行海量模型并行执行计划。Alpa设计了多个编译步骤，以自动推导出每个并行性级别的高效并行执行计划。Alpa实现了一个高效的运行时系统，用于在分布式计算设备上协调两个层次的并行执行。我们的评估表明，即使在手动调整的模型并行训练系统上，Alpa也能生成与之匹配或超越的并行化计划。与专门的系统不同，Alpa还适用于具有异构体系结构和没有手动设计计划的模型。

10. FastMoE: A Fast Mixture-of-Expert Training System arXiv preprint arXiv:2103.13262
1. 训练万亿级的MoE模型需要算法和系统协同设计，以实现高效分布式训练。不幸的是，现有的满足要求的平台强烈依赖于Google的硬件（TPU）和软件（Mesh TensorFlow）栈，并且不向公众开放，特别是GPU和PyTorch社区。
本文介绍了FastMoE，一个基于PyTorch和常见加速器的分布式MoE训练系统。该系统提供了一种分层接口，既可以灵活地设计模型，又可以轻松地适应不同的应用程序，如Transformer-XL和Megatron-LM。与使用PyTorch直接实现MoE模型不同，FastMoE通过复杂的高性能加速技术高度优化了训练速度。该系统支持将不同的专家放置在跨多个节点的多个GPU上，可以使专家的数量线性地扩大。

11. λDNN: Achieving Predictable Distributed DNN Training with Serverless Architectures TC'21
1. “Serverless computing”允许用户将复杂的模型训练分解为多个函数，而无需管理虚拟机或服务器。虽然提供了一个更简单的资源接口（即函数数量和内存大小），但是不充足的函数资源提供（低配或高配）很容易导致无法预测的服务器端 DDNN 训练性能。我们在 AWS Lambda 上的实证研究表明，这种无法预测的服务器端 DDNN 训练性能主要是由参数服务器（PS）资源瓶颈和小的本地批量大小导致的。函数资源配置框架 λλDNN，为无服务器 DDNN 训练工作负载提供可预测的性能，同时节省配置函数的预算。利用 PS 网络带宽和函数 CPU 利用率，我们建立了一个轻量级的分析 DDNN 训练性能模型，以实现我们的 λλDNN 资源配置策略设计，从而保证服务器端函数的 DDNN 训练性能。在 AWS Lambda 上进行的大量原型实验和互补的基于跟踪的模拟表明，与最先进的资源配置策略相比，λλDNN 可以提供可预测的 DDNN 训练性能，并节省高达 66.7% 的函数资源货币成本，同时带来可以接受的运行时开销。

12. STRONGHOLD: Fast and Affordable Billion-scale Deep Learning Model Training SC'22
1. STRONGHOLD可以在不改变用户代码的情况下实现大型DNN模型的训练。STRONGHOLD通过动态将数据卸载到CPU RAM并启用辅助存储来扩展最大可训练模型大小。它自动确定最小量的数据以保持在GPU内存中，以最小化GPU内存使用。与最先进的基于卸载的解决方案相比，STRONGHOLD在32GB V100 GPU上提高了1.9x〜6.Sx的可训练模型大小，并提高了1.2x〜3.7x的训练吞吐量。它已经投入生产中，成功支持大规模的DNN训练。

13. AMP: Automatically Finding Model Parallel Strategies with Heterogeneity Awareness NeurIPS '22
1. AMP是一个自动派生这种策略的框架。AMP识别一个有效的模型并行性策略空间，并通过利用设计用于捕捉模型和集群规格异构性的成本模型，高效地搜索高性能的策略。与现有方法不同，AMP专门针对由不均匀层组成和具有更异构加速器和带宽的集群设置的复杂模型进行支持。我们在公共云中评估了AMP的流行模型和集群设置，并显示AMP返回与专家调整的策略相匹配的并行策略。在具有异构架构的异构集群或模型中，AMP找到的策略比最先进的模型并行系统高1.54倍和1.77倍的吞吐量。

14. Whale: Efficient Giant Model Training over Heterogeneous {GPUs} ATC'22
1. Whale是一个巨型模型分布式训练框架。为了支持各种并行策略及其混合，Whale通过定义两种新的模型注释形式来泛化编程接口，以便包含用户提示。Whale运行时利用这些注释并执行图优化，以将本地深度学习DAG图转换为分布式多GPU执行。Whale还引入了一种新颖的硬件感知并行策略，以平衡方式提高异构GPU上的模型训练性能。在一个拥有512个GPU的生产集群中部署，Whale成功地训练了一个名为M6的行业规模多模态模型，该模型拥有超过一万亿个模型参数，展示了极强的可扩展性和效率。

15. Unity: Accelerating DNN Training Through Joint Optimization of Algebraic Transformations and Parallelization OSDI'22
1. Unity是第一个在分布式DNN训练中同时优化代数变换和并行化的系统。Unity将并行化和代数变换表示为对统一并行计算图（PCG）的替换，该图同时表示分布式DNN训练过程的计算、并行化和通信。通过给定操作符规范的列表自动生成图形替换形式的优化，并使用自动定理证明器形式上验证其正确性。然后，Unity使用一种新的分层搜索算法来共同优化代数变换和并行化，同时保持可扩展性。这些技术的组合提供了一种通用且可扩展的优化分布式DNN训练的方法，可以将新的DNN操作符、并行化策略和模型架构集成到系统中，而仅需进行最少的手动工作。对七个在32个节点上最多使用192个GPU运行的实际DNN进行了Unity评估，并表明Unity在保持优化时间低于20分钟的情况下，比现有的DNN训练框架性能提高了高达3.6倍。Unity可作为开源DNN训练框架FlexFlow的一部分使用，网址为https://github.com/flexflow/flexflow。

16. NASPipe: High Performance and Reproducible Pipeline Parallel Supernet Training via Causal Synchronous Parallelism ASPLOS'22
1. 超网络训练是神经架构搜索中流行且重要的范例。该范例将整个深度神经网络（DNN）架构搜索空间嵌入一个整体超网络中，迭代地激活超网络的子集（即子网络）来适应每个数据批次，并搜索符合特定要求的高质量子网络。虽然在多个GPU上并行训练子网络以加速训练是可取的，但并发的子网络可能会访问相同的DNN层，从而存在竞争风险。现有系统既不支持有效地并行化子网络的训练执行，也不能确定性地解决竞争风险，导致训练过程不可重现并潜在地导致非常重要的准确性损失。
2. 我们提出了NASPipe，这是第一个通过因果同步并行（CSP）流水线调度抽象实现高性能和可重现的分布式超网络训练系统：NASPipe将超网络划分到多个GPU上，并以流水线方式并发执行多个生成的子任务（子网络）；同时，它监控子网络之间的相关性，并确定性地解决由于子网络层共享引起的任何因果依赖。为了获得高性能，NASPipe的CSP调度器利用了这样一个事实：超网络跨度越大，越少的依赖关系在时间上接近的子网络之间表现出来；因此，它积极地安排具有较大时间顺序的子网络进行执行，前提是它们不依赖于未完成的先行子网络。此外，为了减轻GPU内存负担，不必保存整个超网络的参数，NASPipe使用了一种上下文切换技术，将整个超网络存储在CPU内存中，精确预测子网络的调度，并在其执行之前/之后预提取/清除子网络。评估结果表明，NASPipe是唯一保留超网络训练可重现性的系统，同时实现了与三个最近的流水线训练系统（例如GPipe）相当甚至更高的性能（最多可高达7.8倍）。

17. Out-of-order backprop: an effective scheduling technique for deep learning Eurosys'22
1. 神经网络的训练需要大量计算，因此通常使用GPU进行加速。虽然GPU可以提高性能，但在训练过程中GPU的利用率较低。本文提出了一种有效的神经网络训练调度技术——乱序反向传播（ooo back-prop）。通过利用梯度计算的依赖关系，ooo backprop可以重新排序它们的执行，以充分利用GPU资源。我们表明，通过应用ooo backprop和优先处理关键操作，可以普遍提高单个GPU和多个GPU训练中的GPU利用率。我们提出了三种基于ooo backprop的调度算法。对于单个GPU训练，我们使用多流ooo计算进行调度，以掩盖内核启动开销。在数据并行训练中，我们重新排序梯度计算以最大化计算和参数通信的重叠；在管道并行训练中，我们优先处理关键梯度计算以减少管道停顿。我们使用十二个神经网络和五个公共数据集评估了我们的优化算法。与各自最先进的训练系统相比，我们的算法可以将单GPU训练的训练吞吐量提高1.03-1.58倍，数据并行训练的训练吞吐量提高1.10-1.27倍，管道并行训练的训练吞吐量提高1.41-1.99倍。

18. Varuna: Scalable, Low-cost Training of Massive Deep Learning Models Eurosys'22
1. Varuna，这是一个使用通用网络设备训练海量深度学习模型（数十亿个参数）的新系统。现有的深度学习训练系统通常依赖于专用的“超级集群”，这些集群由数百个或数千个GPU组成，并使用专用的高带宽互连技术，例如NV-Link和Infiniband。除了昂贵之外，这种依赖超级集群和定制高速互连的做法还存在以下问题：（a）对作业并行性的可扩展性限制；（b）在超级集群之间分配资源时出现碎片化。
2. 本文介绍了Varuna，这是一个能够在通用网络设备上高效地训练深度学习模型的系统。Varuna能够充分利用网络资源，并自动配置用户的训练作业以有效地利用任何给定的资源。因此，Varuna能够利用成本约为专用GPU的5倍的“低优先级”虚拟机，从而显著降低训练大型模型的成本。我们通过使用5倍更便宜的“spot VMs”来训练包括一个2000亿参数模型在内的大型模型，同时保持高训练吞吐量，证明了Varuna的有效性。Varuna相对于其他模型并行方法，例如基于数据并行和基于管道并行方法，可以将语言模型的端到端训练时间提高多达18倍，并在通用虚拟机上相对于其他管道并行方法提高了多达26%。
3. Varuna的代码可在https://github.com/microsoft/varuna上获得。

19. Megatron-LM SC'21
1. 我们介绍了训练非常大的transformer模型的技术，并实现了一种简单有效的层内模型并行方法，使得可以训练拥有数十亿参数的transformer模型。我们的方法不需要新的编译器或库更改，与pipeline模型并行性是正交且互补的，可以完全通过在本机PyTorch中插入几个通信操作来实现。我们通过使用512个GPU收敛transformer模型来说明这种方法。当与维持39 TeraFLOPs（峰值FLOPs的30％）的强大单GPU基线进行比较时，我们在整个应用程序中维持了15.1 PetaFLOPs的性能，其可扩展性效率为76％。为了证明大型语言模型可以进一步推进最新进展，我们训练了一个与GPT-2类似的83亿参数transformer语言模型和一个与BERT类似的39亿参数模型。我们表明，在类似BERT的模型中仔细考虑层归一化的位置对于实现随着模型大小的增长而增加的性能至关重要。使用GPT-2模型，我们在WikiText103（10.8，与SOTA困惑度的15.8相比）和LAMBADA（66.5％，与SOTA准确率的63.2％相比）数据集上实现了SOTA结果。我们的BERT模型在RACE数据集上实现了SOTA结果（90.9％，与SOTA准确率的89.4％相比）。

20. Chimera: efficiently training large-scale neural networks with bidirectional pipelines SC'21
1. 本文提出了一种新的流水线并行方案Chimera，旨在有效训练大规模深度学习模型。Chimera采用双向流水线，是一种同步的方法，因此不会出现精度损失，这比异步方法更有助于收敛。与最新的同步流水线方法相比，Chimera将泡沫数量降低了50％，并且由于双向流水线的复杂调度，Chimera具有更平衡的激活内存消耗。作者在Transformer基于语言模型上进行了评估。在Piz Daint超级计算机的2048个GPU节点上运行具有13亿参数的GPT-2模型时，Chimera相对于最先进的同步和异步流水线方法，可以提高训练吞吐量1.16倍到2.34倍。

21. Piper: Multidimensional Planner for DNN Parallelization NeurIPS'21
1. 最先进的深度神经网络模型越来越大，导致模型训练所需的计算和存储资源也不断增加，因此开发了许多执行方案，如数据并行、管道模型并行、张量（内层）模型并行和各种节省内存的优化方案。然而，之前的工作都没有解决将DNN计算图优化地分割到多个加速器上，同时结合所有这些并行模式和优化的复杂问题。在这项工作中，我们介绍了Piper，一种有效的优化算法，它基于两级动态规划方法。我们的两级方法是基于这样一个洞察：为每个层次提供张量并行技术（例如，Megatron-LM对变形器层次的切分），可以显著减少搜索空间并使全局问题变得可处理，相比于考虑整个DNN运算符图的张量并行配置。

22. Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training
1. 近年来，Transformer模型的成功推动了深度学习模型参数规模的飞跃。然而，由于单个GPU的内存资源受限，如何选择最佳的并行策略仍然缺乏最佳实践，因为这需要同时具备深度学习和并行计算领域的专业知识。
2. Colossal-AI系统通过引入一个统一的接口来解决上述挑战，以将模型训练的顺序代码扩展到分布式环境中。它支持数据并行、管道并行、张量并行和序列并行等并行训练方法，以及与零冗余优化器集成的异构训练方法。与基线系统相比，Colossal-AI在大规模模型上可以实现高达2.76倍的训练加速。

23. Efficient Sparse Collective Communication and its application to Accelerate Distributed Deep Learning SIGCOMM'21
1. 高效的集体通信对于并行计算应用程序（如分布式训练大规模推荐系统和自然语言处理模型）至关重要。现有的集体通信库专注于针对密集输入优化操作，导致在输入稀疏时传输许多零。这与当前的趋势相反，即大型模型中数据的稀疏性越来越高。
2. 我们提出了OmniReduce，一种利用稀疏性的高效流聚合系统，通过仅发送非零数据块来最大化有效带宽使用。我们证明了这个想法是有益的，并且可以将分布式训练加速高达8.2倍。即使在100 Gbps的情况下，OmniReduce对于网络瓶颈DNN也可以提供1.4-2.9倍的性能改进。

24. PipeTransformer: Automated Elastic Pipelining for Distributed Training of Large-scale Models ICML'21
1. Transformer模型的大小正在以前所未有的速度增长。在发布GPT-3（175B）不到一年的时间内，模型规模已经达到了万亿级别。训练这样的模型需要大量的工程工作和巨大的计算资源，这是大多数研究团队无法承担的奢侈品。在本文中，我们提出了PipeTransformer，它利用自动化和弹性流水线和数据并行化进行Transformer模型的高效分布式训练。PipeTransformer通过在训练期间识别和冻结一些 层，并为训练其余活跃层分配资源，自动调整流水线和数据并行化。具体来说，PipeTransformer动态排除已经收敛的层，将活跃层打包到较少的GPU中，并分叉更多的副本以增加数据并行宽度。我们使用Vision Transformer（ViT）在ImageNet和BERT在GLUE和SQuAD数据集上评估PipeTransformer。结果显示，与最先进的基线相比，PipeTransformer的速度提高了2.4倍。我们还提供了各种性能分析，以更全面地了解我们的算法和系统设计。我们还为PipeTransformer开发了开源的灵活API，提供了在冻结算法、模型定义和训练加速之间的清晰分离，从而允许将其应用于需要类似冻结策略的其他算法。

25. DAPPLE: An Efficient Pipelined Data Parallel Approach for Large Models Training PPOPP'21
1. 在GPU平台上训练大型深度神经网络模型是一项具有挑战性的任务，因为这些平台的互连能力各不相同。最近，管道训练被提出作为提高设备利用率的有效方法。然而，仍然存在几个棘手的问题需要解决：提高计算效率同时确保收敛性，并减少内存使用量而不增加额外的计算成本。本文提出了DAPPLE，一种用于大型DNN模型的数据并行和管道并行的同步训练框架。它采用一种新颖的并行化策略规划器来解决划分和放置问题，并探索数据并行和管道并行的最佳混合策略。我们还提出了一种新的运行时调度算法来减少设备内存使用量，这是与重新计算方法正交的，不会以牺牲训练吞吐量的代价来实现的。实验结果表明，在同步训练场景下，DAPPLE规划器始终优于PipeDream规划器生成的策略，速度最高可提高3.23倍。同时，DAPPLE运行时比GPipe提高了1.6倍的训练吞吐量，并减少了12％的内存消耗。

26. Breaking the Computation and Communication Abstraction Barrier in Distributed Machine Learning Workloads ASPLOS'22
1. 近年来，越来越大的机器学习模型需要分布式训练和推断任务。考虑到训练这些模型的巨大成本，解锁计算和通信优化以获得最佳性能至关重要。然而，当前深度学习框架中计算和通信内核之间的逻辑分离错过了这种跨越的优化机会。打破这种抽象可以提供许多优化，以提高分布式工作负载的性能。手动应用这些优化需要针对每种情况修改底层计算和通信库，这是耗时且容易出错的。
因此，我们提出了 CoCoNeT，其中包含用于表示计算和通信程序的DSL。 CoCoNeT 包含几种机器学习感知的变换，以优化程序，并生成高性能内核的编译器。提供计算和通信作为第一类构造允许用户在高级抽象上工作并应用强大的优化，例如融合或重叠计算和通信。 CoCoNeT 使我们能够使用仅几行代码优化大型语言模型中的数据、模型和管道并行工作负载。实验证明，CoCoNeT 显著优于最先进的分布式机器学习实现。

27. TeraPipe:Large-Scale Language Modeling with Pipeline Parallelism ICML'21
1. “模型并行化已成为训练现代大规模深度语言模型的必要手段。在本文中，我们确定了一个新的、与现有模型并行方法不同的维度：由于Transformer模型的自回归性质，可以在单个训练序列中执行管线并行。这使得与以前的工作相比，可以实现更细粒度的管线。基于这个关键思想，我们设计了TeraPipe，一种高性能的基于令牌的管线并行算法，用于Transformer-based语言模型的同步模型并行训练。我们开发了一种基于动态规划的新算法，以计算给定特定模型和集群配置的最佳流水线执行方案。我们展示了TeraPipe可以将使用48个p3.16xlarge实例的AWS集群在最大的GPT-3模型（1750亿个参数）上训练的速度提高了5.0倍，相比于最先进的模型并行方法。可以在https://此URL找到用于重现的代码。”

28. PipeSwitch: Fast Pipelined Context Switching for Deep Learning Applications OSDI'20
1. DL（深度学习）工作负载包括吞吐量密集的训练任务和延迟敏感的推断任务。目前主要的做法是为训练和推断分别提供专用的GPU集群。由于需要满足严格的服务水平目标（SLO），GPU集群通常是根据峰值负载过度配置的，应用程序和任务类型之间的共享受到限制。
2. 我们提出PipeSwitch，这是一种使推断应用程序未使用的周期可以被训练或其他推断应用程序填充的系统。它允许多个DL应用程序共享GPU，整个GPU内存和毫秒级切换开销。通过引入管道上下文切换，PipeSwitch可以显着提高GPU利用率，而不牺牲SLO。关键思想是利用神经网络模型的分层结构和其逐层计算模式来通过PCIe对模型进行传输并在GPU上执行任务，同时进行模型感知的分组。我们还设计了统一的内存管理和主备工作程序切换机制来配合管道处理，并确保进程级别的隔离。我们已经建立了PipeSwitch原型，并将其与PyTorch集成。对各种DL模型和GPU卡的实验表明，PipeSwitch仅产生3.6-6.6毫秒的任务启动开销和5.4-34.6毫秒的总开销（比NVIDIA MPS快10-50倍），并实现了接近100%的GPU利用率。

29. A Unified Architecture for Accelerating Distributed DNN Training in Heterogeneous GPU/CPU Clusters OSDI'20
1. Data center clusters 既有GPU和CPU进行计算，又有网络带宽进行分布式训练，但是现有的分布式深度神经网络（DNN）训练架构——全约简和参数服务器（PS）并不能充分利用这些异构资源。本文提出了一种名为BytePS的新型分布式DNN训练架构。BytePS可以利用集群中的闲置CPU和带宽资源来加速在GPU上运行的分布式DNN训练任务。它提供了一个被证明为最优的统一通信框架——现有的全约简和PS成为BytePS的两个特例。为了在实践中实现被证明的最优性，BytePS进一步分割参数优化器的功能。它引入了一个称为“Summation Service”的抽象层，用于聚合梯度，对所有优化器都是通用的。Summation Service可以通过AVX指令加速，并且可以在CPU上高效运行，而与DNN模型相关的优化器算法则在GPU上运行以加速计算。BytePS可以加速TensorFlow、PyTorch和MXNet等主要框架的DNN训练。对于最多使用256个GPU的典型DNN训练任务，BytePS的性能比最先进的开源全约简和PS分别提高了最多84%和245%。

30. PipeDream: Pipeline Parallelism for DNN Training SOSP'19
1. PipeDream是一个GPU上的深度神经网络（DNN）训练系统，通过在多台机器之间分阶段执行来并行计算。其管道并行计算模型避免了数据并行训练时遇到的由于大型模型和/或有限网络带宽导致的高通信-计算比率下的减速问题。与数据并行训练相比，PipeDream将大型DNN的通信降低了高达95％，并允许完美重叠通信和计算。PipeDream通过系统地将DNN层分配到所有可用的GPU上以平衡工作并最小化通信，为向后传递正确性版本化模型参数，并以循环方式调度不同输入的正向和反向传递来优化“达到目标准确度所需的时间”。在两个不同的集群上使用五个不同的DNN进行的实验表明，PipeDream在时间到达准确度方面比数据并行训练快了高达5倍。

31. GeePS: Scalable Deep Learning on Distributed GPUs with a GPU-Specialized Parameter Server Eurosys'16
1. 大规模深度学习需要巨大的计算资源来训练多层神经网络。最近的系统提出使用数百到数千台机器来训练具有数十层和数十亿连接的网络。虽然可以使用GPU比传统的CPU内核更高效地进行计算，但在单个GPU上训练这样的网络过于缓慢，而在分布式GPU上训练可能效率不高，因为存在数据移动开销、GPU停顿和有限的GPU内存。本文描述了一种名为GeePS的新型参数服务器，它支持跨多台机器分布式GPU的可扩展深度学习，克服了这些障碍。我们表明，GeePS能够使单节点GPU实现具有良好的可扩展性，例如在16台机器上处理的训练图像数量每秒达到原始优化单节点代码的13倍。此外，GeePS仅使用四个GPU机器就实现了比最先进的仅使用CPU的系统在108个机器上实现的更高的训练吞吐量。

32. SWARM Parallelism: Training Large Models Can Be Surprisingly Communication-Efficient, 27 Jan 2023
1. 许多深度学习应用受益于使用具有数十亿参数的大型模型。训练这些模型因需要专门的高性能计算(HPC)集群而变得非常昂贵。在这项工作中，我们考虑了训练大型模型的替代方案：使用廉价的“可预取”实例或从多个地区汇集现有资源。我们分析了现有模型并行算法在这些条件下的性能，并找到了训练更大模型时通信密集程度较低的配置。基于这些发现，我们提出了SWARM并行算法，这是一种专为连接质量差、异构且不可靠的设备设计的模型并行训练算法。SWARM在节点之间创建临时随机流水线，并在出现故障时重新平衡。我们通过实证验证了我们的发现，并将SWARM并行性与现有大规模训练方法进行了比较。最后，我们将我们的见解与压缩策略相结合，使用预取T4 GPU上的少于200 Mb/s的网络训练了一个具有10亿个共享参数(共享前约为130亿)的大型Transformer语言模型。
