# Neural Networks on Silicon

My name is Fengbin Tu. I'm currently working with Prof. [Yuan Xie](https://www.ece.ucsb.edu/~yuanxie/index.html), as a postdoctoral researcher at the Electrical and Computer Engineering Department, UCSB. Before joining UCSB, I received my Ph.D. degree from the Institute of Microelectronics, Tsinghua University. For more informantion about me and my research, you can go to [my homepage](https://fengbintu.github.io/). One of my research interests is architecture design for deep learning. This is an exciting field where fresh ideas come out every day, so I'm collecting works on related topics. Welcome to join us!

## Table of Contents
 - [My Contributions](#my-contributions)
 - [Conference Papers](#conference-papers)
   - 2014: [ASPLOS](#2014-asplos), [MICRO](#2014-micro)
   - 2015: [ISCA](#2015-isca), [ASPLOS](#2015-asplos), [FPGA](#2015-fpga), [DAC](#2015-dac)
   - 2016: [ISSCC](#2016-isscc), [ISCA](#2016-isca), [MICRO](#2016-micro), [HPCA](#2016-hpca), [DAC](#2016-dac), [FPGA](#2016-fpga), [ICCAD](#2016-iccad), [DATE](#2016-date), [ASPDAC](#2016-aspdac), [VLSI](#2016-vlsi), [FPL](#2016-fpl)
   - 2017: [ISSCC](#2017-isscc), [ISCA](#2017-isca), [MICRO](#2017-micro), [HPCA](#2017-hpca), [ASPLOS](#2017-asplos), [DAC](#2017-dac), [FPGA](#2017-fpga), [ICCAD](#2017-iccad), [DATE](#2017-date), [VLSI](#2017-vlsi), [FCCM](#2017-fccm), [HotChips](#2017-hotchips)
   - 2018: [ISSCC](#2018-isscc), [ISCA](#2018-isca), [MICRO](#2018-micro), [HPCA](#2018-hpca), [ASPLOS](#2018-asplos), [DAC](#2018-dac), [FPGA](#2018-fpga), [ICCAD](#2018-iccad), [DATE](#2018-date), [ASPDAC](#2018-aspdac), [VLSI](#2018-vlsi), [HotChips](#2018-hotchips)
   - 2019: [ISSCC](#2019-isscc), [ISCA](#2019-isca), [MICRO](#2019-micro), [HPCA](#2019-hpca), [ASPLOS](#2019-asplos), [DAC](#2019-dac), [FPGA](#2019-fpga), [ICCAD](#2019-iccad), [ASPDAC](#2019-aspdac), [VLSI](#2019-vlsi), [HotChips](#2019-hotchips), [ASSCC](#2019-asscc)
   - 2020: [ISSCC](#2020-isscc), [ISCA](#2020-isca), [HPCA](#2020-hpca), [ASPLOS](#2020-asplos), [DAC](#2019-dac), [FPGA](#2019-fpga)

 - [Important Topics](#important-topics)
   - [Tutorial and Survey](#tutorial-and-survey)
   - [Benchmarks](#benchmarks)
   - [Network Compression](#network-compression)
   - [Other Topics](#other-topics)

 - [Industry Contributions](#industry-contributions)

## My Contributions
I'm working on energy-efficient architecture design for deep learning. Some featured works are presented here. Hope my new works will come out soon in the near future.

[Jun. 2018] A retention-aware neural acceleration (RANA) framework has been designed, which strengthens DNN accelerators with refresh-optimized eDRAM to save total system energy. RANA includes three techniques from the training, scheduling, architecture levels respectively.
- [**RANA: Towards Efficient Neural Acceleration with Refresh-Optimized Embedded DRAM.**](https://ieeexplore.ieee.org/abstract/document/8416839/) (**ISCA'18**) 
  - **Training Level**: A retention-aware training method is proposed to improve eDRAM's tolerable retention time with no accuracy loss. Bit-level retention errors are injected during training, so the network' s tolerance to retention failures is improved. A higher tolerable failure rate leads to longer tolerable retention time, so more refresh can be removed.
  - **Scheduling Level**: A system energy consumption model is built in consideration of computing energy, on-chip buffer access energy, refresh energy and off-chip memory access energy. RANA schedules networks in a hybrid computation pattern based on this model. Each layer is assigned with the computation pattern that costs the lowest energy.
  - **Architecture Level**: RANA independently disables refresh to eDRAM banks based on their storing data's lifetime, saving more refresh energy. A programmable eDRAM controller is proposed to enable the above fine-grained refresh controls.

[Apr. 2017] A deep convoultional neural network architecture (DNA) has been designed with 1~2 orders higher energy efficiency over the state-of-the-art works. I'm trying to further improve the architecture for ultra low-power compting. 
- [**Deep Convolutional Neural Network Architecture with Reconfigurable Computation Patterns.**](http://ieeexplore.ieee.org/document/7898402/) (**TVLSI popular paper**)
  - This is the first work to assign Input/Output/Weight Reuse to different layers of a CNN, which optimizes system-level energy consumption based on different CONV parameters.
  - A 4-level CONV engine is designed to to support different tiling parameters for higher resource utilization and performance.
  - A layer-based scheduling framework is proposed to optimize both system-level energy efficiency and performance.

## Conference Papers
This is a collection of conference papers that interest me. The emphasis is focused on, but not limited to neural networks on silicon. Papers of significance are marked in **bold**. My comments are marked in *italic*.

### 2014 ASPLOS
- **DianNao: A Small-Footprint High-Throughput Accelerator for Ubiquitous Machine-Learning.** (CAS, Inria)

### 2014 MICRO
- **DaDianNao: A Machine-Learning Supercomputer.** (CAS, Inria, Inner Mongolia University)

### 2015 ISCA
- **ShiDianNao: Shifting Vision Processing Closer to the Sensor.** (CAS, EPFL, Inria)

### 2015 ASPLOS
- **PuDianNao: A Polyvalent Machine Learning Accelerator.** (CAS, USTC, Inria)

### 2015 FPGA
- **Optimizing FPGA-based Accelerator Design for Deep Convolutional Neural Networks.** (Peking University, UCLA)

### 2015 DAC
- Reno: A Highly-Efficient Reconfigurable Neuromorphic Computing Accelerator Design. (Universtiy of Pittsburgh, Tsinghua University, San Francisco State University, Air Force Research Laboratory, University of Massachusetts.)
- Scalable Effort Classifiers for Energy Efficient Machine Learning. (Purdue University, Microsoft Research)
- Design Methodology for Operating in Near-Threshold Computing (NTC) Region. (AMD)
- Opportunistic Turbo Execution in NTC: Exploiting the Paradigm Shift in Performance Bottlenecks. (Utah State University)

### 2016 DAC
- **DeepBurning: Automatic Generation of FPGA-based Learning Accelerators for the Neural Network Family.** (Chinese Academy of Sciences)
  - *Hardware generator: Basic buliding blocks for neural networks, and address generation unit (RTL).*
  - *Compiler: Dynamic control flow (configurations for different models), and data layout in memory.*
  - *Simply report their framework and describe some stages.*
- **C-Brain: A Deep Learning Accelerator that Tames the Diversity of CNNs through Adaptive Data-Level Parallelization.** (Chinese Academy of Sciences)
- **Simplifying Deep Neural Networks for Neuromorphic Architectures.** (Incheon National University)
- **Dynamic Energy-Accuracy Trade-off Using Stochastic Computing in Deep Neural Networks.** (Samsung, Seoul National University, Ulsan National Institute of Science and Technology)
- **Optimal Design of JPEG Hardware under the Approximate Computing Paradigm.** (University of Minnesota, TAMU)
- Perform-ML: Performance Optimized Machine Learning by Platform and Content Aware Customization. (Rice University, UCSD)
- Low-Power Approximate Convolution Computing Unit with Domain-Wall Motion Based “Spin-Memristor” for Image Processing Applications. (Purdue University)
- Cross-Layer Approximations for Neuromorphic Computing: From Devices to Circuits and Systems. (Purdue University)
- Switched by Input: Power Efficient Structure for RRAM-based Convolutional Neural Network. (Tsinghua University)
- A 2.2 GHz SRAM with High Temperature Variation Immunity for Deep Learning Application under 28nm. (UCLA, Bell Labs)

### 2016 ISSCC
- **A 1.42TOPS/W Deep Convolutional Neural Network Recognition Processor for Intelligent IoE Systems.** (KAIST)
- **Eyeriss: An Energy-Efficient Reconfigurable Accelerator for Deep Convolutional Neural Networks.** (MIT, NVIDIA)
- A 126.1mW Real-Time Natural UI/UX Processor with Embedded Deep Learning Core for Low-Power Smart Glasses Systems. (KAIST)
- A 502GOPS and 0.984mW Dual-Mode ADAS SoC with RNN-FIS Engine for Intention Prediction in Automotive Black-Box System. (KAIST)
- A 0.55V 1.1mW Artificial-Intelligence Processor with PVT Compensation for Micro Robots. (KAIST)
- A 4Gpixel/s 8/10b H.265/HEVC Video Decoder Chip for 8K Ultra HD Applications. (Waseda University)

### 2016 ISCA
 - **Cnvlutin: Ineffectual-Neuron-Free Deep Convolutional Neural Network Computing.** (University of Toronto, University of British Columbia)
 - **EIE: Efficient Inference Engine on Compressed Deep Neural Network.** (Stanford University, Tsinghua University)
 - **Minerva: Enabling Low-Power, High-Accuracy Deep Neural Network Accelerators.** (Harvard University)
 - **Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for Convolutional Neural Networks.** (MIT, NVIDIA)
   - *Present an energy analysis framework.*
   - *Propose an energy-efficienct dataflow called Row Stationary, which considers three levels of reuse.*
 - **Neurocube: A Programmable Digital Neuromorphic Architecture with High-Density 3D Memory.** (Georgia Institute of Technology, SRI International)
   - *Propose an architecture integrated in 3D DRAM, with a mesh-like NOC in the logic layer.*
   - *Detailedly describe the data movements in the NOC.*
 - ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars. (University of Utah, HP Labs)
   - *An advance over ISAAC has been published in "Newton: Gravitating Towards the Physical Limits of Crossbar Acceleration" (IEEE Micro).*
 - A Novel Processing-in-memory Architecture for Neural Network Computation in ReRAM-based Main Memory. (UCSB, HP Labs, NVIDIA, Tsinghua University)
 - RedEye: Analog ConvNet Image Sensor Architecture for Continuous Mobile Vision. (Rice University)
 - Cambricon: An Instruction Set Architecture for Neural Networks. (Chinese Academy of Sciences, UCSB)

### 2016 DATE
- **The Neuro Vector Engine: Flexibility to Improve Convolutional Network Efficiency for Wearable Vision.** (Eindhoven University of Technology, Soochow University, TU Berlin)
  - *Propose an SIMD accelerator for CNN.*
- **Efficient FPGA Acceleration of Convolutional Neural Networks Using Logical-3D Compute Array.** (UNIST, Seoul National University)
  - *The compute tile is organized on 3 dimensions: Tm, Tr, Tc.*
- NEURODSP: A Multi-Purpose Energy-Optimized Accelerator for Neural Networks. (CEA LIST)
- MNSIM: Simulation Platform for Memristor-Based Neuromorphic Computing System. (Tsinghua University, UCSB, Arizona State University)
- Accelerated Artificial Neural Networks on FPGA for Fault Detection in Automotive Systems. (Nanyang Technological University, University of Warwick)
- Significance Driven Hybrid 8T-6T SRAM for Energy-Efficient Synaptic Storage in Artificial Neural Networks. (Purdue University)

### 2016 FPGA
- **Going Deeper with Embedded FPGA Platform for Convolutional Neural Network.** \[[Slides](http://www.isfpga.org/fpga2016/index_files/Slides/1_2.pdf)\]\[[Demo](http://www.isfpga.org/fpga2016/index_files/Slides/1_2_demo.m4v)\] (Tsinghua University, MSRA)
  - *The first work I see, which runs the entire flow of CNN, including both CONV and FC layers.*
  - *Point out that CONV layers are computational-centric, while FC layrers are memory-centric.*
  - *The FPGA runs VGG16-SVD without reconfiguring its resources, but the convolver can only support k=3.*
  - *Dynamic-precision data quantization is creative, but not implemented on hardware.*
- **Throughput-Optimized OpenCL-based FPGA Accelerator for Large-Scale Convolutional Neural Networks.** \[[Slides](http://www.isfpga.org/fpga2016/index_files/Slides/1_1.pdf)\] (Arizona State Univ, ARM)
  - *Spatially allocate FPGA's resources to CONV/POOL/NORM/FC layers.*

### 2016 ASPDAC
- **Design Space Exploration of FPGA-Based Deep Convolutional Neural Networks.** (UC Davis)
- **LRADNN: High-Throughput and Energy-Efficient Deep Neural Network Accelerator using Low Rank Approximation.** (Hong Kong University of Science and Technology, Shanghai Jiao Tong University)
- **Efficient Embedded Learning for IoT Devices.** (Purdue University)
- ACR: Enabling Computation Reuse for Approximate Computing. (Chinese Academy of Sciences)

### 2016 VLSI
- **A 0.3‐2.6 TOPS/W Precision‐Scalable Processor for Real‐Time Large‐Scale ConvNets.** (KU Leuven)
  - *Use dynamic precision for different CONV layers, and scales down the MAC array's supply voltage at lower precision.*
  - *Prevent memory fetches and MAC operations based on the ReLU sparsity.*
- **A 1.40mm2 141mW 898GOPS Sparse Neuromorphic Processor in 40nm CMOS.** (University of Michigan)
- A 58.6mW Real-Time Programmable Object Detector with Multi-Scale Multi-Object Support Using Deformable Parts Model on 1920x1080 Video at 30fps. (MIT)
- A Machine-learning Classifier Implemented in a Standard 6T SRAM Array. (Princeton)

### 2016 ICCAD
- **Efficient Memory Compression in Deep Neural Networks Using Coarse-Grain Sparsification for Speech Applications.** (Arizona State University)
- **Memsqueezer: Re-architecting the On-chip memory Sub-system of Deep Learning Accelerator for Embedded Devices.** (Chinese Academy of Sciences)
- **Caffeine: Towards Uniformed Representation and Acceleration for Deep Convolutional Neural Networks.** (Peking University, UCLA, Falcon)
  - *Propose a uniformed convolutional matrix-multiplication representation for accelerating CONV and FC layers on FPGA.*
  - *Propose a weight-major convolutional mapping method for FC layers, which has good data reuse, DRAM access burst length and effective bandwidth.*
- **BoostNoC: Power Efficient Network-on-Chip Architecture for Near Threshold Computing.** (Utah State University)
- Design of Power-Efficient Approximate Multipliers for Approximate Artificial Neural Network. (Brno University of Technology, Brno University of Technology)
- Neural Networks Designing Neural Networks: Multi-Objective Hyper-Parameter Optimization. (McGill University)

### 2016 MICRO
- **From High-Level Deep Neural Models to FPGAs.** (Georgia Institute of Technology, Intel)
  - *Develop a macro dataflow ISA for DNN accelerators.*
  - *Develop hand-optimized template designs that are scalable and highly customizable.*
  - *Provide a Template Resource Optimization search algorithm to co-optimize the accelerator architecture and scheduling.*
- **vDNN: Virtualized Deep Neural Networks for Scalable, Memory-Efficient Neural Network Design.** (NVIDIA)
- **Stripes: Bit-Serial Deep Neural Network Computing.** (University of Toronto, University of British Columbia)
  - *Introduce serial computation and reduced precision computation to neural network accelerator designs, enabling accuracy vs. performance trade-offs.*
  - *Design a bit-serial computing unit to enable linear scaling the performance with precision reduction.*
- **Cambricon-X: An Accelerator for Sparse Neural Networks.** (Chinese Academy of Sciences)
- **NEUTRAMS: Neural Network Transformation and Co-design under Neuromorphic Hardware Constraints.** (Tsinghua University, UCSB)
- **Fused-Layer CNN Accelerators.** (Stony Brook University)
  - *Fuse multiple CNN layers (CONV+POOL) to reduce DRAM access for input/output data.*
- **Bridging the I/O Performance Gap for Big Data Workloads: A New NVDIMM-based Approach.** (The Hong Kong Polytechnic University, NSF/University of Florida)
- **A Patch Memory System For Image Processing and Computer Vision.** (NVIDIA)
- **An Ultra Low-Power Hardware Accelerator for Automatic Speech Recognition.** (Universitat Politecnica de Catalunya)
- Perceptron Learning for Reuse Prediction. (TAMU, Intel Labs)
  - *Train neural networks to predict reuse of cache blocks.*
- A Cloud-Scale Acceleration Architecture. (Microsoft Research)
- Reducing Data Movement Energy via Online Data Clustering and Encoding. (University of Rochester)
- The Microarchitecture of a Real-time Robot Motion Planning Accelerator. (Duke University)
- Chameleon: Versatile and Practical Near-DRAM Acceleration Architecture for Large Memory Systems. (UIUC, Seoul National University)

### 2016 FPL
- **A High Performance FPGA-based Accelerator for Large-Scale Convolutional Neural Network.** (Fudan University)
- **Overcoming Resource Underutilization in Spatial CNN Accelerators.** (Stony Brook University)
  - *Build multiple accelerators, each specialized for specific CNN layers, instead of a single accelerator with uniform tiling parameters.*
- **Accelerating Recurrent Neural Networks in Analytics Servers: Comparison of FPGA, CPU, GPU, and ASIC.** (Intel)

### 2016 HPCA
- **A Performance Analysis Framework for Optimizing OpenCL Applications on FPGAs.** (Nanyang Technological University, HKUST, Cornell University)
- **TABLA: A Unified Template-based Architecture for Accelerating Statistical Machine Learning.** (Georgia Institute of Technology)
- Memristive Boltzmann Machine: A Hardware Accelerator for Combinatorial Optimization and Deep Learning. (University of Rochester)

### 2017 FPGA
- **An OpenCL Deep Learning Accelerator on Arria 10.** (Intel)
  - *Minimum bandwidth requirement: All the intermediate data in AlexNet's CONV layers are cached in the on-chip buffer, so their architecture is compute-bound.*
  - *Reduced operations: Winograd transformation.*
  - *High usage of the available DSPs+Reduced computation -> Higher performance on FPGA -> Competitive efficiency vs. TitanX.*
- **ESE: Efficient Speech Recognition Engine for Compressed LSTM on FPGA.** (Stanford University, DeepPhi, Tsinghua University, NVIDIA)
- **FINN: A Framework for Fast, Scalable Binarized Neural Network Inference.** (Xilinx, Norwegian University of Science and Technology, University of Sydney)
- **Can FPGA Beat GPUs in Accelerating Next-Generation Deep Neural Networks?** (Intel)
- **Accelerating Binarized Convolutional Neural Networks with Software-Programmable FPGAs.** (Cornell University, UCLA, UCSD)
- **Improving the Performance of OpenCL-based FPGA Accelerator for Convolutional Neural Network.** (UW-Madison)
- **Frequency Domain Acceleration of Convolutional Neural Networks on CPU-FPGA Shared Memory System.** (USC)
- **Optimizing Loop Operation and Dataflow in FPGA Acceleration of Deep Convolutional Neural Networks.** (Arizona State University)

### 2017 ISSCC
- **A 2.9TOPS/W Deep Convolutional Neural Network SoC in FD-SOI 28nm for Intelligent Embedded Systems.** (ST)
- **DNPU: An 8.1TOPS/W Reconfigurable CNN-RNN Processor for General Purpose Deep Neural Networks.** (KAIST)
- **ENVISION: A 0.26-to-10TOPS/W Subword-Parallel Computational Accuracy-Voltage-Frequency-Scalable Convolutional Neural Network Processor in 28nm FDSOI.** (KU Leuven)
- **A 288µW Programmable Deep-Learning Processor with 270KB On-Chip Weight Storage Using Non-Uniform Memory Hierarchy for Mobile Intelligence.** (University of Michigan, CubeWorks)
- A 28nm SoC with a 1.2GHz 568nJ/Prediction Sparse Deep-NeuralNetwork Engine with >0.1 Timing Error Rate Tolerance for IoT Applications. (Harvard)
- A Scalable Speech Recognizer with Deep-Neural-Network Acoustic Models and Voice-Activated Power Gating (MIT)
- A 0.62mW Ultra-Low-Power Convolutional-Neural-Network Face Recognition Processor and a CIS Integrated with Always-On Haar-Like Face Detector. (KAIST)

### 2017 HPCA
- **FlexFlow: A Flexible Dataflow Accelerator Architecture for Convolutional Neural Networks.** (Chinese Academy of Sciences)
- **PipeLayer: A Pipelined ReRAM-Based Accelerator for Deep Learning.** (University of Pittsburgh, University of Southern California)
- Towards Pervasive and User Satisfactory CNN across GPU Microarchitectures. (University of Florida)
  - *Satisfaction of CNN (SoC) is the combination of SoCtime, SoCaccuracy and energy consumption.*
  - *The P-CNN framework is composed of offline compilation and run-time management.*
    - *Offline compilation: Generally optimizes runtime, and generates scheduling configurations for the run-time stage.*
    - *Run-time management: Generates tuning tables through accuracy tuning, and calibrate accuracy+runtime (select the best tuning table) during the long-term execution.*
- Supporting Address Translation for Accelerator-Centric Architectures. (UCLA)

### 2017 ASPLOS
- **Tetris: Scalable and Efficient Neural Network Acceleration with 3D Memory.** (Stanford University)
  - *Move accumulation operations close to the DRAM banks.*
  - *Develop a hybrid partitioning scheme that parallelizes the NN computations over multiple accelerators.*
- SC-DCNN: Highly-Scalable Deep Convolutional Neural Network using Stochastic Computing. (Syracuse University, USC, The City College of New York)

### 2017 ISCA
- **Maximizing CNN Accelerator Efficiency Through Resource Partitioning.** (Stony Brook University)
  - *An Extension of their FPL'16 paper.*
- **In-Datacenter Performance Analysis of a Tensor Processing Unit.** (Google)
- **SCALEDEEP: A Scalable Compute Architecture for Learning and Evaluating Deep Networks.** (Purdue University, Intel)
  - *Propose a full-system (server node) architecture, focusing on the challenge of DNN training (intra and inter-layer heterogeneity).*
- **SCNN: An Accelerator for Compressed-sparse Convolutional Neural Networks.** (NVIDIA, MIT, UC Berkeley, Stanford University)
- **Scalpel: Customizing DNN Pruning to the Underlying Hardware Parallelism.** (University of Michigan, ARM)
- Understanding and Optimizing Asynchronous Low-Precision Stochastic Gradient Descent. (Stanford)
- LogCA: A High-Level Performance Model for Hardware Accelerators. (AMD, University of Wisconsin-Madison)
- APPROX-NoC: A Data Approximation Framework for Network-On-Chip Architectures. (TAMU)

### 2017 FCCM
- **Escher: A CNN Accelerator with Flexible Buffering to Minimize Off-Chip Transfer.** (Stony Brook University)
- **Customizing Neural Networks for Efficient FPGA Implementation.**
- **Evaluating Fast Algorithms for Convolutional Neural Networks on FPGAs.**
- **FP-DNN: An Automated Framework for Mapping Deep Neural Networks onto FPGAs with RTL-HLS Hybrid Templates.** (Peking University, HKUST, MSRA, UCLA)
  - *Compute-instensive part: RTL-based generalized matrix multiplication kernel.*
  - *Layer-specific part: HLS-based control logic.*
  - *Memory-instensive part: Several techniques for lower DRAM bandwidth requirements.*
- FPGA accelerated Dense Linear Machine Learning: A Precision-Convergence Trade-off.
- A Configurable FPGA Implementation of the Tanh Function using DCT Interpolation.

### 2017 DAC
- **Deep^3: Leveraging Three Levels of Parallelism for Efficient Deep Learning.** (UCSD, Rice)
- **Real-Time meets Approximate Computing: An Elastic Deep Learning Accelerator Design with Adaptive Trade-off between QoS and QoR.** (CAS)
  - *I'm not sure whether the proposed tuning scenario and direction are reasonable enough to find out feasible solutions.*
- **Exploring Heterogeneous Algorithms for Accelerating Deep Convolutional Neural Networks on FPGAs.** (PKU, CUHK, SenseTime)
- **Hardware-Software Codesign of Highly Accurate, Multiplier-free Deep Neural Networks.** (Brown University)
- **A Kernel Decomposition Architecture for Binary-weight Convolutional Neural Networks.** (KAIST)
- **Design of An Energy-Efficient Accelerator for Training of Convolutional Neural Networks using Frequency-Domain Computation.** (Georgia Tech)
- **New Stochastic Computing Multiplier and Its Application to Deep Neural Networks.** (UNIST)
- **TIME: A Training-in-memory Architecture for Memristor-based Deep Neural Networks.** (THU, UCSB)
- **Fault-Tolerant Training with On-Line Fault Detection for RRAM-Based Neural Computing Systems.** (THU, Duke)
- **Automating the systolic array generation and optimizations for high throughput convolution neural network.** (PKU, UCLA, Falcon)
- **Towards Full-System Energy-Accuracy Tradeoffs: A Case Study of An Approximate Smart Camera System.** (Purdue)
  - *Synergistically tunes componet-level approximation knobs to achieve system-level energy-accuracy tradeoffs.*
- **Error Propagation Aware Timing Relaxation For Approximate Near Threshold Computing.** (KIT)
- RESPARC: A Reconfigurable and Energy-Efficient Architecture with Memristive Crossbars for Deep Spiking Neural Networks. (Purdue)
- Rescuing Memristor-based Neuromorphic Design with High Defects. (University of Pittsburgh, HP Lab, Duke)
- Group Scissor: Scaling Neuromorphic Computing Design to Big Neural Networks. (University of Pittsburgh, Duke)
- Towards Aging-induced Approximations. (KIT, UT Austin)
- SABER: Selection of Approximate Bits for the Design of Error Tolerant Circuits. (University of Minnesota, TAMU)
- On Quality Trade-off Control for Approximate Computing using Iterative Training. (SJTU, CUHK)

### 2017 DATE
- **DVAFS: Trading Computational Accuracy for Energy Through Dynamic-Voltage-Accuracy-Frequency-Scaling.** (KU Leuven)
- **Accelerator-friendly Neural-network Training: Learning Variations and Defects in RRAM Crossbar.** (Shanghai Jiao Tong University, University of Pittsburgh, Lynmax Research)
- **A Novel Zero Weight/Activation-Aware Hardware Architecture of Convolutional Neural Network.** (Seoul National University)
  - *Solve the zero-induced load imbalance problem.*
- **Understanding the Impact of Precision Quantization on the Accuracy and Energy of Neural Networks.** (Brown University)
- **Design Space Exploration of FPGA Accelerators for Convolutional Neural Networks.** (Samsung, UNIST, Seoul National University)
- **MoDNN: Local Distributed Mobile Computing System for Deep Neural Network.** (University of Pittsburgh, George Mason University, University of Maryland)
- **Chain-NN: An Energy-Efficient 1D Chain Architecture for Accelerating Deep Convolutional Neural Networks.** (Waseda University)
- **LookNN: Neural Network with No Multiplication.** (UCSD)
  - *Cluster weights and use LUT to avoid multiplication.*
- Energy-Efficient Approximate Multiplier Design using Bit Significance-Driven Logic Compression. (Newcastle University)
- Revamping Timing Error Resilience to Tackle Choke Points at NTC Systems. (Utah State University)

### 2017 VLSI
- **A 3.43TOPS/W 48.9pJ/Pixel 50.1nJ/Classification 512 Analog Neuron Sparse Coding Neural Network with On-Chip Learning and Classification in 40nm CMOS.** (University of Michigan, Intel)
- **BRein Memory: A 13-Layer 4.2 K Neuron/0.8 M Synapse Binary/Ternary Reconfigurable In-Memory Deep Neural Network Accelerator in 65 nm CMOS.** (Hokkaido University, Tokyo Institute of Technology, Keio University)
- **A 1.06-To-5.09 TOPS/W Reconfigurable Hybrid-Neural-Network Processor for Deep Learning Applications.** (Tsinghua University)
- **A 127mW 1.63TOPS sparse spatio-temporal cognitive SoC for action classification and motion tracking in videos.** (University of Michigan)

### 2017 ICCAD
- **AEP: An Error-bearing Neural Network Accelerator for Energy Efficiency and Model Protection.** (University of Pittsburgh)
- VoCaM: Visualization oriented convolutional neural network acceleration on mobile system. (George Mason University, Duke)
- AdaLearner: An Adaptive Distributed Mobile Learning System for Neural Networks. (Duke)
- MeDNN: A Distributed Mobile System with Enhanced Partition and Deployment for Large-Scale DNNs. (Duke)
- TraNNsformer: Neural Network Transformation for Memristive Crossbar based Neuromorphic System Design. (Purdue).
- A Closed-loop Design to Enhance Weight Stability of Memristor Based Neural Network Chips. (Duke)
- Fault injection attack on deep neural network. (CUHK)
- ORCHARD: Visual Object Recognition Accelerator Based on Approximate In-Memory Processing. (UCSD)

### 2017 HotChips
- **A Dataflow Processing Chip for Training Deep Neural Networks.** (Wave Computing)
- **Brainwave: Accelerating Persistent Neural Networks at Datacenter Scale.** (Microsoft)
- **DNN ENGINE: A 16nm Sub-uJ Deep Neural Network Inference Accelerator for the Embedded Masses.** (Harvard, ARM)
- **DNPU: An Energy-Efficient Deep Neural Network Processor with On-Chip Stereo Matching.** (KAIST)
- **Evaluation of the Tensor Processing Unit (TPU): A Deep Neural Network Accelerator for the Datacenter.** (Google)
- NVIDIA’s Volta GPU: Programmability and Performance for GPU Computing. (NVIDIA)
- Knights Mill: Intel Xeon Phi Processor for Machine Learning. (Intel)
- XPU: A programmable FPGA Accelerator for diverse workloads. (Baidu)

### 2017 MICRO
- **Bit-Pragmatic Deep Neural Network Computing.** (NVIDIA, University of Toronto)
- **CirCNN: Accelerating and Compressing Deep Neural Networks Using Block-Circulant Weight Matrices.** (Syracuse University, City University of New York, USC, California State University, Northeastern University)
- **DRISA: A DRAM-based Reconfigurable In-Situ Accelerator.** (UCSB, Samsung)
- **Scale-Out Acceleration for Machine Learning.** (Georgia Tech, UCSD)
  - Propose CoSMIC, a full computing stack constituting language, compiler, system software, template architecture, and circuit generators, that enable programmable acceleration of learning at scale.
- DeftNN: Addressing Bottlenecks for DNN Execution on GPUs via Synapse Vector Elimination and Near-compute Data Fission. (Univ. of Michigan, Univ. of Nevada)
- Data Movement Aware Computation Partitioning. (PSU, TOBB University of Economics and Technology)
  - *Partition computation on a manycore system for near data processing.*

### 2018 ASPDAC
- **ReGAN: A Pipelined ReRAM-Based Accelerator for Generative Adversarial Networks.** (University of Pittsburgh, Duke)
- **Accelerator-centric Deep Learning Systems for Enhanced Scalability, Energy-efficiency, and Programmability.** (POSTECH)
- **Architectures and Algorithms for User Customization of CNNs.** (Seoul National University, Samsung)
- **Optimizing FPGA-based Convolutional Neural Networks Accelerator for Image Super-Resolution.** (Sogang University)
- **Running sparse and low-precision neural network: when algorithm meets hardware.** (Duke)

### 2018 ISSCC
- **A 55nm Time-Domain Mixed-Signal Neuromorphic Accelerator with Stochastic Synapses and Embedded Reinforcement Learning for Autonomous Micro-Robots.** (Georgia Tech)
- **A Shift Towards Edge Machine-Learning Processing.** (Google)
- **QUEST: A 7.49TOPS Multi-Purpose Log-Quantized DNN Inference Engine Stacked on 96MB 3D SRAM Using Inductive-Coupling Technology in 40nm CMOS.** (Hokkaido University, Ultra Memory, Keio University)
- **UNPU: A 50.6TOPS/W Unified Deep Neural Network Accelerator with 1b-to-16b Fully-Variable Weight Bit-Precision.** (KAIST)
- **A 9.02mW CNN-Stereo-Based Real-Time 3D Hand-Gesture Recognition Processor for Smart Mobile Devices.** (KAIST)
- **An Always-On 3.8μJ/86% CIFAR-10 Mixed-Signal Binary CNN Processor with All Memory on Chip in 28nm CMOS.** (Stanford, KU Leuven)
- **Conv-RAM: An Energy-Efficient SRAM with Embedded Convolution Computation for Low-Power CNN-Based Machine Learning Applications.** (MIT)
- **A 42pJ/Decision 3.12TOPS/W Robust In-Memory Machine Learning Classifier with On-Chip Training.** (UIUC)
- **Brain-Inspired Computing Exploiting Carbon Nanotube FETs and Resistive RAM: Hyperdimensional Computing Case Study.** (Stanford, UC Berkeley, MIT)
- **A 65nm 1Mb Nonvolatile Computing-in-Memory ReRAM Macro with Sub-16ns Multiply-and-Accumulate for Binary DNN AI Edge Processors.** (NTHU)
- **A 65nm 4Kb Algorithm-Dependent Computing-in-Memory SRAM Unit Macro with 2.3ns and 55.8TOPS/W Fully Parallel Product-Sum Operation for Binary DNN Edge Processors.** (NTHU, TSMC, UESTC, ASU)
- **A 1μW Voice Activity Detector Using Analog Feature Extraction and Digital Deep Neural Network.** (Columbia University)

### 2018 HPCA
- **Making Memristive Neural Network Accelerators Reliable.** (University of Rochester)
- **Towards Efficient Microarchitectural Design for Accelerating Unsupervised GAN-based Deep Learning.** (University of Florida)
- **Compressing DMA Engine: Leveraging Activation Sparsity for Training Deep Neural Networks.** (POSTECH, NVIDIA, UT-Austin)
- **In-situ AI: Towards Autonomous and Incremental Deep Learning for IoT Systems.** (University of Florida, Chongqing University, Capital Normal University)
- RC-NVM: Enabling Symmetric Row and Column Memory Accesses for In-Memory Databases. (PKU, NUDT, Duke, UCLA, PSU)
- GraphR: Accelerating Graph Processing Using ReRAM. (Duke, USC, Binghamton University SUNY)
- GraphP: Reducing Communication of PIM-based Graph Processing with Efficient Data Partition. (THU, USC, Stanford)
- PM3: Power Modeling and Power Management for Processing-in-Memory. (PKU)

### 2018 ASPLOS
- **Bridging the Gap Between Neural Networks and Neuromorphic Hardware with A Neural Network Compiler.** (Tsinghua, UCSB)
- **MAERI: Enabling Flexible Dataflow Mapping over DNN Accelerators via Reconfigurable Interconnects.** (Georgia Tech)
  - *Higher PE utilization: Use an augmented reduction tree (reconfigurable interconnects) to construct arbitrary sized virtual neurons.*
- **VIBNN: Hardware Acceleration of Bayesian Neural Networks.** (Syracuse University, USC)
- Exploiting Dynamical Thermal Energy Harvesting for Reusing in Smartphone with Mobile Applications. (Guizhou University, University of Florida)
- Potluck: Cross-application Approximate Deduplication for Computation-Intensive Mobile Applications. (Yale)

### 2018 VLSI
- **STICKER: A 0.41‐62.1 TOPS/W 8bit Neural Network Processor with Multi‐Sparsity Compatible Convolution Arrays and Online Tuning Acceleration for Fully Connected Layers.** (THU)
- **2.9TOPS/W Reconfigurable Dense/Sparse Matrix‐Multiply Accelerator with Unified INT8/INT16/FP16 Datapath in 14nm Tri‐gate CMOS.** (Intel)
- **A Scalable Multi‐TeraOPS Deep Learning Processor Core for AI Training and Inference.** (IBM)
- **An Ultra‐high Energy‐efficient reconfigurable Processor for Deep Neural Networks with Binary/Ternary Weights in 28nm CMOS.** (THU)
- **B‐Face: 0.2 mW CNN‐Based Face Recognition Processor with Face Alignment for Mobile User Identification.** (KAIST)
- **A 141 uW, 2.46 pJ/Neuron Binarized Convolutional Neural Network based Self-learning Speech Recognition Processor in 28nm CMOS.** (THU)
- **A Mixed‐Signal Binarized Convolutional‐Neural-Network Accelerator Integrating Dense Weight Storage and Multiplication for Reduced Data Movement.** (Princeton)
- **PhaseMAC: A 14 TOPS/W 8bit GRO based Phase Domain MAC Circuit for In‐Sensor‐Computed Deep Learning Accelerators.** (Toshiba)

### 2018 FPGA
- **C-LSTM: Enabling Efficient LSTM using Structured Compression Techniques on FPGAs.** (Peking Univ, Syracuse Univ, CUNY)
- **DeltaRNN: A Power-efficient Recurrent Neural Network Accelerator.** (ETHZ, BenevolentAI)
- **Towards a Uniform Template-based Architecture for Accelerating 2D and 3D CNNs on FPGA.** (National Univ of Defense Tech)
- **A Customizable Matrix Multiplication Framework for the Intel HARPv2 Xeon+FPGA Platform - A Deep Learning Case Study.** (The Univ of Sydney, Intel)
- **A Framework for Generating High Throughput CNN Implementations on FPGAs.** (USC)
- Liquid Silicon: A Data-Centric Reconfigurable Architecture enabled by RRAM Technology. (UW Madison)

### 2018 ISCA
- **RANA: Towards Efficient Neural Acceleration with Refresh-Optimized Embedded DRAM.** (THU)
- **Brainwave: A Configurable Cloud-Scale DNN Processor for Real-Time AI.** (Microsoft)
- **PROMISE: An End-to-End Design of a Programmable Mixed-Signal Accelerator for Machine Learning Algorithms.** (UIUC)
- **Computation Reuse in DNNs by Exploiting Input Similarity.** (UPC)
- **GANAX: A Unified SIMD-MIMD Acceleration for Generative Adversarial Network.** (Georiga Tech, IPM, Qualcomm, UCSD, UIUC)
- **SnaPEA: Predictive Early Activation for Reducing Computation in Deep Convolutional Neural Networks.** (UCSD, Georgia Tech, Qualcomm)
- **UCNN: Exploiting Computational Reuse in Deep Neural Networks via Weight Repetition.** (UIUC, NVIDIA)
- **An Energy-Efficient Neural Network Accelerator based on Outlier-Aware Low Precision Computation.** (Seoul National)
- **Prediction based Execution on Deep Neural Networks.** (Florida)
- **Bit Fusion: Bit-Level Dynamically Composable Architecture for Accelerating Deep Neural Networks.** (Georgia Tech, ARM, UCSD)
- **Gist: Efficient Data Encoding for Deep Neural Network Training.** (Michigan, Microsoft, Toronto)
- **The Dark Side of DNN Pruning.** (UPC)
- **Neural Cache: Bit-Serial In-Cache Acceleration of Deep Neural Networks.** (Michigan)
- EVA^2: Exploiting Temporal Redundancy in Live Computer Vision. (Cornell)
- Euphrates: Algorithm-SoC Co-Design for Low-Power Mobile Continuous Vision. (Rochester, Georgia Tech, ARM)
- Feature-Driven and Spatially Folded Digital Neurons for Efficient Spiking Neural Network Simulations. (POSTECH/Berkeley, Seoul National)
- Space-Time Algebra: A Model for Neocortical Computation. (Wisconsin)
- Scaling Datacenter Accelerators With Compute-Reuse Architectures. (Princeton)
   - *Add a NVM-based storage layer to the accelerator, for computation reuse.*
- Enabling Scientific Computing on Memristive Accelerators. (Rochester)

### 2018 DATE
- **MATIC: Learning Around Errors for Efficient Low-Voltage Neural Network Accelerators.** (University of Washington)
   - *Learn around errors resulting from SRAM voltage scaling, demonstrated on a fabricated 65nm test chip.*
- **Maximizing System Performance by Balancing Computation Loads in LSTM Accelerators.** (POSTECH)
   - *Sparse matrix format that load balances computation, demonstrated for LSTMs.*
- **CCR: A Concise Convolution Rule for Sparse Neural Network Accelerators.** (CAS)
   - *Decompose convolution into multiple dense and zero kernels for sparsity savings.*
- **Block Convolution: Towards Memory-Efficient Inference of Large-Scale CNNs on FPGA.** (CAS)
- **moDNN: Memory Optimal DNN Training on GPUs.** (University of Notre Dame, CAS)
- HyperPower: Power and Memory-Constrained Hyper-Parameter Optimization for Neural Networks. (CMU, Google)

### 2018 DAC
- **Compensated-DNN: Energy Efficient Low-Precision Deep Neural Networks by Compensating Quantization Errors.** (**Best Paper**, Purdue, IBM)
  - *Introduce a new fixed-point representation, Fixed Point with Error Compensation (FPEC): Computation bits, +compensation bits that represent quantization error.*
  - *Propose a low-overhead sparse compensation scheme to estimate the error in MAC design.*
- **Calibrating Process Variation at System Level with In-Situ Low-Precision Transfer Learning for Analog Neural Network Processors.** (THU)
- **DPS: Dynamic Precision Scaling for Stochastic Computing-Based Deep Neural Networks.** (UNIST)
- **DyHard-DNN: Even More DNN Acceleration With Dynamic Hardware Reconfiguration.** (Univ. of Virginia)
- **Exploring the Programmability for Deep Learning Processors: from Architecture to Tensorization.** (Univ. of Washington)
- **LCP: Layer Clusters Paralleling Mapping Mechanism for Accelerating Inception and Residual Networks on FPGA.** (THU)
- **A Kernel Decomposition Architecture for Binary-weight Convolutional Neural Networks.** (THU)
- **Ares: A Framework for Quantifying the Resilience of Deep Neural Networks.** (Harvard)
- **ThUnderVolt: Enabling Aggressive Voltage Underscaling and Timing Error Resilience for Energy Efficient
Deep Learning Accelerators** (New York Univ., IIT Kanpur)
- **Loom: Exploiting Weight and Activation Precisions to Accelerate Convolutional Neural Networks.** (Univ. of Toronto)
- **Parallelizing SRAM Arrays with Customized Bit-Cell for Binary Neural Networks.** (Arizona)
- **Thermal-Aware Optimizations of ReRAM-Based Neuromorphic Computing Systems.** (Northwestern Univ.)
- **SNrram: An Efficient Sparse Neural Network Computation Architecture Based on Resistive RandomAccess Memory.** (THU, UCSB)
- **Long Live TIME: Improving Lifetime for Training-In-Memory Engines by Structured Gradient Sparsification.** (THU, CAS, MIT)
- **Bandwidth-Efficient Deep Learning.** (MIT, Stanford)
- **Co-Design of Deep Neural Nets and Neural Net Accelerators for Embedded Vision Applications.** (Berkeley)
- **Sign-Magnitude SC: Getting 10X Accuracy for Free in Stochastic Computing for Deep Neural Networks.** (UNIST)
- **DrAcc: A DRAM Based Accelerator for Accurate CNN Inference.** (National Univ. of Defense Technology, Indiana Univ., Univ. of Pittsburgh)
- **On-Chip Deep Neural Network Storage With Multi-Level eNVM.** (Harvard)
- VRL-DRAM: Improving DRAM Performance via Variable Refresh Latency. (Drexel Univ., ETHZ)

### 2018 HotChips
- **ARM's First Generation ML Processor.** (ARM)
- **The NVIDIA Deep Learning Accelerator.** (NVIDIA)
- **Xilinx Tensor Processor: An Inference Engine, Network Compiler + Runtime for Xilinx FPGAs.** (Xilinx)
- Tachyum Cloud Chip for Hyperscale workloads, deep ML, general, symbolic and bio AI. (Tachyum)
- SMIV: A 16nm SoC with Efficient and Flexible DNN Acceleration for Intelligent IoT Devices. (ARM)
- NVIDIA's Xavier System-on-Chip. (NVIDIA)
- Xilinx Project Everest: HW/SW Programmable Engine. (Xilinx)

### 2018 ICCAD
- **Tetris: Re-architecting Convolutional Neural Network Computation for Machine Learning Accelerators.** (CAS)
- **3DICT: A Reliable and QoS Capable Mobile Process-In-Memory Architecture for Lookup-based CNNs in 3D XPoint ReRAMs.** (Indiana - - University Bloomington, Florida International Univ.)
- **TGPA: Tile-Grained Pipeline Architecture for Low Latency CNN Inference.** (PKU, UCLA, Falcon)
- **NID: Processing Binary Convolutional Neural Network in Commodity DRAM.** (KAIST)
- **Adaptive-Precision Framework for SGD using Deep Q-Learning.** (PKU)
- **Efficient Hardware Acceleration of CNNs using Logarithmic Data Representation with Arbitrary log-base.** (Robert Bosch GmbH)
- **C-GOOD: C-code Generation Framework for Optimized On-device Deep Learning.** (SNU)
- **Mixed Size Crossbar based RRAM CNN Accelerator with Overlapped Mapping Method.** (THU)
- **FCN-Engine: Accelerating Deconvolutional Layers in Classic CNN Processors.** (HUT, CAS, NUS)
- **DNNBuilder: an Automated Tool for Building High-Performance DNN Hardware Accelerators for FPGAs.** (UIUC)
- **DIMA: A Depthwise CNN In-Memory Accelerator.** (Univ. of Central Florida)
- **EMAT: An Efficient Multi-Task Architecture for Transfer Learning using ReRAM.** (Duke)
- **FATE: Fast and Accurate Timing Error Prediction Framework for Low Power DNN Accelerator Design.** (NYU)
- **Designing Adaptive Neural Networks for Energy-Constrained Image Classification.** (CMU)
- Watermarking Deep Neural Networks for Embedded Systems. (UCLA)
- Defensive Dropout for Hardening Deep Neural Networks under Adversarial Attacks. (Northeastern Univ., Boston Univ., Florida International Univ.)
- A Cross-Layer Methodology for Design and Optimization of Networks in 2.5D Systems. (Boston Univ., UCSD)

### 2018 MICRO
- **Addressing Irregularity in Sparse Neural Networks: A Cooperative Software/Hardware Approach.** (USTC, CAS)
- **Diffy: a Deja vu-Free Differential Deep Neural Network Accelerator.** (University of Toronto)
- **Beyond the Memory Wall: A Case for Memory-centric HPC System for Deep Learning.** (KAIST)
- **Towards Memory Friendly Long-Short Term Memory Networks (LSTMs) on Mobile GPUs.** (University of Houston, Capital Normal University)
- **A Network-Centric Hardware/Algorithm Co-Design to Accelerate Distributed Training of Deep Neural Networks.** (UIUC, THU, SJTU, Intel, UCSD)
- **PermDNN: Efficient Compressed Deep Neural Network Architecture with Permuted Diagonal Matrices.** (City University of New York, University of Minnesota, USC)
- **GeneSys: Enabling Continuous Learning through Neural Network Evolution in Hardware.** (Georgia Tech)
- **Processing-in-Memory for Energy-efficient Neural Network Training: A Heterogeneous Approach.** (UCM, UCSD, UCSC)
  - Schedules computing resources provided by CPU and heterogeneous PIMs (fixed-function logic + programmable ARM cores), to optimized energy efficiency and hardware utilization.
- **LerGAN: A Zero-free, Low Data Movement and PIM-based GAN Architecture.** (THU, University of Florida)
- **Multi-dimensional Parallel Training of Winograd Layer through Distributed Near-Data Processing.** (KAIST)
  - Winograd is applied to training to extend traditional data parallelsim with a new dimension named intra-tile parallelism. With intra-tile parallelism, nodes ara dividied into several groups, and weight update communication only occurs independtly in the group. The method shows better scalability for training clusters, as the total commnication doesn't scale with the increasing of node count.
- **SCOPE: A Stochastic Computing Engine for DRAM-based In-situ Accelerator.** (UCSB, Samsung)
- **Morph: Flexible Acceleration for 3D CNN-based Video Understanding.** (UIUC)
- Inter-thread Communication in Multithreaded, Reconfigurable Coarse-grain Arrays. (Technion)
- An Architectural Framework for Accelerating Dynamic Parallel Algorithms on Reconfigurable Hardware. (Cornell)

### 2019 ASPDAC
- **An N-way group association architecture and sparse data group association load balancing algorithm for sparse CNN accelerators.** (THU)
- **TNPU: An Efficient Accelerator Architecture for Training Convolutional Neural Networks.** (ICT)
- **NeuralHMC: An Efficient HMC-Based Accelerator for Deep Neural Networks.** (University of Pittsburgh, Duke)
- **P3M: A PIM-based Neural Network Model Protection Scheme for Deep Learning Accelerator.** (ICT)
- GraphSAR: A Sparsity-Aware Processing-in-Memory Architecture for Large-Scale Graph Processing on ReRAMs. (Tsinghua, MIT, Berkely)

### 2019 ISSCC
- **An 11.5TOPS/W 1024-MAC Butterfly Structure Dual-Core Sparsity-Aware Neural Processing Unit in 8nm Flagship Mobile SoC.** (Samsung)
- **A 20.5TOPS and 217.3GOPS/mm2 Multicore SoC with DNN Accelerator and Image Signal Processor Complying with ISO26262 for Automotive Applications.** (Toshiba)
- **An 879GOPS 243mW 80fps VGA Fully Visual CNN-SLAM Processor for Wide-Range Autonomous Exploration.** (Michigan)
- **A 2.1TFLOPS/W Mobile Deep RL Accelerator with Transposable PE Array and Experience Compression.** (KAIST)
- **A 65nm 0.39-to-140.3TOPS/W 1-to-12b Unified Neural-Network Processor Using Block-Circulant-Enabled Transpose-Domain Acceleration with 8.1× Higher TOPS/mm2 and 6T HBST-TRAM-Based 2D Data-Reuse Architecture.** (THU, National Tsing Hua University, Northeastern University)
- **A 65nm 236.5nJ/Classification Neuromorphic Processor with 7.5% Energy Overhead On-Chip Learning Using Direct Spike-Only Feedback.** (SNU)
- **LNPU: A 25.3TFLOPS/W Sparse Deep-Neural-Network Learning Processor with Fine-Grained Mixed Precision of FP8-FP16.** (KAIST)
- A 1Mb Multibit ReRAM Computing-In-Memory Macro with 14.6ns Parallel MAC Computing Time for CNN-Based AI Edge Processors. (National Tsing Hua University)
- Sandwich-RAM: An Energy-Efficient In-Memory BWN Architecture with Pulse-Width Modulation. (Southeast University, Boxing Electronics, THU)
- A Twin-8T SRAM Computation-In-Memory Macro for Multiple-Bit CNN Based Machine Learning. (National Tsing Hua University, University of Electronic Science and Technology of China, ASU, Georgia Tech)
- A Reconfigurable RRAM Physically Unclonable Function Utilizing PostProcess Randomness Source with <6×10-6 Native Bit Error Rate. (THU, National Tsing Hua University, Georgia Tech)
- A 65nm 1.1-to-9.1TOPS/W Hybrid-Digital-Mixed-Signal Computing Platform for Accelerating Model-Based and Model-Free Swarm Robotics. (Georgia Tech)
- A Compute SRAM with Bit-Serial Integer/Floating-Point Operations for Programmable In-Memory Vector Acceleration. (Michigan)
- All-Digital Time-Domain CNN Engine Using Bidirectional Memory Delay Lines for Energy-Efficient Edge Computing. (UT Austin)

### 2019 HPCA
- **HyPar: Towards Hybrid Parallelism for Deep Learning Accelerator Array.** (Duke, USC)
- **E-RNN: Design Optimization for Efficient Recurrent Neural Networks in FPGAs.** (Syracuse University, Northeastern University, Florida International University, USC, University at Buffalo)
- **Bit Prudent In-Cache Acceleration of Deep Convolutional Neural Networks.** (Michigan, Intel)
- **Shortcut Mining: Exploiting Cross-layer Shortcut Reuse in DCNN Accelerators.** (OSU)
- **NAND-Net: Minimizing Computational Complexity of In-Memory Processing for Binary Neural Networks.** (KAIST)
- **Kelp: QoS for Accelerators in Machine Learning Platforms.** (Microsoft, Google, UT Austin)
- **Machine Learning at Facebook: Understanding Inference at the Edge.** (Facebook)
- The Accelerator Wall: Limits of Chip Specialization. (Princeton)

### 2019 ASPLOS
- **FA3C: FPGA-Accelerated Deep Reinforcement Learning.** (Hongik University, SNU)
- **PUMA: A Programmable Ultra-efficient Memristor-based Accelerator for Machine Learning Inference.** (Purdue, UIUC, HP)
- **FPSA: A Full System Stack Solution for Reconfigurable ReRAM-based NN Accelerator Architecture.** (THU, UCSB)
- **Bit-Tactical: A Software/Hardware Approach to Exploiting Value and Bit Sparsity in Neural Networks.** (Toronto, NVIDIA)
- **TANGRAM: Optimized Coarse-Grained Dataflow for Scalable NN Accelerators.** (Stanford)
- **Packing Sparse Convolutional Neural Networks for Efficient Systolic Array Implementations: Column Combining Under Joint Optimization.** (Harvard)
- **Split-CNN: Splitting Window-based Operations in Convolutional Neural Networks for Memory System Optimization.** (IBM, Kyungpook National University)
- **HOP: Heterogeneity-Aware Decentralized Training.** (USC, THU)
- **Astra: Exploiting Predictability to Optimize Deep Learning.** (Microsoft)
- **ADMM-NN: An Algorithm-Hardware Co-Design Framework of DNNs Using Alternating Direction Methods of Multipliers.** (Northeastern, Syracuse, SUNY, Buffalo, USC)
- **DeepSigns: An End-to-End Watermarking Framework for Protecting the Ownership of Deep Neural Networks.** (UCSD)

### 2019 ISCA
- **Sparse ReRAM Engine: Joint Exploration of Activation and Weight Sparsity on Compressed Neural Network.** (NTU, Academia Sinica, Macronix)
- **MnnFast: A Fast and Scalable System Architecture for Memory-Augmented Neural Networks.** (POSTECH, SNU)
- **TIE: Energy-efficient Tensor Train-based Inference Engine for Deep Neural Network.** (Rutgers University, Nanjing University, USC)
- **Accelerating Distributed Reinforcement Learning with In-Switch Computing.** (UIUC)
- **Eager Pruning: Algorithm and Architecture Support for Fast Training of Deep Neural Networks.** (University of Florida)
- **Laconic Deep Learning Inference Acceleration.** (Toronto)
- **DeepAttest: An End-to-End Attestation Framework for Deep Neural Networks.** (UCSD)
- **A Stochastic-Computing based Deep Learning Framework using Adiabatic Quantum-Flux-Parametron Superconducting Technology.** (Northeastern, Yokohama National University, USC, University of Alberta)
- **Fractal Machine Learning Computers.** (ICT)
- **FloatPIM: In-Memory Acceleration of Deep Neural Network Training with High Precision.** (UCSD)
- 3D-based Video Understanding Acceleration by Leveraging Temporal Locality and Activation Sparsity. (University of Florida, University of Florida)
- Energy-Efficient Video Processing for Virtual Reality. (UIUC, University of Rochester)
- Scalable Interconnects for Reconfigurable Spatial Architectures. (Stanford)
- CoNDA: Enabling Efficient Near-Data Accelerator Communication by Optimizing Data Movement. (CMU, ETHZ)

### 2019 DAC
- **Accuracy vs. Efficiency: Achieving Both through FPGA-Implementation Aware Neural Architecture Search.** (East China Normal University, Pittsburgh, Chongqing University, UCI,  Notre Dame)
- **FPGA/DNN Co-Design: An Efficient Design Methodology for IoT Intelligence on the Edge.** (UIUC, IBM, Inspirit IoT)
- **An Optimized Design Technique of Low-Bit Neural Network Training for Personalization on IoT Devices.** (KAIST)
- **ReForm: Static and Dynamic Resource-Aware DNN Reconfiguration Framework for Mobile Devices.** (George Mason, Clarkson)
- **DRIS-3: Deep Neural Network Reliability Improvement Scheme in 3D Die-Stacked Memory based on Fault Analysis.** (Sungkyunkwan University)
- **ZARA: A Novel Zero-free Dataflow Accelerator for Generative Adversarial Networks in 3D ReRAM.** (Duke)
- **BitBlade: Area and Energy-Efficient Precision-Scalable Neural Network Accelerator with Bitwise Summation.** (POSTECH)
- X-MANN: A Crossbar based Architecture for Memory Augmented Neural Networks. (Purdue, Intel)
- Thermal-Aware Design and Management for Search-based In-Memory Acceleration. (UCSD)
- An Energy-Efficient Network-on-Chip Design using Reinforcement Learning. (George Washington)
- Designing Vertical Processors in Monolithic 3D. (UIUC)

### 2019 MICRO
- **Wire-Aware Architecture and Dataflow for CNN Accelerators.** (Utah)
- **ShapeShifter: Enabling Fine-Grain Data Width Adaptation in Deep Learning.** (Toronto)
- **Simba: Scaling Deep-Learning Inference with Multi-Chip-Module-Based Architecture.** (NVIDIA)
- **ZCOMP: Reducing DNN Cross-Layer Memory Footprint Using Vector Extensions.** (Google, Intel)
- **Boosting the Performance of CNN Accelerators with Dynamic Fine-Grained Channel Gating.** (Cornell)
- **SparTen: A Sparse Tensor Accelerator for Convolutional Neural Networks.** (Purdue)
- **EDEN: Enabling Approximate DRAM for DNN Inference using Error-Resilient Neural Networks.** (ETHZ, CMU)
- **eCNN: a Block-Based and Highly-Parallel CNN Accelerator for Edge Inference.** (NTHU)
- **TensorDIMM: A Practical Near-Memory Processing Architecture for Embeddings and Tensor Operations in Deep Learning.** (KAIST)
- **Understanding Reuse, Performance, and Hardware Cost of DNN Dataflows: A Data-Centric Approach.** (Georgia Tech, NVIDIA)
- **MaxNVM: Maximizing DNN Storage Density and Inference Efficiency with Sparse Encoding and Error Mitigation.** (Harvard, Facebook)
- **Neuron-Level Fuzzy Memoization in RNNs.** (UPC)
- **Manna: An Accelerator for Memory-Augmented Neural Networks.** (Purdue, Intel)
- eAP: A Scalable and Efficient In-Memory Accelerator for Automata Processing. (Virginia)
- ComputeDRAM: In-Memory Compute Using Off-the-Shelf DRAMs. (Princeton)
- ExTensor: An Accelerator for Sparse Tensor Algebra. (UIUC, NVIDIA)
- Efficient SpMV Operation for Large and Highly Sparse Matrices Using Scalable Multi-Way Merge Parallelization. (CMU)
- Sparse Tensor Core: Algorithm and Hardware Co-Design for Vector-wise Sparse Neural Networks on Modern GPUs. (UCSB, Alibaba)
- DynaSprint: Microarchitectural Sprints with Dynamic Utility and Thermal Management. (Waterloo, ARM, Duke)
- MEDAL: Scalable DIMM based Near Data Processing Accelerator for DNA Seeding Algorithm. (UCSB, ICT)
- Tigris: Architecture and Algorithms for 3D Perception in Point Clouds. (Rochester)
- ASV: Accelerated Stereo Vision System. (Rochester)
- Alleviating Irregularity in Graph Analytics Acceleration: a Hardware/Software Co-Design Approach. (UCSB, ICT)

### 2019 ICCAD
- **Zac: Towards Automatic Optimization and Deployment of Quantized Deep Neural Networks on Embedded Devices.** (PKU)
- **NAIS: Neural Architecture and Implementation Search and its Applications in Autonomous Driving.** (UIUC)
- **MAGNet: A Modular Accelerator Generator for Neural Networks.** (NVIDIA)
- **ReDRAM: A Reconfigurable Processing-in-DRAM Platform for Accelerating Bulk Bit-Wise Operations.** (ASU)
- **Accelergy: An Architecture-Level Energy Estimation Methodology for Accelerator Designs.** (MIT)

### 2019 ASSCC
- **A 47.4µJ/epoch Trainable Deep Convolutional Neural Network Accelerator for In-Situ Personalization on Smart Devices.** (KAIST)
- **A 2.25 TOPS/W Fully-Integrated Deep CNN Learning Processor with On-Chip Training.** (NTU)
- **A Sparse-Adaptive CNN Processor with Area/Performance Balanced N-Way Set-Associate Pe Arrays Assisted by a Collision-Aware Scheduler.** (THU, Northeastern)
- A 24 Kb Single-Well Mixed 3T Gain-Cell eDRAM with Body-Bias in 28 nm FD-SOI for Refresh-Free DSP Applications. (EPFL)

### 2019 VLSI
- **Area-Efficient and Variation-Tolerant In-Memory BNN Computing Using 6T SRAM Array.** (POSTECH)
- **A 5.1pJ/Neuron 127.3us/Inference RNN-Based Speech Recognition Processor Using 16 Computingin-Memory SRAM Macros in 65nm CMOS.** (THU, NTU, TsingMicro)
- **A 0.11 pJ/Op, 0.32-128 TOPS, Scalable, Multi-ChipModule-Based Deep Neural Network Accelerator with Ground-Reference Signaling in 16nm.** (NVIDIA)
- **SNAP: A 1.67 – 21.55TOPS/W Sparse Neural Acceleration Processor for Unstructured Sparse Deep Neural Network Inference in 16nm CMOS.** (UMich, NVIDA)
- **A Full HD 60 fps CNN Super Resolution Processor with Selective Caching based Layer Fusion for Mobile Devices.** (KAIST)
- **A 1.32 TOPS/W Energy Efficient Deep Neural Network Learning Processor with Direct Feedback Alignment based Heterogeneous Core Architecture.** (KAIST)
- Considerations of Integrating Computing-In-Memory and Processing-In-Sensorinto Convolutional Neural Network Accelerators for Low-Power Edge Devices. (NTU, NCHU)
- Computational Memory-Based Inference and Training of Deep Neural Networks. (IBM, EPFL, ETHZ, et al)
- A Ternary Based Bit Scalable, 8.80 TOPS/W CNN A95Accelerator with Many-Core Processing-in-Memory Architecture with 896K Synapses/mm2. (Renesas)
- In-Memory Reinforcement Learning with ModeratelyStochastic Conductance Switching of Ferroelectric Tunnel Junctions. (Toshiba)

### 2019 HotChips
- **MLPerf: A Benchmark Suite for Machine Learning from an Academic-Industry Cooperative.** (MLPerf)
- **Zion: Facebook Next-Generation Large-memory Unified Training Platform.** (Facebook)
- **A Scalable Unified Architecture for Neural Network Computing from Nano-Level to High Performance Computing.** (Huawei)
- **Deep Learning Training at Scale – Spring Crest Deep Learning Accelerator.** (Intel)
- **Spring Hill – Intel’s Data Center Inference Chip.** (Intel)
- **Wafer Scale Deep Learning.** (Cerebras)
- **Habana Labs Approach to Scaling AI Training.** (Habana)
- **Ouroboros: A WaveNet Inference Engine for TTS Applications on Embedded Devices.** (Alibaba)
- **A 0.11 pJ/Op, 0.32-128 TOPS, Scalable Multi-Chip-Module-based Deep Neural Network Accelerator Designed with a High-Productivity VLSI Methodology.** (NVIDIA)
- **Xilinx Versal/AI Engine.** (Xilinx)
- A Programmable Embedded Microprocessor for Bit-scalable In-memory Computing. (Princeton)

### 2019 FPGA
- **Synetgy: Algorithm-hardware Co-design for ConvNet Accelerators on Embedded FPGAs.** (THU, Berkeley, Politecnico di Torino, Xilinx)
- **REQ-YOLO: A Resource-Aware, Efficient Quantization Framework for Object Detection on FPGAs.** (PKU, Northeastern）
- **Reconfigurable Convolutional Kernels for Neural Networks on FPGAs.** (University of Kassel)
- **Efficient and Effective Sparse LSTM on FPGA with Bank-Balanced Sparsity.** (Harbin Institute of Technology, Microsoft, THU, Beihang)
- **Cloud-DNN: An Open Framework for Mapping DNN Models to Cloud FPGAs.** (Advanced Digital Sciences Center, UIUC)
- F5-HD: Fast Flexible FPGA-based Framework for Refreshing Hyperdimensional Computing. (UCSD)
- Xilinx Adaptive Compute Acceleration Platform: Versal Architecture. (Xilinx)

### 2020 ISSCC
- **A 3.4-to-13.3TOPS/W 3.6TOPS Dual-Core Deep-Learning Accelerator for Versatile AI Applications in 7nm 5G Smartphone SoC.** (MediaTek)
- **A 12nm Programmable Convolution-Efficient Neural-Processing-Unit Chip Achieving 825TOPS.** (Alibaba)
- **STATICA: A 512-Spin 0.25M-Weight Full-Digital Annealing Processor with a Near-Memory All-SpinUpdates-at-Once Architecture for Combinatorial Optimization with Complete Spin-Spin Interactions.** (Tokyo Institute of Technology, Hokkaido Univ., Univ. of Tokyo)
- **GANPU: A 135TFLOPS/W Multi-DNN Training Processor for GANs with Speculative Dual-Sparsity Exploitation.** (KAIST)
- **A 510nW 0.41V Low-Memory Low-Computation Keyword-Spotting Chip Using Serial FFT-Based MFCC and Binarized Depthwise Separable Convolutional Neural Network in 28nm CMOS.** (Southeast, EPFL, Columbia)
- **A 65nm 24.7μJ/Frame 12.3mW Activation-Similarity Aware Convolutional Neural Network Video Processor Using Hybrid Precision, Inter-Frame Data Reuse and Mixed-Bit-Width Difference-Frame Data Codec.** (THU)
- **A 65nm Computing-in-Memory-Based CNN Processor with 2.9-to-35.8TOPS/W System Energy Efficiency Using Dynamic-Sparsity Performance-Scaling Architecture and Energy-Efficient Inter/Intra-Macro Data Reuse.** (THU, NTHU)
- A 28nm 64Kb Inference-Training Two-Way Transpose Multibit 6T SRAM Compute-in-Memory Macro for AI Edge Chips. (NTU)
- A 351TOPS/W and 372.4GOPS Compute-in-Memory SRAM Macro in 7nm FinFET CMOS for Machine-Learning Applications. (TSMC)
- A 22nm 2Mb ReRAM Compute-in-Memory Macro with 121-28TOPS/W for Multibit MAC Computing for Tiny AI Edge Devices. (NTHU)
- A 28nm 64Kb 6T SRAM Computing-in-Memory Macro with 8b MAC Operation for AI Edge Chips. (NTHU)
- A 1.5μJ/Task Path-Planning Processor for 2D/3D Autonomous Navigation of Micro Robots. (NTHU)
- A 65nm 8.79TOPS/W 23.82mW Mixed-Signal Oscillator-Based NeuroSLAM Accelerator for Applications in Edge Robotics. (Georgia Tech)
- CIM-Spin: A 0.5-to-1.2V Scalable Annealing Processor Using Digital Compute-In-Memory Spin Operators and Register-Based Spins for Combinatorial Optimization Problems. (NTU)
- A Compute-Adaptive Elastic Clock-Chain Technique with Dynamic Timing Enhancement for 2D PE-Array-Based Accelerators. (Northwestern)
- A 74 TMACS/W CMOS-RRAM Neurosynaptic Core with Dynamically Reconfigurable Dataflow and In-situ Transposable Weights for Probabilistic Graphical Models. (Stanford, UCSD, THU, Notre Dame)
- A Fully Integrated Analog ReRAM Based 78.4TOPS/W Compute-In-Memory Chip with Fully Parallel MAC Computing. (THU, NTHU)

### 2020 HPCA
- **Deep Learning Acceleration with Neuron-to-Memory Transformation.**	(UCSD)
- **HyGCN: A GCN Accelerator with Hybrid Architecture.**	(ICT, UCSB)
- **SIGMA: A Sparse and Irregular GEMM Accelerator with Flexible Interconnects for DNN Training.**	(Georgia Tech)
- **PREMA: A Predictive Multi-task Scheduling Algorithm For Preemptible NPUs.**	(KAIST)
- **ALRESCHA: A Lightweight Reconfigurable Sparse-Computation Accelerator.**	(Georgia Tech)
- **SpArch: Efficient Architecture for Sparse Matrix Multiplication.**	(MIT, NVIDIA)
- **A3: Accelerating Attention Mechanisms in Neural Networks with Approximation.**	(SNU)
- **AccPar: Tensor Partitioning for Heterogeneous Deep Learning Accelerator Arrays.**	(Duke, USC)
- **PIXEL: Photonic Neural Network Accelerator.**	(Ohio, George Washington)
- **The Architectural Implications of Facebook’s DNN-based Personalized Recommendation.**	(Facebook)
- **Enabling Highly Efficient Capsule Networks Processing Through A PIM-Based Architecture Design.**	(Houston)
- **Missing the Forest for the Trees: End-to-End AI Application Performance in Edge Data.**	(UT Austin, Intel)
- **Communication Lower Bound in Convolution Accelerators.**	(ICT, THU)
- **Fulcrum: a Simplified Control and Access Mechanism toward Flexible and Practical in-situ Accelerators.**	(Virginia, UCSB, Micron)
- **EFLOPS: Algorithm and System Co-design for a High Performance Distributed Training Platform.**	(Alibaba)
- **Experiences with ML-Driven Design: A NoC Case Study.**	(AMD)
- **Tensaurus: A Versatile Accelerator for Mixed Sparse-Dense Tensor Computations.**	(Cornell, Intel)
- **A Hybrid Systolic-Dataflow Architecture for Inductive Matrix Algorithms.**	(UCLA)
- A Deep Reinforcement Learning Framework for Architectural Exploration: A Routerless NoC Case Study.	(USC, OSU)
- QuickNN: Memory and Performance Optimization of k-d Tree Based Nearest Neighbor Search for 3D Point Clouds.	(Umich, General Motors)
- Orbital Edge Computing: Machine Inference in Space.	(CMU)
- A Scalable and Efficient in-Memory Interconnect Architecture for Automata Processing.	(Virginia)
- Techniques for Reducing the Connected-Standby Energy Consumption of Mobile Devices.	(ETHZ, Cyprus, CMU)

### 2020 ASPLOS
- **Shredder: Learning Noise Distributions to Protect Inference Privacy.**	(UCSD)
- **DNNGuard: An Elastic Heterogeneous DNN Accelerator Architecture against Adversarial Attacks.**	(CAS, USC)
- **Interstellar: Using Halide’s Scheduling Language to Analyze DNN Accelerators.**	(Stanford, THU)
- **DeepSniffer: A DNN Model Extraction Framework Based on Learning Architectural Hints.**	(UCSB)
- **Prague: High-Performance Heterogeneity-Aware Asynchronous Decentralized Training.**	(USC)
- **PatDNN: Achieving Real-Time DNN Execution on Mobile Devices with Pattern-based Weight Pruning.**	(College of William and Mary, Northeastern , USC)
- **Capuchin: Tensor-based GPU Memory Management for Deep Learning.**	(HUST, MSRA, USC)
- **NeuMMU: Architectural Support for Efficient Address Translations in Neural Processing Units.**	(KAIST)
- **FlexTensor: An Automatic Schedule Exploration and Optimization Framework for Tensor Computation on Heterogeneous System.**	(PKU)

### 2020 DAC
- **A Pragmatic Approach to On-device Incremental Learning System with Selective Weight Updates.**
- **A Two-way SRAM Array based Accelerator for Deep Neural Network On-chip Training.**
- **Algorithm-Hardware Co-Design for In-Memory Neural Network Computing with Minimal Peripheral Circuit Overhead.**
- **Algorithm-Hardware Co-Design of Adaptive Floating-Point Encodings for Resilient Deep Learning Inference.**
- **Hardware Acceleration of Graph Neural Networks.**
- **Exploiting Dataflow Sparsity for Efficient Convolutional Neural Networks Training.**
- **Low-Power Acceleration of Deep Neural Network Training Using Computational Storage Devices.**
- **Prediction Confidence based Low Complexity Gradient Computation for Accelerating DNN Training.**
- **SparseTrain: Exploiting Dataflow Sparsity for Efficient Convolutional Neural Networks Training.**
- **SCA: A Secure CNN Accelerator for both Training and Inference.**
- **STC: Significance-aware Transform-based Codec Framework for External Memory Access Reduction.**

### 2020 FPGA
- **AutoDNNchip: An Automated DNN Chip Generator through Compilation, Optimization, and Exploration.** （Rice, UIUC)
- **Accelerating GCN Training on CPU-FPGA Heterogeneous Platforms.** (USC)
- Massively Simulating Adiabatic Bifurcations with FPGA to Solve Combinatorial Optimization. (Central Florida)

### 2020 ISCA
- **Data Compression Accelerator on IBM POWER9 and z15 Processors.** (IBM)
- **High-Performance Deep-Learning Coprocessor Integrated into x86 SoC with Server-Class CPUs.**	(Centaur )
- **Think Fast: A Tensor Streaming Processor (TSP) for Accelerating Deep Learning Workloads.** (Groq)
- **MLPerf Inference: A Benchmarking Methodology for Machine Learning Inference Systems.**	
- **A Multi-Neural Network Acceleration Architecture.** (SNU)
- **SmartExchange: Trading Higher-Cost Memory Storage/Access for Lower-Cost Computation.** (Rice, TAMU, UCSB)
- **Centaur: A Chiplet-Based, Hybrid Sparse-Dense Accelerator for Personalized Recommendations.** (KAIST)
- **DeepRecSys: A System for Optimizing End-to-End At-Scale Neural Recommendation Inference.**	(Facebook, Harvard)
- **An In-Network Architecture for Accelerating Shared-Memory Multiprocessor Collectives.**	(NVIDIA)
- **DRQ: Dynamic Region-Based Quantization for Deep Neural Network Acceleration.**	(SJTU)
- The IBM z15 High Frequency Mainframe Branch Predictor. (ETHZ)
- Déjà View: Spatio-Temporal Compute Reuse for Energy-Efficient 360° VR Video Streaming. (Penn State)
- uGEMM: Unary Computing Architecture for GEMM Applications. (Wisconsin)
- Gorgon: Accelerating Machine Learning from Relational Data. (Stanford)
- RecNMP: Accelerating Personalized Recommendation with Near-Memory Processing. (Facebook)
- JPEG-ACT: Accelerating Deep Learning via Transform-Based Lossy Compression. (UBC)
- Commutative Data Reordering: A New Technique to Reduce Data Movement Energy on Sparse Inference Workloads. (Sandia, Rochester)
- Echo: Compiler-Based GPU Memory Footprint Reduction for LSTM RNN Training. (Toronto, Intel)

## Important Topics
This is a collection of papers on other important topics related to neural networks. Papers of significance are marked in **bold**. My comments are in marked in *italic*.

### Tutorial and Survey
- [Tutorial on Hardware Architectures for Deep Neural Networks.](http://eyeriss.mit.edu/tutorial.html) (MIT)
- [A Survey of Neuromorphic Computing and Neural Networks in Hardware.](https://arxiv.org/abs/1705.06963) (Oak Ridge National Lab)
- [A Survey of FPGA Based Neural Network Accelerator.](https://arxiv.org/abs/1712.08934) (Tsinghua)
- [Toolflows for Mapping Convolutional Neural Networks on FPGAs: A Survey and Future Directions.](https://arxiv.org/abs/1803.05900) (Imperial College London)

### Benchmarks
- [DAWNBench](https://dawn.cs.stanford.edu//benchmark/): An End-to-End Deep Learning Benchmark and Competition. (Stanford)
- [MLPerf](https://mlperf.org/): A broad ML benchmark suite for measuring performance of ML software frameworks, ML hardware accelerators, and ML cloud platforms.
- [Fathom: Reference Workloads for Modern Deep Learning Methods.](http://arxiv.org/abs/1608.06581) (Harvard University)
- **AlexNet**: Imagenet Classification with Deep Convolutional Neural Networks. (University of Toronto, **2012 NIPS**)
- **Network in Network**. (National University of Singapore, **2014 ICLR**)
- **ZFNet**: Visualizing and Understanding Convolutional Networks. (New York University, **2014 ECCV**)
- **OverFeat**: Integrated Recognition, Localization and Detection using Convolutional Networks. (New York University, **2014 CVPR**)
- **VGG**: Very Deep Convolutional Networks for Large-Scale Image Recognition. (Univerisity of Oxford, **2015 ICLR**)
- **GoogLeNet**: Going Deeper with Convolutions. (Google, University of North Carolina, University of Michigan, **2015 CVPR**)
- **ResNet**: Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. (MSRA, **2015 ICCV**)
- **MobileNetV1**: [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.](https://arxiv.org/abs/1704.04861)  [[code]](https://github.com/tensorflow/models/tree/master/research/slim) (Google, **2017 CVPR**)
- **MobileNetV2**: [MobileNetV2: Inverted Residuals and Linear Bottlenecks.](https://arxiv.org/abs/1801.04381) [[code]](https://github.com/tensorflow/models/tree/master/research/slim) (Google, **2018 CVPR**)

### Network Compression
- [Neural Network Distiller](https://nervanasystems.github.io/distiller/index.html): Intel's open-source Python package for neural network compression research. [[Chinese]](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650744106&idx=4&sn=6e101276f64a9c91b8dc24b89ddaeec3&chksm=871ae154b06d6842c3cc376242322298db0288ab696e92fac4268be8c85b75daa81d72d763ad&mpshare=1&scene=1&srcid=0622gO3XXgG8qIvMWxc8Sj11#rd)
#### Conference Papers
- [Learning both Weights and Connections for Efficient Neural Network.](https://arxiv.org/abs/1506.02626) (Stanford University, NVIDIA, **2015 NIPS**)
  - *Prune connections by thresholding weight values.*
  - *Retain accuracy with iterative retraining.*
- [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding.](http://arxiv.org/abs/1510.00149) (Stanford University, Tsinghua University, **2016 ICLR**)
- [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size.](http://arxiv.org/abs/1602.07360) (DeepScale & UC Berkeley, Stanford University)
- [8-Bit Approximations for Parallelism in Deep Learning.](http://arxiv.org/abs/1511.04561) (Universia della Svizzera italiana, **2016 ICLR**)
- [Neural Networks with Few Multiplications.](https://arxiv.org/abs/1510.03009) (Universite de Montreal, **2016 ICLR**)
- [Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications.](http://arxiv.org/abs/1511.06530) (Samsung, Seoul National University, **2016 ICLR**)
- [Hardware-oriented Approximation of Convolutional Neural Networks.](https://arxiv.org/abs/1604.03168) (UC Davis, **2016 ICLR Workshop**)
- [Soft Weight-Sharing for Neural Network Compression.](https://arxiv.org/abs/1702.04008) (University of Amsterdam, CIFAR, **2017 ICLR**)
- [Designing Energy-Efficient Convolutional Neural Networks using Energy-Aware Pruning.](https://arxiv.org/abs/1611.05128) (MIT, **2017 CVPR**)
  - *Estimate the energy comsuption of a CNN based on their Eyeriss (ISCA'16) paper.*
  - *Propose an energy-aware pruning method.*
- [Scalable and Sustainable Deep Learning via Randomized Hashing.](https://arxiv.org/abs/1602.08194) (Rice University, **2017 KDD**)
- [TernGrad: Ternary Gradients to Reduce Communication in Distributed Deep Learning.](https://arxiv.org/abs/1705.07878) [[code]](https://github.com/wenwei202/terngrad) (Duke University, Hewlett Packard Labs, University of Nevada – Reno, University of Pittsburgh, **2017 NIPS**)
- [Flexpoint: An Adaptive Numerical Format for Efficient Training of Deep Neural Networks.](https://arxiv.org/abs/1711.02213v2) (Intel, **2017 NIPS**)
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.](https://arxiv.org/abs/1712.05877) (Google, **2018 CVPR**)
  - *A quantization scheme to improve the tradeoff between accuracy and on-device latency, especially for MobileNet.*
- [Channel Pruning for Accelerating Very Deep Neural Networks.](https://arxiv.org/abs/1707.06168) [[code]](https://github.com/yihui-he/channel-pruning) (**2017 ICCV**)
- [To prune, or not to prune: Exploring the Efficacy of Pruning for Model Compression.](https://arxiv.org/abs/1710.01878) [[code](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/model_pruning)] (Google, **2018 ICLR Workshop**)
  - *Compare the accuracy of "large, but prune" models (large-sparse) and their "smaller, but dense" (small-dense) counterparts with identical memory footprint.*
  - *For a given number of non-zero parameters, sparse MobileNets are able to outperform dense MobileNets.*
- [Training and Inference with Integers in Deep Neural Networks.](https://arxiv.org/abs/1802.04680) [[code]](https://github.com/boluoweifenda/WAGE) (Tsinghua, **2018 ICLR**)
  - *A new method termed as "WAGE" to discretize both training and inference, where weights (W), activations (A), gradients (G) and errors (E) among layers are shifted and linearly constrained to low-bitwidth integers.*
  - *Training in hardware systems such as integer-based deep learning accelerators and neuromorphic chips with comparable accuracy and higher energy efficiency, which is crucial to future AI applications in variable scenarios with transfer and continual learning demands.*
- [AMC: AutoML for Model Compression and Acceleration on Mobile Devices](https://arxiv.org/abs/1802.03494) (CMU, Google, MIT, **2018 ECCV**)

#### arXiv Papers
- [Reduced-Precision Strategies for Bounded Memory in Deep Neural Nets.](https://arxiv.org/abs/1511.05236) (University of Toronto, University of British Columbia)
- [Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1.](http://arxiv.org/abs/1602.02830)
  - *Constrain both the weights and the activations to either +1 or -1.*
- [Quantized Neural Networks: Training Neural Networks with Low Precision Weights and Activations.](http://arxiv.org/abs/1609.07061) (Universite de Montreal)
- [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks.](http://arxiv.org/abs/1603.05279) [[code]](https://github.com/allenai/XNOR-Net) (Allen Institute for AI, University of Washington)
- [DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients.](http://arxiv.org/abs/1606.06160) (Megvii)
- [Deep Learning with Limited Numerical Precision.](https://arxiv.org/abs/1502.02551) (IBM)
- [Dynamic Network Surgery for Efficient DNNs.](http://arxiv.org/abs/1608.04493) (Intel Labs China)
- [Incremental Network Quantization: Towards Lossless CNNs with Low-Precision Weights.](https://arxiv.org/abs/1702.03044) [[code]](https://github.com/Zhouaojun/Incremental-Network-Quantization) (Intel Labs China)
- [Exploring the Regularity of Sparse Structure in Convolutional Neural Networks.](https://arxiv.org/abs/1705.08922) (Stanford, NVIDIA, Tsinghua)
  - *Coarser-grained pruning can save memory storage and access while maintaining the accuracy.*
- [A Quantization-Friendly Separable Convolution for MobileNets.](https://arxiv.org/abs/1803.08607) (Qualcomm)

### Other Topics

#### GAN
- [Generative Adversarial Nets.](https://arxiv.org/abs/1406.2661) (Universite de Montreal, **2014 NIPS**)
  - *Two "adversarial" MLP models G and D: a generative model G that captures the data distribution and a discriminative model D that estimates the probability that a sample came from the training data rather than G*.
  - *D is trained to learn the above probability*.
  - *G is trained to maximize the probability of D making a mistake.*.

#### NAS
- [Neural Architecture Search: A Survey.](https://arxiv.org/abs/1808.05377) (University of Freiburg, Bosch)

#### Others
- [You Only Look Once: Unified, Real-Time Object Detection.](https://arxiv.org/abs/1506.02640) [[code]](https://github.com/gliese581gg/YOLO_tensorflow) (University of Washington, Allen Institute for AI, Facebook AI Research, **2016 CVPR**)
- [A-Fast-RCNN: Hard positive generation via adversary for object detection.](https://arxiv.org/abs/1704.03414) (CMU, **2017 CVPR**)
- [Dilated Residual Networks](https://arxiv.org/abs/1705.09914). [[code]](https://github.com/fyu/drn) (Princeton, Intel, **2017 CVPR**)
- [Deformable Convolutional Networks.](https://arxiv.org/abs/1703.06211) [[code]](https://github.com/msracver/Deformable-ConvNets) (MSRA, **2017 ICCV**)
- [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices.](https://arxiv.org/abs/1707.01083) (Megvii, **2017 CVPR**)
- [Federated Optimization: Distributed Machine Learning for On-Device Intelligence.](https://arxiv.org/abs/1610.02527) (University of Edinburgh, Google)
- [Deep Complex Networks.](https://arxiv.org/abs/1705.09792) (Université de Montréal, Element AI)
- [One Model To Learn Them All.](https://arxiv.org/abs/1706.05137) (Google, University of Toronto)
- [Densely Connected Convolutional Networks.](https://arxiv.org/abs/1608.06993) (Cornell, Tsinghua, FAIR, **2017 CVPR**)
- [YOLO9000: Better, Faster, Stronger.](https://arxiv.org/abs/1612.08242) [[code]](https://github.com/longcw/yolo2-pytorch) (University of Washington, **2017 CVPR**)

## Industry Contributions
 - [Movidius](http://www.movidius.com/)
   - Myriad 2: Hardware-accelerated visual intelligence at ultra-low power.
   - Fathom Neural Compute Stick: The world's first discrete deep learning accelerator (Myriad 2 VPU inside).
   - Myriad X: On-device AI and computer vision.
 - NVIDIA
   - Jetson TX1: Embedded visual computing developing platform.
   - DGX-1: Deep learning supercomputer.
   - Tesla V100: A data center GPU with Tensor Cores inside.
   - [NVDLA](http://nvdla.org/): The NVIDIA Deep Learning Accelerator (NVDLA) is a free and open architecture that promotes a standard way to design deep learning inference accelerators.
 - Google
   - TPU (Tensor Processing Unit).
   - [TPUv2](https://www.nextplatform.com/2017/05/22/hood-googles-tpu2-machine-learning-clusters/): Train and run machine learning models.
   - [TPUv3](https://www.nextplatform.com/2018/05/10/tearing-apart-googles-tpu-3-0-ai-coprocessor/): Liquid cooling.
   - [Edge TPU](https://cloud.google.com/edge-tpu/): Google’s purpose-built ASIC designed to run inference at the edge.
 - [Nervana](https://www.nervanasys.com/)
   - Nervana Engine: Hardware optimized for deep learning.
 - [Wave Computing](http://wavecomp.com/)
   - Clockless **CGRA** architecture.
 - Tesla
   - [Full Self-Driving (FSD) Computer](https://youtu.be/aZK1fARxYsE) [[Chinese]](https://mp.weixin.qq.com/s?__biz=MzI3MDQ2MjA3OA==&mid=2247485026&idx=1&sn=925f2fa98f3f26b2914a1ccf2945b331&chksm=ead1fb73dda6726510644dec2c0bb9eb843c09787510f5f5d6dd58b294974fc76893d8bb4b9c&mpshare=1&scene=1&srcid=0427hbz2qxPiCZnMQwYk5pyk&key=1f3fd4f811b89077c14af3300883bc4a823452392f3297d3e59a8bb68602f817e99e79929f25fab1f3a74b164c4b79871ad67c9427d7e4f7d63ab479f8ff218c2c0955cb3dbbb807bcc5ee3076c17ba4&ascene=1&uin=MTA1MTIxMDI2Mg%3D%3D&devicetype=Windows+10&version=62060739&lang=zh_CN&pass_ticket=uOc2Mww8wyVqU3z4bdW81cO9hV4D2TmxLXDuYEbXAf%2BUyCa9xI0L%2Bz0melirJf3c)
