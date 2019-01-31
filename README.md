# Neural Networks on Silicon

My name is Fengbin Tu. I'm currently pursuing my Ph.D. degree with the Institute of Microelectronics, Tsinghua University, Beijing, China. For more informantion about me and my research, you can go to [my homepage](https://fengbintu.github.io/). One of my research interests is architecture design for deep learning. This is an exciting field where fresh ideas come out every day, so I'm collecting works on related topics. Welcome to join us!

## Table of Contents
 - [My Contributions](#my-contributions)
 - [Conference Papers](#conference-papers)
   - 2014: [ASPLOS](#2014-asplos), [MICRO](#2014-micro)
   - 2015: [ISCA](#2015-isca), [ASPLOS](#2015-asplos), [FPGA](#2015-fpga), [DAC](#2015-dac)
   - 2016: [ISSCC](#2016-isscc), [ISCA](#2016-isca), [MICRO](#2016-micro), [HPCA](#2016-hpca), [DAC](#2016-dac), [FPGA](#2016-fpga), [ICCAD](#2016-iccad), [DATE](#2016-date), [ASPDAC](#2016-aspdac), [VLSI](#2016-vlsi), [FPL](#2016-fpl)
   - 2017: [ISSCC](#2017-isscc), [ISCA](#2017-isca), [MICRO](#2017-micro), [HPCA](#2017-hpca), [ASPLOS](#2017-asplos), [DAC](#2017-dac), [FPGA](#2017-fpga), [ICCAD](#2017-iccad), [DATE](#2017-date), [VLSI](#2017-vlsi), [FCCM](#2017-fccm), [HotChips](#2017-hotchips)
   - 2018: [ISSCC](#2018-isscc), [ISCA](#2018-isca), [MICRO](#2018-micro), [HPCA](#2018-hpca), [ASPLOS](#2018-asplos), [DAC](#2018-dac), [FPGA](#2018-fpga), [ICCAD](#2018-iccad), [DATE](#2018-date), [ASPDAC](#2018-aspdac), [VLSI](#2018-vlsi), [HotChips](#2018-hotchips)
   - 2019: [ASPDAC](#2019-aspdac)

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
- **DNPU: An 8.1TOPS/W Reconfigurable CNN-RNN Processor for GeneralPurpose Deep Neural Networks.** (KAIST)
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
- **Addressing Compute and Memory Bottlenecks for DNN Execution on GPUs.** (University of Michigan)
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
 - [NVIDIA](http://www.nvidia.com/)
   - Jetson TX1: Embedded visual computing developing platform.
   - DGX-1: Deep learning supercomputer.
   - Tesla V100: A data center GPU with Tensor Cores inside.
   - [NVDLA](http://nvdla.org/): The NVIDIA Deep Learning Accelerator (NVDLA) is a free and open architecture that promotes a standard way to design deep learning inference accelerators.
 - Google
   - TPU (Tensor Processing Unit).
   - [TPUv2](https://www.nextplatform.com/2017/05/22/hood-googles-tpu2-machine-learning-clusters/): Train and run machine learning models.
   - [TPUv3](https://www.nextplatform.com/2018/05/10/tearing-apart-googles-tpu-3-0-ai-coprocessor/): Liquid cooling.
 - [Nervana](https://www.nervanasys.com/)
   - Nervana Engine: Hardware optimized for deep learning.
 - [Wave Computing](http://wavecomp.com/)
   - Clockless **CGRA** architecture.
