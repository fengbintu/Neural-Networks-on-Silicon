# Neural Networks on Silicon

My name is Fengbin Tu. I'm currently pursuing my Ph.D. degree with the Institute of Microelectronics, Tsinghua University, Beijing, China. For more informantion about me and my research, you can go to [my homepage](https://fengbintu.github.io/). One of my research interests is architecture design for deep learning. This is an exciting field where fresh ideas come out every day, so I'm collecting works on related topics. Welcome to join us!

## Table of Contents
 - [My Contributions](#my-contributions)
 - [Conference Papers](#conference-papers)
   - [2015 DAC](#2015-dac)
   - [2016 DAC](#2016-dac)
   - [2016 ISSCC](#2016-isscc)
   - [2016 ISCA](#2016-isca)
   - [2016 DATE](#2016-date)
   - [2016 FPGA](#2016-fpga)
   - [2016 ASPDAC](#2016-aspdac)
   - [2016 VLSI](#2016-vlsi)
   - [2016 ICCAD](#2016-iccad) 
   - [2016 MICRO](#2016-micro)
   - [2016 FPL](#2016-fpl)
   - [2016 HPCA](#2016-hpca)
   - [2017 FPGA](#2017-fpga)
   - [2017 ISSCC](#2017-isscc)
   - [2017 HPCA](#2017-hpca)
   - [2017 ASPLOS](#2017-asplos)
   - [2017 ISCA](#2017-isca)
   - [2017 FCCM](#2017-fccm)
   - [2017 DAC](#2017-dac)
   - [2017 DATE](#2017-date)
   - [2017 VLSI](#2017-vlsi)
   - [2017 ICCAD](#2017-iccad)
   - [2017 HotChips](#2017-hotchips)
   - [2017 MICRO](#2017-micro)
   - [2018 ASPDAC](#2018-aspdac)
   - [2018 ISSCC](#2018-isscc)
   - [2018 HPCA](#2018-hpca)
   - [2018 ASPLOS](#2018-asplos)
   
 - [Important Topics](#important-topics)
   - [Tutorial and Survey](#tutorial-and-survey) 
   - [Benchmarks](#benchmarks)
   - [Network Compression](#network-compression)
   - [Other Topics](#other-topics)

 - [Industry Contributions](#industry-contributions)

## My Contributions
I'm working on energy-efficient architecture design for deep learning. A deep convoultional neural network architecture (DNA) has been designed with 1~2 orders higher energy efficiency over the state-of-the-art works. I'm trying to improve the architecture for ultra low-power compting. Hope my new works will come out soon in the near future.

- [**Deep Convolutional Neural Network Architecture with Reconfigurable Computation Patterns.**](http://ieeexplore.ieee.org/document/7898402/) (**TVLSI**)
  - *This is the first work to assign Input/Output/Weight Reuse to different layers of a CNN, which optimizes system-level energy consumption based on different CONV parameters.*
  - *A 4-level CONV engine is designed to to support different tiling parameters for higher resource utilization and performance.*
  - *A layer-based scheduling framework is proposed to optimize both system-level energy efficiency and performance.*

## Conference Papers
This is a collection of conference papers that interest me. The emphasis is focused on, but not limited to neural networks on silicon. Papers of significance are marked in **bold**. My comments are marked in *italic*.
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
- **Scalable and Efficient Neural Network Acceleration with 3D Memory.** (Stanford University)
  - *Use more area for PEs and less area for SRAM buffers.* 
  - *Move portions of the NN computations close to the DRAM banks.*
  - *3D memory simplifies dataflow scheduling.*
  - *Develop a hybrid partitioning scheme that parallelizes the NN computations over multiple accelerators.*
- SC-DCNN: Highly-Scalable Deep Convolutional Neural Network using Stochastic Computing. (Syracuse University, USC, The City College of New York)

### 2017 ISCA
- **Maximizing CNN Accelerator Efficiency Through Resource Partitioning.** (Stony Brook University)
  - *An Extension of their FPL'16 paper.* 
- **In-Datacenter Performance Analysis of a Tensor Processing Unit.** (Google)
- **SCALEDEEP: A Scalable Compute Architecture for Learning and Evaluating Deep Networks.** (Purdue University, Intel)
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
- **Understanding the Impact of Precision Quantization on the Accuracy and Energy of Neural Networks.** (Brown University)
- **Design Space Exploration of FPGA Accelerators for Convolutional Neural Networks.** (Samsung, UNIST, Seoul National University)
- **MoDNN: Local Distributed Mobile Computing System for Deep Neural Network.** (University of Pittsburgh, George Mason University, University of Maryland)
- **Chain-NN: An Energy-Efficient 1D Chain Architecture for Accelerating Deep Convolutional Neural Networks.** (Waseda University)
- **LookNN: Neural Network with No Multiplication.** (UCSD)
- Energy-Efficient Approximate Multiplier Design using Bit Significance-Driven Logic Compression. (Newcastle University)
- Revamping Timing Error Resilience to Tackle Choke Points at NTC Systems. (Utah State University)

### 2017 VLSI
- **A 3.43TOPS/W 48.9pJ/Pixel 50.1nJ/Classification 512 Analog Neuron Sparse Coding Neural Network with On-Chip Learning and Classification in 40nm CMOS.** (University of Michigan, Intel)
- **BRein Memory: A 13-Layer 4.2 K Neuron/0.8 M Synapse Binary/Ternary Reconfigurable In-Memory Deep Neural Network Accelerator in 65 nm CMOS.** (Hokkaido University, Tokyo Institute of Technology, Keio University)
- **A 1.06-To-5.09 TOPS/W Reconfigurable Hybrid-Neural-Network Processor for Deep Learning Applications.** (Tsinghua University)
- **A 127mW 1.63TOPS sparse spatio-temporal cognitive SoC for action classification and motion tracking in videos.** (University of Michigan)

### 2017 ICCAD
- **AEP: An Error-bearing Neural Network Accelerator for Energy Efficiency and Model Protection.** (University of Pittsburgh)
- VoCaM: Visualization oriented convolutional neural network acceleration on mobile system.
- AdaLearner: An Adaptive Distributed Mobile Learning System for Neural Networks.
- MeDNN: A Distributed Mobile System with Enhanced Partition and Deployment for Large-Scale DNNs.
- TraNNsformer: Neural Network Transformation for Efficient Crossbar Based Neuromorphic System Design.
- A Closed-loop Design to Enhance Weight Stability of Memristor Based Neural Network Chips.
- Fault injection attack on deep neural network.
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
- **Distributed FPGA Acceleration for Learning.** (Georgia Tech, UCSD)
- **Bit-Pragmatic Deep Neural Network Computing.** (NVIDIA, University of Toronto)
- **CirCNN: Accelerating and Compressing Deep Neural Networks Using Block-Circulant Weight Matrices.** (Syracuse University, City University of New York, USC, California State University, Northeastern University)
- **Addressing Compute and Memory Bottlenecks for DNN Execution on GPUs.** (University of Michigan)
- DRISA: A DRAM-based Reconfigurable In-Situ Accelerator. (UCSB, Samsung)

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
- **UNPU: A 50.6TOPS/W Unified Deep Neural Network Accelerator with 1bto-16b Fully-Variable Weight Bit-Precision.** (KAIST)
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
- **VIBNN: Hardware Acceleration of Bayesian Neural Networks.** (Syracuse University, USC)
- Exploiting Dynamical Thermal Energy Harvesting for Reusing in Smartphone with Mobile Applications. (Guizhou University, University of Florida)
- Potluck: Cross-application Approximate Deduplication for Computation-Intensive Mobile Applications. (Yale)

## Important Topics
This is a collection of papers on other important topics related to neural networks. Papers of significance are marked in **bold**. My comments are in marked in *italic*.

### Tutorial and Survey
- [Tutorial on Hardware Architectures for Deep Neural Networks.](http://eyeriss.mit.edu/tutorial.html) (MIT)
- [A Survey of Neuromorphic Computing and Neural Networks in Hardware.](https://arxiv.org/abs/1705.06963) (Oak Ridge National Lab)
- [A Survey of FPGA Based Neural Network Accelerator.](https://arxiv.org/abs/1712.08934) (Tsinghua)

### Benchmarks
- [Fathom: Reference Workloads for Modern Deep Learning Methods.](http://arxiv.org/abs/1608.06581) (Harvard University)
- **AlexNet**: Imagenet Classification with Deep Convolutional Neural Networks. (University of Toronto, **2012 NIPS**)
- **Network in Network**. (National University of Singapore, **2014 ICLR**)
- **ZFNet**: Visualizing and Understanding Convolutional Networks. (New York University, **2014 ECCV**)
- **OverFeat**: Integrated Recognition, Localization and Detection using Convolutional Networks. (New York University, **2014 CVPR**)
- **VGG**: Very Deep Convolutional Networks for Large-Scale Image Recognition. (Univerisity of Oxford, **2015 ICLR**)
- **GoogLeNet**: Going Deeper with Convolutions. (Google, University of North Carolina, University of Michigan, **2015 CVPR**)
- **ResNet**: Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. (MSRA, **2015 ICCV**)

### Network Compression
#### Conference Papers
- Learning both Weights and Connections for Efficient Neural Network. (Stanford University, NVIDIA, **2015 NIPS**)
  - *Prune connections by thresholding weight values.* 
  - *Retain accuracy with iterative retraining.*
- [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding.](http://arxiv.org/abs/1510.00149) (Stanford University, Tsinghua University, **2016 ICLR**)
- [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size.](http://arxiv.org/abs/1602.07360) (DeepScale & UC Berkeley, Stanford University)
- [8-Bit Approximations for Parallelism in Deep Learning.](http://arxiv.org/abs/1511.04561) (Universia della Svizzera italiana, **2016 ICLR**)
- [Neural Networks with Few Multiplications.](https://arxiv.org/abs/1510.03009) (Universite de Montreal, **2016 ICLR**)
- [Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications.](http://arxiv.org/abs/1511.06530) (Samsung, Seoul National University, **2016 ICLR**)
- [Hardware-oriented Approximation of Convolutional Neural Networks.](https://arxiv.org/abs/1604.03168) (UC Davis, **2016 ICLR Workshop**)
- [Soft Weight-Sharing for Neural Network Compression.](https://arxiv.org/abs/1702.04008) (University of Amsterdam, CIFAR, **2017 ICLR**)
- Designing Energy-Efficient Convolutional Neural Networks using Energy-Aware Pruning. (MIT, **2017 CVPR**)
  - *Estimate the energy comsuption of a CNN based on their Eyeriss (ISCA'16) paper.* 
  - *Propose an energy-aware pruning method.*
- Scalable and Sustainable Deep Learning via Randomized Hashing. (Rice University, **2017 KDD**)
- [TernGrad: Ternary Gradients to Reduce Communication in Distributed Deep Learning.](https://arxiv.org/abs/1705.07878) (Duke University, Hewlett Packard Labs, University of Nevada – Reno, University of Pittsburgh, **2017 NIPS**)
- [Flexpoint: An Adaptive Numerical Format for Efficient Training of Deep Neural Networks.](https://arxiv.org/abs/1711.02213v2) (Intel, **2017 NIPS**)

#### arXiv Papers
- [Reduced-Precision Strategies for Bounded Memory in Deep Neural Nets.](https://arxiv.org/abs/1511.05236) (University of Toronto, University of British Columbia)
- [Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1.](http://arxiv.org/abs/1602.02830)
  - *Constrain both the weights and the activations to either +1 or -1.*
- [Quantized Neural Networks: Training Neural Networks with Low Precision Weights and Activations.](http://arxiv.org/abs/1609.07061) (Universite de Montreal)
- [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks.](http://arxiv.org/abs/1603.05279) (Allen Institute for AI, University of Washington)
- [DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients.](http://arxiv.org/abs/1606.06160) (Megvii)
- [Deep Learning with Limited Numerical Precision.](https://arxiv.org/abs/1502.02551) (IBM)
- [Dynamic Network Surgery for Efficient DNNs.](http://arxiv.org/abs/1608.04493) (Intel Labs China)
- [Incremental Network Quantization: Towards Lossless CNNs with Low-Precision Weights.](https://arxiv.org/abs/1702.03044) (Intel Labs China)
- [Exploring the Regularity of Sparse Structure in Convolutional Neural Networks.](https://arxiv.org/abs/1705.08922) (Stanford, NVIDIA, Tsinghua)
  - *Coarser-grained pruning can save memory storage and access while maintaining the accuracy.* 

### Other Topics

#### GAN
- [Generative Adversarial Nets.](https://arxiv.org/abs/1406.2661) (Universite de Montreal, **2014 NIPS**)
  - *Two "adversarial" MLP models G and D: a generative model G that captures the data distribution and a discriminative model D that estimates the probability that a sample came from the training data rather than G*.
  - *D is trained to learn the above probability*.
  - *G is trained to maximize the probability of D making a mistake.*.
  
#### Kaiming He
- [Mask R-CNN.](https://arxiv.org/abs/1703.06870) (FAIR)

#### Others
- You Only Look Once: Unified, Real-Time Object Detection. (University of Washington, Allen Institute for AI, Facebook AI Research, **2016 CVPR**)
- A-Fast-RCNN: Hard positive generation via adversary for object detection. (CMU, **2017 CVPR**)
- Dilated Residual Networks. (Princeton, Intel, **2017 CVPR**)
- [Deformable Convolutional Networks.](https://arxiv.org/abs/1703.06211) (MSRA)
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.](https://arxiv.org/abs/1704.04861) (Google)
- [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices.](https://arxiv.org/abs/1707.01083) (Megvii)
- [Federated Optimization: Distributed Machine Learning for On-Device Intelligence.](https://arxiv.org/abs/1610.02527) (University of Edinburgh, Google)
- [Deep Complex Networks.](https://arxiv.org/abs/1705.09792) (Université de Montréal, Element AI)
- [One Model To Learn Them All.](https://arxiv.org/abs/1706.05137) (Google, University of Toronto)
- [Densely Connected Convolutional Networks.](https://arxiv.org/abs/1608.06993) (Cornell, Tsinghua, FAIR)

## Industry Contributions
 - [Movidius](http://www.movidius.com/)
   - Myriad 2: Hardware-accelerated visual intelligence at ultra-low power.
   - Fathom Neural Compute Stick: The world's first discrete deep learning accelerator (Myriad 2 VPU inside).
   - Myriad X: On-device AI and computer vision.
 - [NVIDIA](http://www.nvidia.com/)
   - Jetson TX1: Embedded visual computing developing platform.
   - DGX-1: Deep learning supercomputer.
   - Tesla V100: A Data center GPU with Tensor Cores inside.
   - [NVDLA](http://nvdla.org/): The NVIDIA Deep Learning Accelerator (NVDLA) is a free and open architecture that promotes a standard way to design deep learning inference accelerators.
 - Google
   - TPU (Tensor Processing Unit).
   - Cloud TPU: Train and run machine learning models.
 - [Nervana](https://www.nervanasys.com/)
   - Nervana Engine: Hardware optimized for deep learning.
 - [Wave Computing](http://wavecomp.com/)
   - Clockless **CGRA** architecture.


