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
   - 2020: [ISSCC](#2020-isscc), [ISCA](#2020-isca), [MICRO](#2020-micro), [HPCA](#2020-hpca), [ASPLOS](#2020-asplos), [DAC](#2020-dac), [FPGA](#2020-fpga), [ICCAD](#2020-iccad), [VLSI](#2020-vlsi),  [HotChips](#2020-hotchips)
   - 2021: [ISSCC](#2021-isscc), [ISCA](#2021-isca), [HPCA](#2021-hpca), [ASPLOS](#2021-asplos), [VLSI](#2021-vlsi)

## My Contributions
I'm working on energy-efficient architecture design for deep learning. Some featured works are presented here. Hope my new papers will come out soon in the near future.

[Aug. 2020] I have designed a deep learning processor (Evolver) with on-device quantization-voltage-frequency (QVF) tuning. Compared with conventional QVF tuning that determines policies offline, Evolver make optimal customizations for local user scenarios.

- [**Evolver: A Deep Learning Processor with On-Device Quantization-Voltage-Frequency Tuning.**](https://ieeexplore.ieee.org/document/9209075) (**JSSC'21**)
  - Evolver contains a reinforcement learning unit (RLU) that searches QVF polices based on its direct feedbacks. An outlier-skipping scheme is proposed to save unnecessary training for invalid policies under the profiled latency and energy constraints.
  - We exploit the inherent sparsity of feature/error maps in DNN training’s feedforward and backpropagation passes, and design a bidirectional speculation unit (BSU) to capture runtime sparsity and discard zero-output computation, thus reducing training cost. The feedforward speculation also benefits the execution mode.
  - Since the runtime sparsity causes time-varying workload parallelism that harms performance and efficiency, we design a reconfigurable computing engine (RCE) with an online configuration compiler (OCC) for Evolver, in order to dynamically reconfigure dataflow parallelism to match workload parallelism.

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
This is a collection of conference papers that interest me. The emphasis is focused on, but not limited to neural networks on silicon. 

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

### 2020 HotChips
- **Google’s Training Chips Revealed: TPUv2 and TPUv3.** （Google)
- **Software Co-design for the First Wafer-Scale Processor (and Beyond).** (Cerebras)
- **Manticore: A 4096-core RISC-V Chiplet Architecture for Ultra-efficient Floating-point Computing.** (ETHZ)
- **Baidu Kunlun – An AI Processor for Diversified Workloads.** (Baidu)
- **Hanguang 800 NPU – The Ultimate AI Inference Solution for Data Centers.** (Alibaba)
- **Silicon Photonics for Artificial Intelligence Acceleration.** (Lightmatter)
- Xuantie-910: Innovating Cloud and Edge Computing by RISC-V. (Alibaba)
- A Technical Overview of the ARM Cortex-M55 and Ethos-U55: ARM’s Most Capable Processors for Endpoint AI. (ARM)
- PGMA: A Scalable Bayesian Inference Accelerator for Unsupervised Learning. (Harvard)

### 2020 VLSI
- **PNPU: A 146.52TOPS/W Deep-Neural-Network Learning Processor with Stochastic Coarse-Fine Pruning and Adaptive Input/Output/Weight Skipping.** (KAIST)
- **A 3.0 TFLOPS 0.62V Scalable Processor Core for High Compute Utilization AI Training and Inference.** (IBM)
- **A 617 TOPS/W All Digital Binary Neural Network Accelerator in 10nm FinFET CMOS.** (Intel)
- **An Ultra-Low Latency 7.8-13.6 pJ/b Reconfigurable Neural Network-Assisted Polar Decoder with Multi-Code Length Support.** (NTU)
- **A 4.45ms Low-Latency 3D Point-Cloud-Based Neural Network Processor for Hand Pose Estimation in Immersive Wearable Devices.** (KAIST)
- **A 3mm2 Programmable Bayesian Inference Accelerator for Unsupervised Machine Perception Using Parallel Gibbs Sampling in 16nm.** (Harvard)
- 1.03pW/b Ultra-Low Leakage Voltage-Stacked SRAM for Intelligent Edge Processors. (Umich)
- Z-PIM: An Energy-Efficient Sparsity-Aware Processing-In-Memory Architecture with Fully-Variable Weight Precision. (KAIST)

### 2020 MICRO
- **SuperNPU: An Extremely Fast Neural Processing Unit Using Superconducting Logic Devices.** (Kyushu University）
- **Printed Machine Learning Classifiers.** (UIUC, KIT）
- **Look-Up Table based Energy Efficient Processing in Cache Support for Neural Network Acceleration.** (PSU, Intel)
- **FReaC Cache: Folded-Logic Reconfigurable Computing in the Last Level Cache.** (UIUC, IBM)
- **Newton: A DRAM-Maker's Accelerator-in-Memory (AiM) Architecture for Machine Learning.** (Purdue, SK Hynix)
- **VR-DANN: Real-Time Video Recognition via Decoder-Assisted Neural Network Acceleration.** (SJTU)
- **Procrustes: A Dataflow and Accelerator for Sparse Deep Neural Network Training.** (University of British Columbia, Microsoft)
- **Duplo: Lifting Redundant Memory Accesses of Deep Neural Networks for GPU Tensor Cores.** (Yonsei University, EcoCloud, EPFL)
- **DUET: Boosting Deep Neural Network Efficiency on Dual-Module Architecture.** (UCSB, Alibaba)
- **ConfuciuX: Autonomous Hardware Resource Assignment for DNN Accelerators using Reinforcement Learning.** (GaTech)
- **Planaria: Dynamic Architecture Fission for Spatial Multi-Tenant Acceleration of Deep Neural Networks.** (UCSD, Bigstream, Kansas, NVIDIA, Google)
- **TFE: Energy-Efficient Transferred Filter-Based Engine to Compress and Accelerate Convolutional Neural Networks.** (THU, Alibaba)
- **MatRaptor: A Sparse-Sparse Matrix Multiplication Accelerator Based on Row-Wise Product.** (Cornell)
- **TensorDash: Exploiting Sparsity to Accelerate Deep Neural Network Training.** (Toronto)
- **SAVE: Sparsity-Aware Vector Engine for Accelerating DNN Training and Inference on CPUs.** (UIUC)
- **GOBO: Quantizing Attention-Based NLP Models for Low Latency and Energy Efficient Inference.** (Toronto)
- **TrainBox: An Extreme-Scale Neural Network Training Server Architecture by Systematically Balancing Operations.** (SNU)
- **AWB-GCN: A Graph Convolutional Network Accelerator with Runtime Workload Rebalancing.** (Boston et al.)
- **Mesorasi: Architecture Support for Point Cloud Analytics via Delayed-Aggregation.** (Rochestor, ARM)
- **NCPU: An Embedded Neural CPU Architecture on Resource-Constrained Low Power Devices for Real-Time End-to-End Performance.** (Northwestern University)
- FlexWatts: A Power- and Workload-Aware Hybrid Power Delivery Network for Energy-Efficient Microprocessors.	(ETHZ, Intel, Technion, NTU)
- AutoScale: Energy Efficiency Optimization for Stochastic Edge Inference Using Reinforcement Learning.	(Facebook)
- CATCAM: Constant-time Alteration Ternary CAM with Scalable In-Memory Architecture.	(THU, Southeast University)
- DUAL: Acceleration of Clustering Algorithms using Digital-Based Processing In-Memory.	(UCSD)
- Bit-Exact ECC Recovery (BEER): Determining DRAM On-Die ECC Functions by Exploiting DRAM Data Retention Characteristics.	(ETHZ)

### 2020 ICCAD
- ReTransformer: ReRAM-based Processing-in-Memory Architecture for Transformer Acceleration.	(Duke)
- Energy-efficient XNOR-free In-Memory BNN Accelerator with Input Distribution Regularization.	(POSTECH)
- HyperTune: Dynamic Hyperparameter Tuning for Efficient Distribution of DNN Training Over Heterogeneous Systems.	(UCI, NGD)
- SynergicLearning: Neural Network-Based Feature Extraction for Highly-Accurate Hyperdimensional Learning.	(USC)
- Optimizing Stochastic Computing for Low Latency Inference of Convolutional Neural Networks.	(Nanjing University)
- HAPI: Hardware-Aware Progressive Inference.	(Samsung)
- MobiLattice: A Depth-wise DCNN Accelerator with Hybrid Digital/Analog Nonvolatile Processing-In-Memory Block.	(PKU, Duke)
- A Many-Core Accelerator Design for On-Chip Deep Reinforcement Learning.	(ICT)
- DRAMA: An Approximate DRAM Architecture for High-performance and Energy-efficient Deep Training System.	(Kyung Hee Univ., NUS)
- FPGA-based Low-Batch Training Accelerator for Modern CNNs Featuring High Bandwidth Memory.	(ASU, Intel)

### 2021 ISSCC
- **The A100 Datacenter GPU and Ampere Architecture.** (NVIDIA）
- **Kunlun: A 14nm High-Performance AI Processor for Diversified Workloads.** (Baidu）
- **A 12nm Autonomous-Driving Processor with 60.4TOPS, 13.8TOPS/W CNN Executed by Task-Separated ASIL D Control.** (Renesas）
- **BioAIP: A Reconfigurable Biomedical AI Processor with Adaptive Learning for Versatile Intelligent Health Monitoring.** (UESTC）
- **A 0.2-to-3.6TOPS/W Programmable Convolutional Imager SoC with In-Sensor Current-Domain Ternary-Weighted MAC Operations for Feature Extraction and Region-of-Interest Detection.** (Leuven）
- **A 7nm 4-Core AI Chip with 25.6TFLOPS Hybrid FP8 Training, 102.4TOPS INT4 Inference and Workload-Aware Throttling.** (IBM）
- **A 28nm 12.1TOPS/W Dual-Mode CNN Processor Using Effective-Weight-Based Convolution and Error-Compensation-Based Prediction.** (THU）
- **A 40nm 4.81TFLOPS/W 8b Floating-Point Training Processor for Non-Sparse Neural Networks Using Shared Exponent Bias and 24-Way Fused Multiply-Add Tree.** (SNU）
- **PIU: A 248GOPS/W Stream-Based Processor for Irregular Probabilistic Inference Networks Using Precision-Scalable Posit Arithmetic in 28nm.** (Leuven）
- **A 6K-MAC Feature-Map-Sparsity-Aware Neural Processing Unit in 5nm Flagship Mobile SoC.** (Samsung）
- **A 1/2.3inch 12.3Mpixel with On-Chip 4.97TOPS/W CNN Processor Back-Illuminated Stacked CMOS Image Sensor.** (Sony）
- **A 184μW Real-Time Hand-Gesture Recognition System with Hybrid Tiny Classifiers for Smart Wearable Devices.** (Nanyang）
- **A 25mm2 SoC for IoT Devices with 18ms Noise-Robust Speech-to-Text Latency via Bayesian Speech Denoising and Attention-Based Sequence-to-Sequence DNN Speech Recognition in 16nm FinFET.** (Harvard, Tufts, ARM, Cornell）
- **A Background-Noise and Process-Variation-Tolerant 109nW Acoustic Feature Extractor Based on Spike-Domain Divisive-Energy Normalization for an Always-On Keyword Spotting Device.** (Columnbia）
- A 148nW General-Purpose Event-Driven Intelligent Wake-Up Chip for AIoT Devices Using Asynchronous Spike-Based Feature Extractor and Convolutional Neural Network. (PKU）
- A Programmable Neural-Network Inference Accelerator Based on Scalable In-Memory Computing. (Princeton）
- A 2.75-to-75.9TOPS/W Computing-in-Memory NN Processor Supporting Set-Associate Block-Wise Zero Skipping and Ping-Pong CIM with Simultaneous Computation and Weight Updating. (THU）
- A 65nm 3T Dynamic Analog RAM-Based Computing-in-Memory Macro and CNN Accelerator with Retention Enhancement, Adaptive Analog Sparsity and 44TOPS/W System Energy Efficiency. (Northwestern）
- A 5.99-to-691.1TOPS/W Tensor-Train In-Memory-Computing Processor Using Bit-Level-SparsityBased Optimization and Variable-Precision Quantization. (THU, UESTC, NTHU）
- A 22nm 4Mb 8b-Precision ReRAM Computing-in-Memory Macro with 11.91 to 195.7TOPS/W for Tiny AI Edge Devices. (NTHU, TSMC）
- eDRAM-CIM: Compute-In-Memory Design with Reconfigurable Embedded-Dynamic-Memory Array Realizing Adaptive Data Converters and Charge-Domain Computing. (UT Austin, Intel）
- A 28nm 384kb 6T-SRAM Computation-in-Memory Macro with 8b of Precision for AI Edge Chips. (NTHU, Industrial Technology Research Institute, TSMC）
- An 89TOPS/W and 16.3TOPS/mm2 All-Digital SRAM-Based Full-Precision Compute-In Memory Macro in 22nm for Machine-Learning Edge Applications. (TSMC）
- A 20nm 6GB Function-In-Memory DRAM, Based on HBM2 with a 1.2TFLOPS Programmable Computing Unit Using Bank-Level Parallelism, for Machine Learning Applications. (Samsung）
- A 21×21 Dynamic-Precision Bit-Serial Computing Graph Accelerator for Solving Partial Differential Equations Using Finite Difference Method. (Nanyang）

### 2021 ASPLOS
- **Exploiting Gustavson's Algorithm to Accelerate Sparse Matrix Multiplication.**	(MIT, NVIDIA)
- **SIMDRAM: A Framework for Bit-Serial SIMD Processing using DRAM.**	(ETHZ, CMU)
- **RecSSD: Near Data Processing for Solid State Drive Based Recommendation Inference.**	(Harvard, Facebook, ASU)
- DiAG: A Dataflow-inspired Architecture for General-purpose Processors.	(UIUC)
- Field-Configurable Multi-resolution Inference: Rethinking Quantization.	(Harvard, Franklin & Marshall College)
- Defensive Approximation: Securing CNNs using Approximate Computing.	(University of Sfax et al.)

### 2021 HPCA
- **A Computational Stack for Cross-Domain Acceleration.**	(UCSD et al.)
- **Heterogeneous Dataflow Accelerators for Multi-DNN Workloads.**	(GaTech, Facebook, NVIDIA)
- **SPAGHETTI: Streaming Accelerators for Highly Sparse GEMM on FPGAs.**	(SFU et al.)
- **SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning.**	(MIT)
- **Mix and Match: A Novel FPGA-Centric Deep Neural Network Quantization Framework.**	(Northeastern et al.)
- **Tensor Casting: Co-Designing Algorithm-Architecture for Personalized Recommendation Training.**	(KAIST)
- **GradPIM: A Practical Processing-in-DRAM Architecture for Gradient Descent.**	(SNU, Yonsei)
- **SpaceA: Sparse Matrix Vector Multiplication on Processing-in-Memory Accelerator.**	(UCSB, PKU)
- **Layerweaver: Maximizing Resource Utilization of Neural Processing Units via Layer-Wise Scheduling.**	(Sungkyunkwan, SNU)
- **Efficient Tensor Migration and Allocation on Heterogeneous Memory Systems for Deep Learning.**	(UCM, Microsoft)
- **CSCNN: Algorithm-hardware Co-design for CNN Accelerators using Centrosymmetric Filters.**	(GWU, Ohio)
- **Adapt-NoC: A Flexible Network-on-Chip Design for Heterogeneous Manycore Architectures.**	(GWU)
- **GCNAX: A Flexible and Energy-efficient Accelerator for Graph Convolutional Neural Networks.**	(GWU, Ohio)
- **Ascend: a Scalable and Unified Architecture for Ubiquitous Deep Neural Network Computing.**	(Huawei)
- **Understanding Training Efficiency of Deep Learning Recommendation Models at Scale.**	(Facebook)
- **Eudoxus: Characterizing and Accelerating Localization in Autonomous Machines.**	(Rochester et al.)
- **NeuroMeter: An Integrated Power, Area, and Timing Modeling Framework for Machine Learning Accelerators.**	(UCSB. Google)
- **Chasing Carbon: The Elusive Environmental Footprint of Computing.**	(Harvard, Facebook)
- **FuseKNA: Fused Kernel Convolution based Accelerator for Deep Neural Networks.**	(THU)
- **FAFNIR: Accelerating Sparse Gathering by Using Efficient Near-Memory Intelligent Reduction.**	(GaTech)
- **VIA: A Smart Scratchpad for Vector Units With Application to Sparse Matrix Computations.**	(Barcelona Supercomputing Center et al.)
- Cheetah: Optimizing and Accelerating Homomorphic Encryption for Private Inference.	(NYU, SNU, Harvard, Facebook)
- CAPE: A Content-Addressable Processing Engine.	(Cornell, PSU)
- Prodigy: Improving the Memory Latency of Data-Indirect Irregular Workloads Using Hardware-Software Co-Design.	(Umich et al.)
- BRIM: Bistable Resistively-Coupled Ising Machine.	(Rochester)
- An Analog Preconditioner for Solving Linear Systems.	(Sandia et al.)

### 2021 ISCA
- Ten Lessons From Three Generations Shaped Google's TPUv4i (Google)
- Sparsity-Aware and Re-Configurable NPU Architecture for Samsung Flagship Mobile SoC (Samsung)
- Energy Efficiency Boost in the AI-Infused POWER10 Processor (IBM)
- Hardware Architecture and Software Stack for PIM Based on Commercial DRAM Technology (Samsung)
- Pioneering Chiplet Technology and Design for the AMD EPYC™ and Ryzen™ Processor Families (AMD)
- RaPiD: AI Accelerator for Ultra-Low Precision Training and Inference (IBM)
- REDUCT: Keep It Close, Keep It Cool! - Scaling DNN Inference on Multi-Core CPUs with Near-Cache Compute (Intel)
- Communication Algorithm-Architecture Co-Design for Distributed Deep Learning (UCSB, TAMU)
- ABC-DIMM: Alleviating the Bottleneck of Communication in DIMM-Based Near-Memory Processing with Inter-DIMM Broadcast (THU)
- Sieve: Scalable In-Situ DRAM-Based Accelerator Designs for Massively Parallel k-mer Matching (Virginia)
- FORMS: Fine-Grained Polarized ReRAM-Based In-Situ Computation for Mixed-Signal DNN Accelerator (Northeastern et al)
- BOSS: Bandwidth-Optimized Search Accelerator for Storage-Class Memory (SNU)
- Accelerated Seeding for Genome Sequence Alignment with Enumerated Radix Trees (Umich)
- Aurochs: An Architecture for Dataflow Threads (Stanford)
- PipeZK: Accelerating Zero-Knowledge Proof with a Pipelined Architecture (PKU et al)
- CODIC: A Low-Cost Substrate for Enabling Custom In-DRAM Functionalities and Optimizations (ETHZ)
- Enabling Compute-Communication Overlap in Distributed Deep Learning Training Platforms (GaTech)
- CoSA: Scheduling by Constrained Optimization for Spatial Accelerators (Berkeley)
- η-LSTM: Co-Designing Highly-Efficient Large LSTM Training via Exploiting Memory-Saving and Architectural Design Opportunities (Washington et al)
- FlexMiner: A Pattern-Aware Accelerator for Graph Pattern Mining (MIT)
- PolyGraph: Exposing the Value of Flexibility for Graph Processing Accelerators (UCLA)
- Large-Scale Graph Processing on FPGAs with Caches for Thousands of Simultaneous Misses (EPFL)
- SPACE: Locality-Aware Processing in Heterogeneous Memory for Personalized Recommendations (Yonsei)
- ELSA: Hardware-Software Co-Design for Efficient, Lightweight Self-Attention Mechanism in Neural Networks (SNU)
- Cambricon-Q: A Hybrid Architecture for Efficient Training (CAS)
- TENET: A Framework for Modeling Tensor Dataflow Based on Relation-Centric Notation (PKU et al)
- NASGuard: A Novel Accelerator Architecture for Robust Neural Architecture Search (NAS) Networks (CAS)
- NASA: Accelerating Neural Network Design with a NAS Processor (CAS)
- Albireo: Energy-Efficient Acceleration of Convolutional Neural Networks via Silicon Photonics (Ohio et al)
- QUAC-TRNG: High-Throughput True Random Number Generation Using Quadruple Row Activation in Commodity DRAM Chips (ETHZ)
- NN-Baton: DNN Workload Orchestration and Chiplet Granularity Exploration for Multichip Accelerators (THU)
- SNAFU: An Ultra-Low-Power, Energy-Minimal CGRA-Generation Framework and Architecture (CMU)
- SARA: Scaling a Reconfigurable Dataflow Accelerator (Stanford)
- HASCO: Towards Agile HArdware and Software CO-design for Tensor Computation (PKU et al)
- SpZip: Architectural Support for Effective Data Compression In Irregular Applications (MIT)
- Dual-Side Sparse Tensor Core (Microsoft）
- RingCNN: Exploiting Algebraically-Sparse Ring Tensors for Energy-Efficient CNN-Based Computational Imaging (NTHU)
- GoSPA: An Energy-Efficient High-Performance Globally Optimized SParse Convolutional Neural Network Accelerator (Rutgers)

### 2021 VLSI
- MN-Core - A Highly Efficient and Scalable Approach to Deep Learning (Preferred Networks)
- CHIMERA: A 0.92 TOPS, 2.2 TOPS/W Edge AI Accelerator with 2 MByte On-Chip Foundry Resistive RAM for Efficient Training and Inference	(Standford, TSMC)
- OmniDRL: A 29.3 TFLOPS/W Deep Reinforcement Learning Processor with Dual-Mode Weight Compression and On-Chip Sparse Weight Transposer	(KAIST)
- DepFiN: A 12nm, 3.8TOPs Depth-First CNN Processor for High Res. Image Processing	(Leuven)
- PNNPU: A 11.9 TOPS/W High-Speed 3D Point Cloud-Based Neural Network Processor with Block-Based Point Processing for Regular DRAM Access	(KAIST)
- A 28nm 276.55TFLOPS/W Sparse Deep-Neural-Network Training Processor with Implicit Redundancy Speculation and Batch Normalization Reformulation	(THU)
- A 13.7 TFLOPSW Floating-point DNN Processor using Heterogeneous Computing Architecture with Exponent-Computing-in-Memory	(KAIST)
- PIMCA: A 3.4-Mb Programmable In-Memory Computing Accelerator in 28nm for On-Chip DNN Inference	(ASU)
- A 6.54-to-26.03 TOPS/W Computing-In-Memory RNN Processor Using Input Similarity Optimization and Attention-Based Context-Breaking with Output Speculation	(THU, NTHU)
- Fully Row/Column-Parallel In-Memory Computing SRAM Macro Employing Capacitor-Based Mixed-Signal Computation with 5-b Inputs	(Princeton)
- HERMES Core – A 14nm CMOS and PCM-Based In-Memory Compute Core Using an Array of 300ps/LSB Linearized CCO-Based ADCs and Local Digital Processing	(IBM)
- A 20x28 Spins Hybrid In-Memory Annealing Computer Featuring Voltage-Mode Analog Spin Operator for Solving Combinatorial Optimization Problems	(NTU, UCSB)
- Analog In-Memory Computing in FeFET-Based 1T1R Array for Edge AI Applications	(Sony)
- Energy-Efficient Reliable HZO FeFET Computation-in-Memory with Local Multiply & Global Accumulate Array for Source-Follower & Charge-Sharing Voltage Sensing	(Tokyo)
