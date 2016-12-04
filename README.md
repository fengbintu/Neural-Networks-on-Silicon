# Neural Networks on Silicon

My name is Fengbin Tu. I'm currently pursuing the Ph.D. degree with the Institute of Microelectronics, Tsinghua University, Beijing, China. My research interests include accelerators for neural networks, deep learning and approximate computing. This is an exciting field where fresh ideas come out every day. Welcome to join us!

## Table of Contents
 - Conference Papers
   - [2015 DAC](#2015-dac)
   - [2016 DAC](#2016-dac)
   - [2016 ISSCC](#2016-isscc)
   - [2016 ISCA](#2016-isca)
   - [2016 DATE](#2016-date)
   - [2016 FPGA](#2016-fpga)
   - [2016 ASPDAC](#2016-aspdac)
   - [2016 VLSI] (#2016-vlsi)
   - [2016 ICCAD] (#2016-iccad) 
   - [2016 MICRO] (#2016-micro)
   - [2016 FPL] (#2016-fpl)
 - Important Topics
   - [Benchmarks](#benchmarks)
   - [Network Compression](#network-compression)
   - [Other Topics](#other-topics)

 - Research Groups
 - Industry Contributions

## Conference Papers
This is a collection of conference papers that interest me. The emphasis is focused on, but not limited to neural networks on silicon. Papers of significance are marked in **bold**. My comments are in marked in *italic*.
### 2015 DAC
- Reno: A Highly-Efficient Reconfigurable Neuromorphic Computing Accelerator Design. (Universtiy of Pittsburgh, Tsinghua University et al.)
- Scalable Effort Classifiers for Energy Efficient Machine Learning. (Purdue University, Microsoft Research)
- Design Methodology for Operating in Near-Threshold Computing (NTC) Region. (AMD)
- Opportunistic Turbo Execution in NTC: Exploiting the Paradigm Shift in Performance Bottlenecks. (Utah State University)

### 2016 DAC
- **DeepBurning: Automatic Generation of FPGA-based Learning Accelerators for the Neural Network Family.** (Chinese Academy of Sciences)
- **C-Brain:A Deep Learning Accelerator that Tames the Diversity of CNNs through Adaptive Data-Level Parallelization.** (Chinese Academy of Sciences)
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
- **vDNN: Virtualized Deep Neural Networks for Scalable, Memory-Efficient Neural Network Design.** (NVIDIA)
- **Stripes: Bit-Serial Deep Neural Network Computing.** (University of Toronto, University of British Columbia)
- **An Accelerator for Sparse Neural Networks.** (Chinese Academy of Sciences)
- **NEUTRAMS: Neural Network Transformation and Co-design under Neuromorphic Hardware Constraints.** (Tsinghua University, UCSB)
- **Fused-Layer CNN Accelerators.** (Stony Brook University)
- **Bridging the I/O Performance Gap for Big Data Workloads: A New NVDIMM-based Approach.** (The Hong Kong Polytechnic University, NSF/University of Florida)
- **A Patch Memory System For Image Processing and Computer Vision.** (NVDIA)
- A Cloud-Scale Acceleration Architecture. (Microsoft Research)
- Reducing Data Movement Energy via Online Data Clustering and Encoding. (University of Rochester)
- The Microarchitecture of a Real-time Robot Motion Planning Accelerator. (Duke University)
- An Ultra Low-Power Hardware Accelerator for Automatic Speech Recognition. (Universitat Politecnica de Catalunya)
- Chameleon: Versatile and Practical Near-DRAM Acceleration Architecture for Large Memory Systems. (UIUC, Seoul National University)

### 2016 FPL
- **A High Performance FPGA-based Accelerator for Large-Scale Convolutional Neural Network.** (Fudan University)
- **Overcoming Resource Underutilization in Spatial CNN Accelerators.** (Stony Brook University)
  - *Build multiple accelerators, each specialized for specific CNN layers, instead of a single accelerator with uniform tiling parameters.* 
- **Accelerating Recurrent Neural Networks in Analytics Servers: Comparison of FPGA, CPU, GPU, and ASIC.** (Intel)

## Important Topics
This is a collection of papers on other important topics related to neural networks. Papers of significance are marked in **bold**. My comments are in marked in *italic*.

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
- Learning both Weights and Connections for Efficient Neural Network. (Stanford University, NVDIA, **2015 NIPS**)
- [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding.](http://arxiv.org/abs/1510.00149) (Stanford University, Tsinghua University, **2016 ICLR**)
- [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size.](http://arxiv.org/abs/1602.07360) (DeepScale & UC Berkeley, Stanford University)
- [8-Bit Approximations for Parallelism in Deep Learning.](http://arxiv.org/abs/1511.04561) (Universia della Svizzera italiana, **2016 ICLR**)
- [Neural Networks with Few Multiplications.](https://arxiv.org/abs/1510.03009) (Universite de Montreal, **2016 ICLR**)
- [Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications.](http://arxiv.org/abs/1511.06530) (Samsung, Seoul National University, **2016 ICLR**)
- [Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1.](http://arxiv.org/abs/1602.02830)
  - *Constrain both the weights and the activations to either +1 or -1.*
- [Quantized Neural Networks: Training Neural Networks with Low Precision Weights and Activations.](http://arxiv.org/abs/1609.07061) (Universite de Montreal)
- [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks.](http://arxiv.org/abs/1603.05279) (Allen Institute for AI, University of Washington)
- [DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients.](http://arxiv.org/abs/1606.06160) (Megvii)
- [Deep Learning with Limited Numerical Precision.](https://arxiv.org/abs/1502.02551) (IBM)
- [Dynamic Network Surgery for Efficient DNNs.](http://arxiv.org/abs/1608.04493) (Intel Labs China)

### Other Topics
#### Object Detection
- You Only Look Once: Unified, Real-Time Object Detection. (University of Washington, Allen Institute for AI, Facebook AI Research, **2016 CVPR**)

#### GAN
- [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) (Universite de Montreal, **2014 NIPS**)
 - *Two "adversarial" MLP models G and D: a generative model G that captures the data distribution and a discriminative model D that estimates the probability that a sample came from the training data rather than G*.
 - *D is trained to learn the above probability*.
 - *G is trained to maximize the probability of D making a mistake.*.

##### Neural Turing Machine
#### Attention and Memory
#### Reinforcement Learning
#### Sequentialization

## Research Groups

## Industry Contributions
 - [Movidius](http://www.movidius.com/)
   - Myriad 2: Hardware-accelerated visual intelligence at ultra-low power.
   - Fathom Neural Compute Stick: The world's first discrete deep learning accelerator (Myriad 2 VPU inside).
 - [NVIDIA](http://www.nvidia.com/)
   - Jetson TX1: Embedded visual computing developing platform.
   - DGX-1: Deep learning supercomputer.
 - Google
   - TPU (Tensor Processing Unit).
 - [Nervana](https://www.nervanasys.com/)
   - Nervana Engine: Hardware optimized for deep learning.
 - [Wave Computing](http://wavecomp.com/)
   - Deep Learning Computers Powered by Dataflow Technology.
