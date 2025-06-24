# Hardware and Software Prerequisites

Before you begin using the BioNeMo framework, please ensure the hardware and software prerequisites outlined below are
met.

## Hardware

The BioNeMo Framework is compatible with environments that have access to NVIDIA GPUs. `bfloat16` precision requires an
Ampere generation GPU or higher ([Compute Capability ≥8.0](https://developer.nvidia.com/cuda-gpus)). You may be able
to run BioNeMo on GPUs without `bfloat16`, but this use-case is not supported by the development team.

### GPU Support Matrix

The following datacenter and desktop GPUs have Compute Capability ≥8.0 and are supported hardware for BioNeMo:

| GPU | Compute Capability | Support |
|-----|--------------------|---------|
| H100 | 9.0 | Full |
| L4 | 8.9 | Full |
| L40 | 8.9 | Full |
| A100 | 8.0 | Full |
| A40 | 8.6 | Full |
| A30 | 8.0 | Full |
| A10 | 8.6 | Full |
| A16 | 8.6 | Full |
| A2 | 8.6 | Full |
| RTX 6000 | 8.9 | Full |
| RTX A6000 | 8.6 | Full |
| RTX A5000 | 8.6 | Full |
| RTX A4000 | 8.6 | Full |

## Software

The BioNeMo Framework is supported on x86 Linux systems.

Please ensure that the following are installed in your desired execution environment:

* Appropriate GPU drivers (minimum version: 560; lower versions may be compatible, but cannot be guaranteed)
* Docker (with GPU support, Docker Engine 19.03 or above)
* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
to allow Docker to access the GPUs
