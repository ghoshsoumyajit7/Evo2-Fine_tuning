# Frequently Asked Questions

### Is BioNeMo Framework free to use?

Yes, BioNeMo Framework is free to use. BioNeMo Framework code is licensed under the Apache 2.0 License. The Apache 2.0
License is a permissive open-source license that allows users to freely use, modify, and distribute software. With this
license, users have the right to use the software for any purpose, including commercial use, without requiring royalties
or attribution. Overall, our choice of the Apache 2.0 License allows for wide adoption and use of BioNeMo Framework,
while also providing a high degree of freedom and flexibility for users.

For users that would like NVIDIA AI Enterprise support for
[BioNeMo Framework](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/containers/bionemo-framework) container
usage, refer to the
[NVAIE Landing Page](https://www.nvidia.com/en-us/data-center/products/ai-enterprise/)
for more information.

### How do I install BioNeMo Framework?

BioNeMo Framework is distributed as a Docker container through NVIDIA NGC. To download the pre-built Docker container
and data assets, you will need a free NVIDIA NGC account.

Alternatively, you can install individual sub-packages from within BioNeMo Framework by following the corresponding
README pages the [BioNeMo Framework GitHub](https://github.com/NVIDIA/bionemo-framework). Please note that this is a
beta feature and may require some additional effort to install seamlessly. We are actively working on testing this
functionality and expect it will be a fully supported feature in future releases. You can review our
[release notes](https://docs.nvidia.com/bionemo-framework/latest/user-guide/appendix/releasenotes-fw/) to stay up to
date on our releases.

### How do I update BioNeMo Framework to the latest version?

To update the BioNeMo Framework Docker container, you need to pull the latest version of the Docker image using the
command `docker pull`. For available tags, refer to the
[BioNeMo Framework page in the NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/containers/bionemo-framework).

### What are the system requirements for BioNeMo Framework?

Generally, BioNeMo Framework should run on any NVIDIA GPU with Compute Capability ≥8.0. For a full list of supported
hardware, refer to the [Hardware and Software Prerequisites](../getting-started/pre-reqs.md).

### Can I contribute code or models to BioNeMo Framework?

Yes, BioNeMo Framework is open source and we welcome contributions from organizations and individuals.
You can do so either by forking the repository and directly opening a PR against our `main` branch from your fork or by
[contacting us](https://www.nvidia.com/en-us/industries/healthcare/contact-sales/) fo r further assistance. BioNeMo
Framework's mission is to stay extremely light weight and primarily support building blocks required for various AI
models. As such, we currently prioritize feature extensions, bug fixes, and new independent modules such as dataloaders,
tokenizers, custom architecture blocks, and other reusable features over end-to-end model implementations. We might
consider end-to-end model implementations on a case-by-case basis. If you're interested in this contribution of this
kind, we recommend [reaching out to us](https://www.nvidia.com/en-us/industries/healthcare/contact-sales/) first

For more information about external contributions, refer to the [Contributing](../contributing/contributing.md) and
[Code Review](../contributing/code-review.md) pages.

### How do I report bugs or suggest new features?

To report a bug or suggest a new feature, open an issue on the
[BioNeMo Framework GitHub site](https://github.com/NVIDIA/bionemo-framework/issues). For the fastest turnaround,
thoroughly describe your issue, including any steps and/or _minimal_ data sets necessary to reproduce (when possible),
as well as the expected behavior.

### Can I train models in Jupyter notebooks using BioNeMo Framework?

At the current time, notebook-based training is not supported due to restrictions imposed by the Megatron framework that
underpins the BioNeMo Framework models. However, the user may call training scripts using a subprocess, either through
the use of the [Python Subprocess module](https://docs.python.org/3/library/subprocess.html) or through
Jupyter's [Shell Assignment](https://ipython.readthedocs.io/en/stable/interactive/python-ipython-diff.html#shell-assignment)
or [Bash Cell Magic](https://ipython.readthedocs.io/en/stable/interactive/magics.html#cellmagic-bash). For the latter
two options, we caution the user to be careful when using Python and shell variables as we have observed unpredictable
and unreproducible behavior in certain instances.
