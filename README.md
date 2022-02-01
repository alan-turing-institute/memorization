# On Memorization in Probabilistic Deep Generative Models (NeurIPS 2021)

This repository contains the code necessary to reproduce the experiments in 
[On Memorization in Probabilistic Deep Generative Models][neurips-link]. You 
can also use this code to measure memorization in other types of probabilistic 
deep generative models. If you use our code in your own work please cite the 
paper using, for instance, the following BibTeX entry:

```bibtex
@inproceedings{van2021memorization,
  title={On Memorization in Probabilistic Deep Generative Models},
  author={{Van den Burg}, G. J. J. and Williams, C. K. I.},
  booktitle={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

If you have any questions or encounter an issue when using this code, please 
send an email to ``gertjanvandenburg at gmail dot com``.

## Introduction

The files in the ``scripts`` directory are needed to reproduce the experiments 
and generate the figures in the paper. The experiments are organized using the 
``Makefile`` provided. To reproduce the experiments or recreate the figures 
from the analysis, you'll have to install a number of dependencies. We use 
[PyTorch](https://pytorch.org) to implement the deep learning algorithms. If 
you don't wish to re-run all the models, you can download the result files 
used in the paper ([see below](#result-files)).

The scripts are all written in Python, and the necessary external dependencies 
can be found in the ``requirements.txt`` file. These can be installed using:

```
$ pip install -r requirements.txt
```

To recreate the figures the following system dependencies are also needed: 
``pdflatex``, ``latexmk``, ``lualatex``, and ``make``. These programs are 
available for all major platforms.

## Reproducing the results

To train the models on the different data sets, you can run:

```
$ make memorization
```

Note that depending on your machine this may take some time, so it might be 
easier to simply download the result files instead. It is also worth 
mentioning that while we have made an effort to ensure reproducibility by 
setting the random seed in PyTorch, platform or package version differences 
may result in slightly different output files (see also [PyTorch 
Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)).

All figures in the paper are generated from the raw result files using Python 
scripts. First, the ``summarize.py`` script takes the raw result files and 
creates summary files for each data set. Next, the analysis scripts are used 
to generate the figures, most of which are LaTeX files that require 
compilation using PDFLaTeX or LuaLaTeX. Simply run:

```
$ make analysis
```

to create the summaries and the output files. When using the result files 
linked below this will give the exact same figures as shown in the paper.

## Result files

Due to their size, the raw result files are not contained in this repository, 
but can be downloaded separately from [this link][result-link] (about 2.6GB). 
After downloading the ``results.zip`` file, unpack it and move the ``results`` 
directory to where you've cloned this repository (so adjacent to the 
``scripts`` directory). Below is a concise overview of the necessary commands:

```bash
$ git clone https://github.com/alan-turing-institute/memorization
$ cd memorization
$ wget https://gertjanvandenburg.com/projects/memorization/results.zip # or download the file in some other way
$ unzip results.zip
$ touch results/*/*/*          # update modification time of the result files
$ make analysis                # optionally, run ``make -n analysis`` first to see what will happen
```

After unpacking the zip file, you can optionally verify the integrity of the 
results using the SHA-256 checksums provided:

```bash
$ sha256sum --check results.sha256
```

## License

The code in this repository is licensed under the MIT license. See the 
[LICENSE file](LICENSE) for further details. Reuse of the code in this 
repository is allowed, but should cite [our paper][neurips-link].

## Notes

If you find any problems or have a suggestion for improvement of this 
repository, please [let me know](mailto:gertjanvandenburg@gmail.com) as it 
will help make this resource better for everyone. 

[neurips-link]: https://papers.nips.cc/paper/2021/hash/eae15aabaa768ae4a5993a8a4f4fa6e4-Abstract.html
[result-link]: https://gertjanvandenburg.com/projects/memorization/results.zip
