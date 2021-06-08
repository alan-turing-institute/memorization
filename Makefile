# Makefile for Memorization Experiments
#
# Author: G.J.J. van den Burg
# Copyright (c) 2021, The Alan Turing Institute
# License: See the LICENSE file.
#

SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS += --no-builtin-rules

SCRIPT_DIR=./scripts
SUMMARY_DIR=./summaries
OUTPUT_DIR=./output
RESULT_DIR=./results
PAPER_IMG_DIR=../../../paper/images/

RESULT_DIR_BMNIST_LR3=$(RESULT_DIR)/binarized_mnist_extra_lr3/results
RESULT_DIR_BMNIST_LR4=$(RESULT_DIR)/binarized_mnist_extra_lr4/results
RESULT_DIR_CIFAR10=$(RESULT_DIR)/cifar10/results
RESULT_DIR_CELEBA=$(RESULT_DIR)/celeba/results

CKPT_DIR_BMNIST_LR3=$(RESULT_DIR)/binarized_mnist_extra_lr3/checkpoints
CKPT_DIR_BMNIST_LR4=$(RESULT_DIR)/binarized_mnist_extra_lr4/checkpoints
CKPT_DIR_CIFAR10=$(RESULT_DIR)/cifar10/checkpoints
CKPT_DIR_CELEBA=$(RESULT_DIR)/celeba/checkpoints

# dependencies of memorization.py
MEM_FILES=\
	  $(SCRIPT_DIR)/constants.py \
	  $(SCRIPT_DIR)/dataset.py \
	  $(SCRIPT_DIR)/models.py \
	  $(SCRIPT_DIR)/trainer.py \
	  $(SCRIPT_DIR)/seed_generator.py

##########
#        #
# GLOBAL #
#        #
##########

.PHONY: all

all: memorization summaries analysis

############################
#                          #
# MEMORIZATION EXPERIMENTS #
#                          #
############################

.PHONY: memorization

memorization: \
	mem_mnist_lr3 \
	mem_mnist_lr4 \
	mem_cifar10 \
	mem_celeba

############################
# BinarizedMNIST lr = 1e-3 #
############################

BMNIST_LR3_CV_TARGETS=
BMNIST_LR3_FULL_TARGETS=

define make_targets_cv_bmnist_lr3
BMNIST_LR3_CV_TARGETS += $(RESULT_DIR_BMNIST_LR3)/BinarizedMNIST_BernoulliMLPVAE_512-256-16_seed$(1)_cv_repeat$(2)_fold$(3).json.gz
endef

define make_targets_full_bmnist_lr3
BMNIST_LR3_FULL_TARGETS += $(RESULT_DIR_BMNIST_LR3)/BinarizedMNIST_BernoulliMLPVAE_512-256-16_seed$(1)_full_repeat$(2).json.gz
endef

$(foreach repeat,$(shell seq 0 9),\
	$(foreach fold,$(shell seq 0 9),\
	$(eval $(call make_targets_cv_bmnist_lr3,42,$(repeat),$(fold)))\
))

$(eval $(call make_targets_full_bmnist_lr3,42,0))

.PHONY: mem_mnist_lr3 mem_mnist_cv_lr3 mem_mnist_full_lr3

mem_mnist_lr3: mem_mnist_cv_lr3 mem_mnist_full_lr3

mem_mnist_cv_lr3: $(BMNIST_LR3_CV_TARGETS)

mem_mnist_full_lr3: $(BMNIST_LR3_FULL_TARGETS)

.PRECIOUS: $(BMNIST_LR3_FULL_TARGETS) $(BMNIST_LR3_CV_TARGETS)

$(BMNIST_LR3_CV_TARGETS) &: $(SCRIPT_DIR)/memorization.py $(MEM_FILES)
	python $< \
		--dataset BinarizedMNIST \
		--model BernoulliMLPVAE \
		--mode split-cv \
		--latent-dim 16 \
		--batch-size 64 \
		--learning-rate 1e-3 \
		--epochs 100 \
		--repeats 10 \
		--seed 42 \
		--compute-px-every 5 \
		--result-dir $(RESULT_DIR_BMNIST_LR3) \
		--checkpoint-every 100 \
		--checkpoint-dir $(CKPT_DIR_BMNIST_LR3)

$(BMNIST_LR3_FULL_TARGETS) &: $(SCRIPT_DIR)/memorization.py $(MEM_FILES)
	python $< \
		--dataset BinarizedMNIST \
		--model BernoulliMLPVAE \
		--mode full \
		--latent-dim 16 \
		--batch-size 64 \
		--learning-rate 1e-3 \
		--epochs 100 \
		--repeats 1 \
		--seed 42 \
		--compute-px-every 5 \
		--result-dir $(RESULT_DIR_BMNIST_LR3) \
		--checkpoint-every 100 \
		--checkpoint-dir $(CKPT_DIR_BMNIST_LR3)

############################
# BinarizedMNIST lr = 1e-4 #
############################

BMNIST_LR4_CV_TARGETS=
BMNIST_LR4_FULL_TARGETS=

define make_targets_cv_bmnist_lr4
BMNIST_LR4_CV_TARGETS += $(RESULT_DIR_BMNIST_LR4)/BinarizedMNIST_BernoulliMLPVAE_512-256-16_seed$(1)_cv_repeat$(2)_fold$(3).json.gz
endef

define make_targets_full_bmnist_lr4
BMNIST_LR4_FULL_TARGETS += $(RESULT_DIR_BMNIST_LR4)/BinarizedMNIST_BernoulliMLPVAE_512-256-16_seed$(1)_full_repeat$(2).json.gz
endef

$(foreach repeat,$(shell seq 0 9),\
	$(foreach fold,$(shell seq 0 9),\
	$(eval $(call make_targets_cv_bmnist_lr4,42,$(repeat),$(fold)))\
))

$(eval $(call make_targets_full_bmnist_lr4,42,0))

.PHONY: mem_mnist_lr4 mem_mnist_cv_lr4 mem_mnist_full_lr4

mem_mnist_lr4: mem_mnist_cv_lr4 mem_mnist_full_lr4

mem_mnist_cv_lr4: $(BMNIST_LR4_CV_TARGETS)

mem_mnist_full_lr4: $(BMNIST_LR4_FULL_TARGETS)

.PRECIOUS: $(BMNIST_LR4_FULL_TARGETS) $(BMNIST_LR4_CV_TARGETS)

$(BMNIST_LR4_CV_TARGETS) &: $(SCRIPT_DIR)/memorization.py $(MEM_FILES)
	python $< \
		--dataset BinarizedMNIST \
		--model BernoulliMLPVAE \
		--mode split-cv \
		--latent-dim 16 \
		--batch-size 64 \
		--learning-rate 1e-4 \
		--epochs 100 \
		--repeats 10 \
		--seed 42 \
		--compute-px-every 5 \
		--result-dir $(RESULT_DIR_BMNIST_LR4) \
		--checkpoint-every 100 \
		--checkpoint-dir $(CKPT_DIR_BMNIST_LR4)

$(BMNIST_LR4_FULL_TARGETS) &: $(SCRIPT_DIR)/memorization.py $(MEM_FILES)
	python $< \
		--dataset BinarizedMNIST \
		--model BernoulliMLPVAE \
		--mode full \
		--latent-dim 16 \
		--batch-size 64 \
		--learning-rate 1e-4 \
		--epochs 100 \
		--repeats 1 \
		--seed 42 \
		--compute-px-every 5 \
		--result-dir $(RESULT_DIR_BMNIST_LR4) \
		--checkpoint-every 100 \
		--checkpoint-dir $(CKPT_DIR_BMNIST_LR4)

############
# CIFAR-10 #
############

CIFAR10_CV_TARGETS=
CIFAR10_FULL_TARGETS=

define make_targets_cv_cifar10
CIFAR10_CV_TARGETS += $(RESULT_DIR_CIFAR10)/CIFAR10_DiagonalGaussianDCVAE_NF32-L64_seed$(1)_cv_repeat$(2)_fold$(3).json.gz
endef

define make_targets_full_cifar10
CIFAR10_FULL_TARGETS += $(RESULT_DIR_CIFAR10)/CIFAR10_DiagonalGaussianDCVAE_NF32-L64_seed$(1)_full_repeat$(2).json.gz
endef

$(foreach repeat,$(shell seq 0 9),\
	$(foreach fold,$(shell seq 0 9),\
	$(eval $(call make_targets_cv_cifar10,42,$(repeat),$(fold)))\
))

$(eval $(call make_targets_full_cifar10,42,0))

.PHONY: mem_cifar10 mem_cifar10_cv mem_cifar10_full

mem_cifar10: mem_cifar10_cv mem_cifar10_full

mem_cifar10_cv: $(CIFAR10_CV_TARGETS)

mem_cifar10_full: $(CIFAR10_FULL_TARGETS)

.PRECIOUS: $(CIFAR10_FULL_TARGETS) $(CIFAR10_CV_TARGETS)

$(CIFAR10_CV_TARGETS) &: $(SCRIPT_DIR)/memorization.py $(MEM_FILES)
	python $< \
		--dataset CIFAR10 \
		--model DiagonalGaussianDCVAE \
		--mode split-cv \
		--latent-dim 64 \
		--batch-size 64 \
		--learning-rate 1e-3 \
		--epochs 100 \
		--repeats 10 \
		--seed 42 \
		--compute-px-every 50 \
		--result-dir $(RESULT_DIR_CIFAR10) \
		--checkpoint-every 50 \
		--checkpoint-dir $(CKPT_DIR_CIFAR10)

$(CIFAR10_FULL_TARGETS) &: $(SCRIPT_DIR)/memorization.py $(MEM_FILES)
	python $< \
		--dataset CIFAR10 \
		--model DiagonalGaussianDCVAE \
		--mode full \
		--latent-dim 64 \
		--batch-size 64 \
		--learning-rate 1e-3 \
		--epochs 100 \
		--repeats 1 \
		--seed 42 \
		--compute-px-every 50 \
		--result-dir $(RESULT_DIR_CIFAR10) \
		--checkpoint-every 50 \
		--checkpoint-dir $(CKPT_DIR_CIFAR10)
##########
# CelebA #
##########

CELEBA_CV_TARGETS=
CELEBA_FULL_TARGETS=

define make_targets_cv_celeba
CELEBA_CV_TARGETS += $(RESULT_DIR_CELEBA)/CelebA_ConstantGaussianDCVAE_NF32-L32_seed$(1)_cv_repeat$(2)_fold$(3).json.gz
endef

define make_targets_full_celeba
CELEBA_FULL_TARGETS += $(RESULT_DIR_CELEBA)/CelebA_ConstantGaussianDCVAE_NF32-L32_seed$(1)_full_repeat$(2).json.gz
endef

$(foreach repeat,$(shell seq 0 9),\
	$(foreach fold,$(shell seq 0 9),\
	$(eval $(call make_targets_cv_celeba,42,$(repeat),$(fold)))\
))

$(eval $(call make_targets_full_celeba,42,0))

.PHONY: mem_celeba mem_celeba_cv mem_celeba_full

mem_celeba: mem_celeba_cv mem_celeba_full

mem_celeba_cv: $(CELEBA_CV_TARGETS)

mem_celeba_full: $(CELEBA_FULL_TARGETS)

.PRECIOUS: $(CELEBA_FULL_TARGETS) $(CELEBA_CV_TARGETS)

$(CELEBA_CV_TARGETS) &: $(SCRIPT_DIR)/memorization.py $(MEM_FILES)
	python $< \
		--dataset CelebA \
		--model ConstantGaussianDCVAE \
		--mode split-cv \
		--latent-dim 32 \
		--batch-size 64 \
		--learning-rate 1e-3 \
		--epochs 50 \
		--repeats 10 \
		--seed 42 \
		--compute-px-every 25 \
		--result-dir $(RESULT_DIR_CELEBA) \
		--checkpoint-every 25 \
		--checkpoint-dir $(CKPT_DIR_CELEBA)

$(CELEBA_FULL_TARGETS) &: $(SCRIPT_DIR)/memorization.py $(MEM_FILES)
	python $< \
		--dataset CelebA \
		--model ConstantGaussianDCVAE \
		--mode full \
		--latent-dim 32 \
		--batch-size 64 \
		--learning-rate 1e-3 \
		--epochs 50 \
		--repeats 1 \
		--seed 42 \
		--compute-px-every 25 \
		--result-dir $(RESULT_DIR_CELEBA) \
		--checkpoint-every 25 \
		--checkpoint-dir $(CKPT_DIR_CELEBA)


#################
#               #
# SUMMARIZATION #
#               #
#################

.PHONY: summaries summary-dir

summaries: \
	$(SUMMARY_DIR)/mem_bmnist_lr3.npz \
	$(SUMMARY_DIR)/mem_bmnist_lr4.npz \
	$(SUMMARY_DIR)/mem_cifar10.npz \
	$(SUMMARY_DIR)/mem_celeba.npz

summary-dir:
	mkdir -p $(SUMMARY_DIR)

$(SUMMARY_DIR)/mem_bmnist_lr3.npz: $(SCRIPT_DIR)/summarize.py \
	$(BMNIST_LR3_CV_TARGETS) | summary-dir
	python $< -o $@ --result-files $(BMNIST_LR3_CV_TARGETS)

$(SUMMARY_DIR)/mem_bmnist_lr4.npz: $(SCRIPT_DIR)/summarize.py \
	$(BMNIST_LR4_CV_TARGETS) | summary-dir
	python $< -o $@ --result-files $(BMNIST_LR4_CV_TARGETS)

$(SUMMARY_DIR)/mem_cifar10.npz: $(SCRIPT_DIR)/summarize.py \
	$(CIFAR10_CV_TARGETS) | summary-dir
	python $< -o $@ --result-files $(CIFAR10_CV_TARGETS)

$(SUMMARY_DIR)/mem_celeba.npz: $(SCRIPT_DIR)/summarize.py \
	$(CELEBA_CV_TARGETS) | summary-dir
	python $< -o $@ --result-files $(CELEBA_CV_TARGETS)


############
#          #
# ANALYSIS #
#          #
############

.PHONY: analysis

analysis: \
	qualitative \
	losses \
	quantiles \
	mnist_densities \
	mem_logprob \
	nearest_neighbors

#############################
# Qualitative illustrations #
#############################

.PHONY: qualitative \
	qualitative_celeba \
	qualitative_cifar10 \
	qualitative_bmnist_lr3 \
	qualitative_bmnist_lr4 \
	qualitative-dir

qualitative: \
	qualitative_celeba \
	qualitative_cifar10 \
	qualitative_bmnist_lr3 \
	qualitative_bmnist_lr4

qualitative-dir:
	mkdir -p $(OUTPUT_DIR)/qualitative

QUALITATIVE_TARGETS=

define make_qualitative
QUALITATIVE_TARGETS_$(1)=\
	$(OUTPUT_DIR)/qualitative/$(1)_top.png \
	$(OUTPUT_DIR)/qualitative/$(1)_middle.png \
	$(OUTPUT_DIR)/qualitative/$(1)_bottom.png

QUALITATIVE_TARGETS += $$(QUALITATIVE_TARGETS_$(1))

qualitative_$(1): $$(QUALITATIVE_TARGETS_$(1))
endef

$(foreach dset,celeba cifar10 bmnist_lr3 bmnist_lr4,\
	$(eval $(call make_qualitative,$(dset)))\
)

$(OUTPUT_DIR)/qualitative/%_top.png: \
	$(SUMMARY_DIR)/mem_%.npz \
	$(SCRIPT_DIR)/analysis_figure_minmaxmedian.py | qualitative-dir
	python $(SCRIPT_DIR)/analysis_figure_minmaxmedian.py -i $< -o $@ \
		--mode top

$(OUTPUT_DIR)/qualitative/%_middle.png: \
	$(SUMMARY_DIR)/mem_%.npz \
	$(SCRIPT_DIR)/analysis_figure_minmaxmedian.py | qualitative-dir
	python $(SCRIPT_DIR)/analysis_figure_minmaxmedian.py -i $< -o $@ \
		--mode middle

$(OUTPUT_DIR)/qualitative/%_bottom.png: \
	$(SUMMARY_DIR)/mem_%.npz \
	$(SCRIPT_DIR)/analysis_figure_minmaxmedian.py | qualitative-dir
	python $(SCRIPT_DIR)/analysis_figure_minmaxmedian.py -i $< -o $@ \
		--mode bottom

###############
# Loss Curves #
###############

.PHONY: losses losses-dir

losses: $(OUTPUT_DIR)/losses/MNIST_losses.pdf

losses-dir:
	mkdir -p $(OUTPUT_DIR)/losses

$(OUTPUT_DIR)/losses/MNIST_losses.tex: \
	$(SCRIPT_DIR)/analysis_figure_losses.py \
	$(RESULT_DIR_BMNIST_LR3)/BinarizedMNIST_BernoulliMLPVAE_512-256-16_seed42_full_repeat0.json.gz \
	$(RESULT_DIR_BMNIST_LR4)/BinarizedMNIST_BernoulliMLPVAE_512-256-16_seed42_full_repeat0.json.gz\
	| losses-dir
	python $< -o $@ \
		--lr3-file $(RESULT_DIR_BMNIST_LR3)/BinarizedMNIST_BernoulliMLPVAE_512-256-16_seed42_full_repeat0.json.gz \
		--lr4-file $(RESULT_DIR_BMNIST_LR4)/BinarizedMNIST_BernoulliMLPVAE_512-256-16_seed42_full_repeat0.json.gz

$(OUTPUT_DIR)/losses/%.pdf: $(OUTPUT_DIR)/losses/%.tex
	cd $(OUTPUT_DIR)/losses && \
		latexmk -pdf \
		-pdflatex="pdflatex -interaction=nonstopmode --shell-escape" \
		MNIST_losses.tex

#############
# Quantiles #
#############

.PHONY: quantiles quantile-dir

quantiles: $(OUTPUT_DIR)/quantiles/MNIST_quantiles.pdf

quantile-dir:
	mkdir -p $(OUTPUT_DIR)/quantiles

$(OUTPUT_DIR)/quantiles/MNIST_quantiles.tex: \
	$(SCRIPT_DIR)/analysis_figure_mem_quantiles.py \
	$(SUMMARY_DIR)/mem_bmnist_lr3.npz \
	$(SUMMARY_DIR)/mem_bmnist_lr4.npz | quantile-dir
	python $(SCRIPT_DIR)/analysis_figure_mem_quantiles.py -o $@ \
		--lr3-file $(SUMMARY_DIR)/mem_bmnist_lr3.npz \
		--lr4-file $(SUMMARY_DIR)/mem_bmnist_lr4.npz

$(OUTPUT_DIR)/quantiles/%.pdf: $(OUTPUT_DIR)/quantiles/%.tex
	cd $(OUTPUT_DIR)/quantiles && \
		latexmk -pdf \
		-pdflatex="pdflatex -interaction=nonstopmode --shell-escape" \
		MNIST_quantiles.tex

###################
# MNIST Densities #
###################

.PHONY: mnist_densities mnist_density-dir

mnist_densities: $(OUTPUT_DIR)/mnist_densities/MNIST_densities.pdf

mnist_density-dir:
	mkdir -p $(OUTPUT_DIR)/mnist_densities

$(OUTPUT_DIR)/mnist_densities/MNIST_densities.tex: \
	$(SCRIPT_DIR)/analysis_figure_mem_histograms.py \
	$(SUMMARY_DIR)/mem_bmnist_lr3.npz \
	$(SUMMARY_DIR)/mem_bmnist_lr4.npz | mnist_density-dir
	python $(SCRIPT_DIR)/analysis_figure_mem_histograms.py -o $@ \
		--lr3-file $(SUMMARY_DIR)/mem_bmnist_lr3.npz \
		--lr4-file $(SUMMARY_DIR)/mem_bmnist_lr4.npz

$(OUTPUT_DIR)/mnist_densities/%.pdf: $(OUTPUT_DIR)/mnist_densities/%.tex
	cd $(OUTPUT_DIR)/mnist_densities && lualatex MNIST_densities.tex

#################################
# Memorization in log prob bins #
#################################

.PHONY: mem_logprob \
	mem_logprob_celeba \
	mem_logprob_bmnist_lr3 \
	mem_logprob_bmnist_lr4

mem_logprob: \
	mem_logprob_celeba \
	mem_logprob_bmnist_lr3 \
	mem_logprob_bmnist_lr4

mem_logprob-dir:
	mkdir -p $(OUTPUT_DIR)/mem_logprob/

MEM_LOGPROB_TARGETS=

define make_mem_logprob
MEM_LOGPROB_TARGETS_$(1)=\
	$(OUTPUT_DIR)/mem_logprob/mem_logpx_binned_$(1).pdf \
	$(OUTPUT_DIR)/mem_logprob/mem_logpx_prop_$(1).pdf

MEM_LOGPROB_TARGETS += $$(MEM_LOGPROB_TARGETS_$(1))

mem_logprob_$(1): $$(MEM_LOGPROB_TARGETS_$(1))
endef

$(foreach dset,celeba bmnist_lr3 bmnist_lr4,\
	$(eval $(call make_mem_logprob,$(dset)))\
)

$(OUTPUT_DIR)/mem_logprob/mem_logpx_binned_%.tex: \
	$(SUMMARY_DIR)/mem_%.npz \
	$(SCRIPT_DIR)/analysis_figure_histogram_logpx_bins.py | mem_logprob-dir
	python $(SCRIPT_DIR)/analysis_figure_histogram_logpx_bins.py \
		-i $< --seed 555 -o $@

$(OUTPUT_DIR)/mem_logprob/mem_logpx_prop_%.tex: \
	$(SUMMARY_DIR)/mem_%.npz \
	$(SCRIPT_DIR)/analysis_figure_logpx_props.py | mem_logprob-dir
	python $(SCRIPT_DIR)/analysis_figure_logpx_props.py -i $< -o $@

$(OUTPUT_DIR)/mem_logprob/%.pdf: $(OUTPUT_DIR)/mem_logprob/%.tex
	cd $(OUTPUT_DIR)/mem_logprob && \
		latexmk -pdf \
		-pdflatex="pdflatex -interaction=nonstopmode --shell-escape" \
		$(notdir $<)

#####################
# Nearest Neighbors #
#####################

.PHONY: nearest_neighbors \
	nearest_neighbors_celeba \
	nearest_neighbors_cifar10 \
	nearest_neighbors_bmnist_lr3 \
	nearest_neighbors_bmnist_lr4

nearest_neighbors: \
	nearest_neighbors_celeba \
	nearest_neighbors_cifar10 \
	nearest_neighbors_bmnist_lr3 \
	nearest_neighbors_bmnist_lr4

nearest_neighbors-dir:
	mkdir -p $(OUTPUT_DIR)/nearest_neighbors/

NEAREST_NEIGHBOR_TARGETS=

define make_nearest_neighbors
NEAREST_NEIGHBOR_TARGETS_$(1)=\
	$(OUTPUT_DIR)/nearest_neighbors/nn_histogram_$(1).pdf \
	$(OUTPUT_DIR)/nearest_neighbors/nn_scatter_$(1).pdf

NEAREST_NEIGHBOR_TARGETS += $$(NEAREST_NEIGHBOR_TARGETS_$(1))

nearest_neighbors_$(1): $$(NEAREST_NEIGHBOR_TARGETS_$(1))
endef

$(foreach dset,celeba bmnist_lr3 bmnist_lr4,\
	$(eval $(call make_nearest_neighbors,$(dset)))\
)

$(OUTPUT_DIR)/nearest_neighbors/nn_histogram_%.tex: \
	$(OUTPUT_DIR)/nearest_neighbors/nns_%.npz \
	$(SUMMARY_DIR)/mem_%.npz \
	$(SCRIPT_DIR)/analysis_figure_nn_histograms.py | nearest_neighbors-dir
	python $(SCRIPT_DIR)/analysis_figure_nn_histograms.py \
		--nns $< --results $(SUMMARY_DIR)/mem_$*.npz -o $@

$(OUTPUT_DIR)/nearest_neighbors/nn_scatter_%.tex: \
	$(OUTPUT_DIR)/nearest_neighbors/nns_%.npz \
	$(SUMMARY_DIR)/mem_%.npz \
	$(SCRIPT_DIR)/analysis_figure_nn_scatter.py | nearest_neighbors-dir
	python $(SCRIPT_DIR)/analysis_figure_nn_scatter.py \
		--nns $< --results $(SUMMARY_DIR)/mem_$*.npz -o $@

$(OUTPUT_DIR)/nearest_neighbors/nn_scatter_%.pdf: \
	$(OUTPUT_DIR)/nearest_neighbors/nn_scatter_%.tex
	cd $(OUTPUT_DIR)/nearest_neighbors && \
		latexmk -pdf \
		-pdflatex="pdflatex -interaction=nonstopmode --shell-escape" \
		$(notdir $<)

$(OUTPUT_DIR)/nearest_neighbors/nn_histogram_%.pdf: \
	$(OUTPUT_DIR)/nearest_neighbors/nn_histogram_%.tex
	cd $(OUTPUT_DIR)/nearest_neighbors && lualatex $(notdir $<)

$(OUTPUT_DIR)/nearest_neighbors/nns_celeba.npz: \
	$(CKPT_DIR_CELEBA)/CelebA_ConstantGaussianDCVAE_NF32-L32_seed42_full_repeat0_epoch50.params \
	$(SCRIPT_DIR)/analysis_nearest_neighbor_distance.py \
	       	| nearest_neighbors-dir
	python $(SCRIPT_DIR)/analysis_nearest_neighbor_distance.py \
		--model ConstantGaussianDCVAE \
		--dataset CelebA \
		--latent-dim 32 \
		--seed 123 \
		--checkpoint $< \
		--output $@

$(OUTPUT_DIR)/nearest_neighbors/nns_bmnist_lr3.npz: \
	$(CKPT_DIR_BMNIST_LR3)/BinarizedMNIST_BernoulliMLPVAE_512-256-16_seed42_full_repeat0_epoch100.params \
	$(SCRIPT_DIR)/analysis_nearest_neighbor_distance.py \
		| nearest_neighbors-dir
	python $(SCRIPT_DIR)/analysis_nearest_neighbor_distance.py \
		--model BernoulliMLPVAE \
		--dataset BinarizedMNIST \
		--latent-dim 16 \
		--seed 42 \
		--checkpoint $< \
		--output $@

$(OUTPUT_DIR)/nearest_neighbors/nns_bmnist_lr4.npz: \
	$(CKPT_DIR_BMNIST_LR4)/BinarizedMNIST_BernoulliMLPVAE_512-256-16_seed42_full_repeat0_epoch100.params \
	$(SCRIPT_DIR)/analysis_nearest_neighbor_distance.py \
		| nearest_neighbors-dir
	python $(SCRIPT_DIR)/analysis_nearest_neighbor_distance.py \
		--model BernoulliMLPVAE \
		--dataset BinarizedMNIST \
		--latent-dim 16 \
		--seed 42 \
		--checkpoint $< \
		--output $@

############
#          #
# Clean up #
#          #
############

.PHONY: clean

clean: clean_output clean_memorization

check_clean:
	@echo -n "This will remove all memorization result files. Are you sure? [y/N] " && read ans && [ "$$ans" == "y" ]

clean_output:
	rm -rf $(OUTPUT_DIR)
	rm -rf $(SUMMARY_DIR)

clean_memorization: check_clean
	rm -rf $(RESULT_DIR)

