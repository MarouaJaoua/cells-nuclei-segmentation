.PHONY: clean data lint requirements sync_data_down sync_data_up

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = cell-nuclei-segmentation
PYTHON_VERSION = 3.9
PYTHON_INTERPRETER = python


#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 source




## Set up python interpreter environment
create_environment:
	
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

setup:
	$(PYTHON_INTERPRETER) setup.py install --user
## Preprocess data
preprocess: requirements setup
	$(PYTHON_INTERPRETER) source/preprocess/preprocess_data.py \
							--dataset "hpa" \
                            --device "cuda" \
                            --gpu_ids "2" \
                            --image_size 512

## Train model
train_nuclei: requirements setup
	$(PYTHON_INTERPRETER) source/train/train.py \
				  --num_kernel 16 \
                  --kernel_size 3 \
                  --lr  0.0001 \
                  --epoch 50 \
                  --dataset "hpa" \
                  --device "cuda" \
                  --optimizer "adam" \
                  --model "unet" \
                  --batch_size "32" \
                  --gpu_ids "2" \
                  --num_workers "16" \
                  --experiment_name "nuc_exp" \
                  --target_type "nuclei" \
                  --image_size 512

train_cells: requirements setup
	$(PYTHON_INTERPRETER) source/train/train.py \
				  --num_kernel 16 \
                  --kernel_size 3 \
                  --lr  0.0001 \
                  --epoch 50 \
                  --dataset "hpa" \
                  --device "cuda" \
                  --optimizer "adam" \
                  --model "unet" \
                  --batch_size "32" \
                  --gpu_ids "2" \
                  --num_workers "16" \
                  --experiment_name "cell_exp" \
                  --target_type "cell" \
                  --image_size 512


## Evaluate model
evaluate_nuclei: requirements setup
	$(PYTHON_INTERPRETER) source/evaluate/evaluate.py \
											  --num_kernel 8 \
                                              --kernel_size 3 \
                                              --lr  0.0001 \
                                              --epoch 50 \
                                              --dataset "hpa" \
                                              --device "cpu" \
                                              --optimizer "adam" \
                                              --model "unet" \
                                              --batch_size "16" \
                                              --gpu_ids "2" \
                                              --num_workers "16" \
                                              --experiment_name "810_nuclei_16" \
                                              --target_type "nuclei" \
                                              --image_size 512

evaluate_cells: requirements setup
	$(PYTHON_INTERPRETER) source/evaluate/evaluate.py \
											  --num_kernel 16 \
                                              --kernel_size 3 \
                                              --lr  0.0001 \
                                              --epoch 50 \
                                              --dataset "hpa" \
                                              --device "cpu" \
                                              --optimizer "adam" \
                                              --model "unet" \
                                              --batch_size "32" \
                                              --gpu_ids "2" \
                                              --num_workers "16" \
                                              --experiment_name "800_cell_32" \
                                              --target_type "cell" \
                                              --image_size 512


## Generate masks
generate_nuclei_mask: requirements setup
	$(PYTHON_INTERPRETER) source/evaluate/generate_masks.py \
											  --num_kernel 8 \
                                              --kernel_size 3 \
                                              --lr  0.0001 \
                                              --epoch 50 \
                                              --dataset "hpa" \
                                              --device "cpu" \
                                              --optimizer "adam" \
                                              --model "unet" \
                                              --batch_size "16" \
                                              --gpu_ids "2" \
                                              --num_workers "16" \
                                              --experiment_name "810_nuclei_16" \
                                              --target_type "nuclei" \
                                              --image_size 512

generate_cells_mask: requirements setup
	$(PYTHON_INTERPRETER) source/evaluate/generate_masks.py \
											  --num_kernel 16 \
                                              --kernel_size 3 \
                                              --lr  0.0001 \
                                              --epoch 50 \
                                              --dataset "hpa" \
                                              --device "cpu" \
                                              --optimizer "adam" \
                                              --model "unet" \
                                              --batch_size "32" \
                                              --gpu_ids "2" \
                                              --num_workers "16" \
                                              --experiment_name "800_cell_32" \
                                              --target_type "cell" \
                                              --image_size 512


test: requirements setup
	$(PYTHON_INTERPRETER) -m pytest

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
