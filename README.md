# Subjective Acts and Intents
This repo contains the code and data for analyzing subjective judgments of witness responses in U.S. congressional hearings ([paper](https://arxiv.org/abs/2104.04470) to appear in NAACL 2021). If you make use of the data or code, please cite:

`
Ferracane, Elisa TBD
`

## Advanced Natural Language Processing HW4

Our changes to the codebase for curriculum learning(stage wise) can be viewed from `curriculum-learning-stage` branch and for the student-expert model, the branch name is `student-teacher-changes`

### Commands for training using Curriculum learning
#### Easy
```
python -u -m models.bert_hier.main_cv --dataset CongressionalHearingEasy --model-family roberta --model roberta-base --max-seq-length 512 --evaluate-test --patience 30 --lr 3e-5 --warmup-proportion 0.1 --weight-decay 0.1 --batch-size 8 --epochs 30 --seed 1234 --metrics-json metrics_roberta_easy_hierarchical_gold_sentiments_coarse_num_r_text_test_5_dec.json --first-input-column 16 --use-second-input --second-input-column 2 > ch_roberta_easy_hierarchical_gold_sentiments_coarse_num_r_text_test_5_dec.log
```
#### Medium
```
python -u -m models.bert_hier.main_cv --dataset CongressionalHearingMed --model-family roberta --model roberta-base --max-seq-length 512 --evaluate-test --patience 30 --lr 3e-5 --warmup-proportion 0.1 --weight-decay 0.1 --batch-size 8 --epochs 30 --seed 1234 --metrics-json metrics_roberta_med_hierarchical_gold_sentiments_coarse_num_r_text_test_5_dec.json --first-input-column 16 --use-second-input --second-input-column 2 > ch_roberta_med_hierarchical_gold_sentiments_coarse_num_r_text_test_5_dec.log
```

#### Hard
```
python -u -m models.bert_hier.main_cv --dataset CongressionalHearingHard --model-family roberta --model roberta-base --max-seq-length 512 --evaluate-test --patience 30 --lr 3e-5 --warmup-proportion 0.1 --weight-decay 0.1 --batch-size 8 --epochs 30 --seed 1234 --metrics-json metrics_roberta_hard_hierarchical_gold_sentiments_coarse_num_r_text_test_5_dec.json --first-input-column 16 --use-second-input --second-input-column 2 > ch_roberta_hard_hierarchical_gold_sentiments_coarse_num_r_text_test_5_dec.log
```

### Commands for training using Student-Expert model
#### Training expert model using only explanations
```
python -u -m models.bert_hier.main_cv --dataset CongressionalHearingExplanations --model-family roberta --model roberta-base --max-seq-length 512 --evaluate-test --patience 30 --lr 3e-5 --warmup-proportion 0.1 --weight-decay 0.1 --batch-size 8 --epochs 30 --seed 1234 --metrics-json roberta-explanations/metrics_roberta_hierarchical_explanations_concat_final.json --first-input-column 38 --save-path 'model_checkpoints/roberta-explanations-only' > roberta-explanations/ch_roberta_hierarchical_explanations_concat_final.log
```
#### Training student-expert model
```
python -u -m models.bert_hier.main_cv --dataset CongressionalHearingExplanations --model-family roberta --model roberta-base --max-seq-length 512 --evaluate-test --patience 30 --lr 3e-5 --warmup-proportion 0.1 --weight-decay 0.1 --batch-size 8 --epochs 30 --seed 1234 --metrics-json roberta-student-expert/metrics_roberta_hierarchical_student_expert_concat_test.json --first-input-column 16 --use-second-input --second-input-column 2 --use-third-input --third-input-column 38 --use_expert_model --expert_model_path_fold_0 ./model_checkpoints/roberta-explanations-only/CongressionalHearingFoldsExplanations/fold0/2022-12-08_18-03-55.pt --expert_model_path_fold_1 ./model_checkpoints/roberta-explanations-only/CongressionalHearingFoldsExplanations/fold1/2022-12-08_18-42-25.pt --expert_model_path_fold_2 ./model_checkpoints/roberta-explanations-only/CongressionalHearingFoldsExplanations/fold2/2022-12-08_19-20-58.pt --expert_model_path_fold_3 ./model_checkpoints/roberta-explanations-only/CongressionalHearingFoldsExplanations/fold3/2022-12-08_19-59-28.pt --expert_lambda 0.6 --save-path 'model_checkpoints/student_expert' > roberta-student-expert-colab-lambda/ch_roberta_hierarchical_student_expert_concat_test.log
```

## Dataset
If you're here just for the data, you can download it here: [gold_cv_dev_data.tar.gz](data/gold/gold_cv_dev_data.tar.gz). Unpack with `tar -zxvf gold_cv_dev_data.tar.gz`.

## Code
### Setup:
First, create a conda environment and activate it:
```
conda create --name subjective python=3.8
conda activate subjective
```

Install pytorch and cuda:
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

Clone this repo and install the requirements:
```
git clone https://github.com/elisaF/subjective_discourse
cd subjective_discourse
pip install -r requirements.txt
```

Unpack the data splits:
```
cd subjective_discourse/data/gold
tar -zxvf gold_cv_dev_data.tar.gz
```

### Classification Task:
The multi-label classification task consists of predicting all the possible response labels, and is evaluated with macro-averaged F1.

**Roberta:** predict the response labels using the response text.

```shell
python -u -m models.bert.main_cv --dataset CongressionalHearing --model-family roberta --model roberta-base --max-seq-length 512 --evaluate-test --patience 5 --lr 3e-5 --warmup-proportion 0.1 --weight-decay 0.1 --batch-size 8 --epochs 30 --seed 1234 --metrics-json metrics_roberta_classification_r_text_test.json --first-input-column 2  > ch_roberta_classification_r_text_test.log 2>&1
```

**Hierarchical:** predict the response labels while also training to predict the conversation acts.

```shell
python -u -m models.bert_hier.main_cv --dataset CongressionalHearing --model-family roberta --model roberta-base --max-seq-length 512 --evaluate-test --patience 30 --lr 3e-5 --warmup-proportion 0.1 --weight-decay 0.1 --batch-size 8 --epochs 30 --seed 1234 --metrics-json metrics_roberta_hierarchical_r_text_test.json --first-input-column 2 > ch_roberta_hierarchical_r_text_test.log 2>&1 &
```

**+Question:** predict the response labels as in the hierarchical model, but additionally using the last question.
```shell
python -u -m models.bert_hier.main_cv --dataset CongressionalHearing --model-family roberta --model roberta-base --max-seq-length 512 --evaluate-test --patience 30 --lr 3e-5 --warmup-proportion 0.1 --weight-decay 0.1 --batch-size 8 --epochs 30 --seed 1234 --metrics-json metrics_roberta_hierarchical_q_text_last_question_r_text_test.json --first-input-column 4 --use-second-input --second-input-column 2  > ch_roberta_hierarchical_q_text_last_question_r_text_test.log 2>&1
```

**+Annotator:** predict the response labels as in the hierarchical model, but additionally using the annotator sentiments.
```shell
python -u -m models.bert_hier.main_cv --dataset CongressionalHearing --model-family roberta --model roberta-base --max-seq-length 512 --evaluate-test --patience 30 --lr 3e-5 --warmup-proportion 0.1 --weight-decay 0.1 --batch-size 8 --epochs 30 --seed 1234 --metrics-json metrics_roberta_hierarchical_gold_sentiments_coarse_num_r_text_test.json --first-input-column 16 --use-second-input --second-input-column 2  > ch_roberta_hierarchical_gold_sentiments_coarse_num_r_text_test.log 2>&1
```

**+Question+Annotator:** predict the response labels as in the hierarchical model, but additionally using the last question and the annotator sentiments.
```shell
python -u -m models.bert_hier.main_cv --dataset CongressionalHearing --model-family roberta --model roberta-base --max-seq-length 512 --evaluate-test --patience 30 --lr 3e-5 --warmup-proportion 0.1 --weight-decay 0.1 --batch-size 8 --epochs 30 --seed 1234 --metrics-json metrics_roberta_hierarchical_r_text_gold_sentiments_coarse_num_q_text_last_question_test.json --first-input-column 2 --use-second-input --second-input-column 16  --use-third-input --third-input-column 4 > ch_roberta_hierarchical_r_text_gold_sentiments_coarse_num_q_text_last_question__test.log 2>&1
```
### Regression Task:
The regression task consists of predicting the normalized entropy of the response label distribution, and is evaluated with RMSE.

**Roberta:** predict the response labels using only the response text. Note this experimental model is run on the dev fold
```
cd subjective_discourse/code/models/hedwig
../../shell_scripts/run_roberta_regression_dev.sh
```
