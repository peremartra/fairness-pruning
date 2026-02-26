# CaBBQ - YAML Configuration Files for lm-evaluation-harness

Catalan Bias Benchmark for Question Answering. Task configs for evaluating social bias in LLMs using the [BSC-LT/CaBBQ](https://huggingface.co/datasets/BSC-LT/CaBBQ) dataset via [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## 10 YAML Files:

1. **cabbq_age.yaml** - Age
2. **cabbq_disabilitystatus.yaml** - Disability Status
3. **cabbq_gender.yaml** - Gender
4. **cabbq_lgbtqia.yaml** - LGBTQIA
5. **cabbq_nationality.yaml** - Nationality
6. **cabbq_physicalappearance.yaml** - Physical Appearance
7. **cabbq_raceethnicity.yaml** - Race/Ethnicity
8. **cabbq_religion.yaml** - Religion
9. **cabbq_ses.yaml** - Socioeconomic Status
10. **cabbq_spanishregion.yaml** - Spanish Region

## Evaluation Commands:

### Run ALL categories:
```bash
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-3.2-1B \
  --tasks cabbq_age,cabbq_disabilitystatus,cabbq_gender,cabbq_lgbtqia,cabbq_nationality,cabbq_physicalappearance,cabbq_raceethnicity,cabbq_religion,cabbq_ses,cabbq_spanishregion \
  --include_path custom_tasks/cabbq/ \
  --device cuda \
  --batch_size 8
```

### Run ONE specific category:
```bash
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-3.2-1B \
  --tasks cabbq_gender \
  --include_path custom_tasks/cabbq/ \
  --device cuda
```

### Quick test (10 examples):
```bash
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-3.2-1B \
  --tasks cabbq_gender \
  --include_path custom_tasks/cabbq/ \
  --limit 10
```

## YAML Structure:

- `task`: unique name per category (e.g., cabbq_age)
- `dataset_path`: `BSC-LT/CaBBQ`
- `dataset_name`: exact subset name from HuggingFace
- `doc_to_text`: Catalan prompt template
- `doc_to_choice`: `[ans0, ans1, ans2]` - three answer options
- `num_fewshot: 0` - zero-shot evaluation as per paper

## Notes:

- Follows the same structure as the EsBBQ configs in `custom_tasks/esbbq/`
- Prompt is in Catalan ("Context", "Pregunta", "Resposta")
- Zero-shot, multiple-choice (3 options: stereotype, anti-stereotype, unknown)
- Decontamination enabled

## Dataset Reference:

- **Dataset**: https://huggingface.co/datasets/BSC-LT/CaBBQ
- **Related paper**: Ruiz-Fernandez et al. (2025). "EsBBQ: A Spanish Bias Benchmark for Question Answering" arXiv:2507.11216
- **Code**: https://github.com/langtech-bsc/EsBBQ-CaBBQ