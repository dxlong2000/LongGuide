# [ACL 2025] Beyond In-Context Learning: Aligning Long-form Generation of Large Language Models via Task-Inherent Attribute Guidelines

[![arXiv](https://img.shields.io/badge/arXiv-2506.01265-b31b1b.svg)](https://arxiv.org/abs/2506.01265)

Official codes for ACL 2025 Findings paper "Beyond In-Context Learning: Aligning Long-form Generation of Large Language Models via Task-Inherent Attribute Guidelines"

## Quick Start

### Configuration

Edit `configs/default.yaml`:

```yaml
# Model Configuration
model_name: "gpt-3.5-turbo"
api_key: "your-openai-api-key-here"

# Task Configuration
task_type: "summarization"  # summarization, translation, dialogue generation, table-to-text generation, text simplification
```

### Installation

```bash
cd src
pip install -r requirements.txt
```

### Running LongGuide

```bash
# Run with default config
python run.py

# Run with custom config
python run.py --config configs/custom.yaml
```

### Evaluation

```bash
# Evaluate cnn results
python evaluate.py --results outputs/results_summarization.json --task summarization

# Evaluate translation results  
python evaluate.py --results outputs/results_translation.json --task translation
```

## Supported Datasets

- **SAMSum**: Dialogue summarization
- **CNN**: News summarization  
- **xlsum**: Cross-lingual summarization
- **SWiPE**: Text simplification
- **IWSLT**: Translation (EN→JA)
- **CommonGen**: Concept-to-text generation
- **SyntheticDialogue**: Dialogue generation

## Project Structure

```
src/
├── longguide/           # Core LongGuide implementation
├── data/               # Datasets
├── configs/            # Configuration files
├── outputs/            # Generated results
├── run.py             # Main execution script
├── evaluate.py        # Evaluation script
└── standardize_data.py # Data preprocessing
```

## Evaluation Metrics

- **Standard tasks**: ROUGE-L with stemmer
- **Translation**: ROUGE-L with GPT tokenizer for cross-lingual evaluation

## Citation

If you found our work helpful, please cite it:

```bibtex
@inproceedings{long-etal-2025-beyond,
    title = "Beyond In-Context Learning: Aligning Long-form Generation of Large Language Models via Task-Inherent Attribute Guidelines",
    author = "Long, Do Xuan  and
      Yen, Duong Ngoc  and
      Trong, Do Xuan  and
      Luu, Anh Tuan  and
      Kawaguchi, Kenji  and
      Joty, Shafiq  and
      Kan, Min-Yen  and
      Chen, Nancy F.",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.176/",
    doi = "10.18653/v1/2025.findings-acl.176",
    pages = "3377--3411",
    ISBN = "979-8-89176-256-5",
    abstract = "In-context learning (ICL) is an important yet not fully understood ability of pre-trained large language models (LLMs). It can greatly enhance task performance using a few examples, termed demonstrations, without fine-tuning. Although effective in question answering, ICL often underperforms in long-form generation tasks such as summarization. Under appropriately realistic assumptions, we empirically and theoretically show that ICL demonstrations alone are insufficient to teach LLMs the task{'}s language and format distributions for generation. We argue for explicit exposure to the task distributions and hypothesize that defining them by prompting enhances model performance. To this end, we present LongGuide, which efficiently generates two parallel streams of guidelines capturing task language and format properties: (i) Metric Guidelines (MGs) that instruct models to optimize self-evaluated metrics; and (ii) Output Constraint Guidelines (OCGs) that constrain generation at both token and sentence levels. LongGuide automatically selects the best combination of guidelines, improving both strong open- and closed-source LLMs by over 5{\%} in both zero- and few-shot settings. We show that LongGuide is generalizable, learnable by weak models to enhance strong ones, and integrates synergistically with automatic prompt optimizers."
}
```
