# Structured Generation Evaluation

## Dataset
We use the dataset below:
[Arun63/sharegpt-structured-output-json](https://huggingface.co/datasets/Arun63/sharegpt-structured-output-json)
[Arun63/sharegpt-quizz-generation-json-output](https://huggingface.co/datasets/Arun63/sharegpt-quizz-generation-json-output/viewer/default/train?row=3&views%5B%5D=train)

Run the command below to prepare dataset
```bash
uv run download_sharegpt_data.py --output-dir sharegpt_data
```

## Generate structured data with LLM

