<div align= "center">
    <h1> CSE 5525 - Final Project </h1>
    <h4> Kai Zhang (zhang.13253), Zhongwei Wan (wan.512), Kuan Chieh Lo (lo.311)</h4>
    <h4>First Time NLP</h4>
</div>

## Introduction

Large Language Models (LLMs) are trained on vast amounts of text data and have demonstrated remarkable capabilities in understanding and generating natural language. However, many LLMs have not been explicitly trained on visual data. Despite this, LLMs may have latent knowledge about visual objects through textual descriptions. For instance, a model trained only on text may have never "seen" a tree, yet it could have processed numerous descriptions detailing a tree's appearance.

Understanding whether LLMs can learn and reason about visual objects through language is interesting. It can test whether LLMs can ground language to visual objects. Also, if LLMs can conceptualize visual objects based solely on text, it would expand their application in multimodal tasks without necessitating additional training on expensive visual data. An example of this would be asking an LLM to describe the shape of a common object, such as a circle, and assessing whether the model correctly identifies attributes commonly associated with circles, such as "round" or "smooth".

## Installation

Please run the script below to install the required dependencies.

```bash
pip install -r requirements.txt
```

## Data

We created a diverse collection of image inputs to evaluate the LLM's ability to interpret visual elements across varying complexity levels. Simple visual representations were generated using the [ASCII Art](https://www.asciiart.eu/image-to-ascii) online converter.

The dataset has only one evaluation set located in the `./data` directory.
Each data sample contains the following key attributes:

* `url`: Source link to the original content
* `ascii_art`: ASCII representation in text format
* `category-1/2/3`: Hierarchical classification labels describing the ASCII art content
* `choices`: Available options for the recognition task, including both correct and incorrect answers
* `labels`: Binary indicators corresponding to each choice
* `image_path`: Reference to the associated image file

To begin with, please run the following script for data preprocessing:

```bash
bash script/prepare_data.sh
```

## Script

To evaluate models through API:
```bash
export API_KEY=xxx
python3 src/evaluation_by_api.py --api_key $API_KEY --model_name xxx --base_url https://xxxxxxxx/v1 --template_name xxx --output_file_path xxx.jsonl --mode text-only
```
