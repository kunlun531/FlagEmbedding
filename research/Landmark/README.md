<div align="center">
<h1> Landmark embedding: a chunking-free embedding method for retrieval augmented long-context large language models. [<a href="https://arxiv.org/pdf/2402.11573">paper</a>]</h1>
</div>

Landmark Embedding is a new method for retrieval augmentation in long-context large language models. Instead of chunking text, it processes a coherent long context to generate high-quality embeddings for fine-grained units (e.g., sentences).

It is known for the following features:
- **Chunking-free architecture**: Keeps the long context coherent for high-quality, fine-grained embeddings.
- **Position-aware objective function**: Prioritizes the boundary of an information span, enabling comprehensive retrieval.
- **Multi-stage learning algorithm**: Effectively uses available data for cost-effective training.

## Environment
```bash
conda create -n landmark python=3.10
conda activate landmark

# You may need to adjust the cuda version
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers==4.41.0 deepspeed accelerate datasets peft pandas nltk
pip install flash-attn --no-build-isolation
```

## Model

| Model                                                        | Introduction                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Lk123/LMK](https://huggingface.co/Lk123/LMK) | This checkpoint was trained with the Mistral-7B backbone and corresponds to the Stage-II model described in the paper. For better downstream or domain-specific performance, it is recommended to fine-tune the model on a private dataset. |

## Usage
We provide an `LMKEmbedder` class to simplify the inference process. This class encapsulates the core logic for generating embeddings for both queries and passages.

- **For queries (`encode_queries`)**: This method automatically appends the landmark token (`eos_token`) to each query string.
- **For passages (`encode_corpus`)**: This is the core of Landmark Embedding. It takes one or more long passages, **automatically splits them into sentences** using `nltk`, and then generates a distinct, context-aware embedding for **each sentence** without applying chunking.

Here is an example of how to use the `LMKEmbedder`.

```python
import torch
from infer import LMKEmbedder  # Import the class from your inference script

# 1. Initialize LMKEmbedder
model = LMKEmbedder(
    "Lk123/LMK",
    use_fp16=True,
    devices=['cuda:0']
)

# 2. Define your query and passage(s)
queries = ['What is the primary color of a ripe banana?']
passages = [
    "The banana is a widely consumed tropical fruit that belongs to the genus Musa. It originates from "
    "Southeast Asia and has been cultivated for thousands of years. The fruit grows in large hanging clusters "
    "called hands, with individual bananas known as fingers. Bananas undergo a fascinating ripening process "
    "where the primary color transforms from green to yellow as chlorophyll breaks down and carotenoids "
    "become more visible. The primary color of a ripe banana is typically a bright, vibrant yellow, though "
    "some varieties may exhibit slight variations. As bananas continue to ripen beyond their peak, brown spots "
    "begin to appear and the skin gradually darkens. Before reaching full maturity, the skin maintains a "
    "solid green hue due to high chlorophyll content. Nutritionally, bananas are an excellent source of "
    "potassium, vitamin B6, and dietary fiber, making them a healthy choice for many diets. People worldwide "
    "enjoy bananas in various forms - eaten raw, blended into smoothies, baked in desserts, or sliced into "
    "breakfast cereals. In many tropical regions, bananas are also cooked as a vegetable when still green. "
    "The global banana trade represents a significant agricultural industry, with major exporters including "
    "Ecuador, the Philippines, and Costa Rica."
]

# 3. Encode the query
# The LMKEmbedder automatically adds the landmark token.
# Output shape: (1, embedding_dim)
q_embeddings = model.encode_queries(queries, convert_to_numpy=False)

# 4. Encode all sentences within the passage
# The LMKEmbedder automatically splits the passage into sentences and embeds each one.
# Output shape: (N, embedding_dim), where N is the number of sentences in the passage.
p_embeddings = model.encode_corpus(passages, convert_to_numpy=False)

# 5. Compute similarity scores between the query and each sentence embedding
scores = q_embeddings @ p_embeddings.T

print("Similarity Scores (Query vs. each sentence in passage):")
print(scores)
print("\nScores shape:")
print(scores.shape)
# Expected scores shape: (1, N), where N is the number of sentences in the passage.
```

## Citation

If you find this repository useful, please give us a star ⭐.

To cite our work:

```
@article{luo2024bge,
  title={Bge landmark embedding: A chunking-free embedding method for retrieval augmented long-context large language models},
  author={Luo, Kun and Liu, Zheng and Xiao, Shitao and Liu, Kang},
  journal={arXiv preprint arXiv:2402.11573},
  year={2024}
}
```
