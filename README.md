# Person Identification

This project focuses on two tasks: multi-class multi-label classification of people’s visual characteristics and re-identification in large datasets. 

### People Classification
The model classifies several visual attributes from a given image, such as:
- **Age** (young, teenager, adult, old)
- **Accessories** (hat, backpack, bag, handbag)
- **Clothing** (upper/lower body clothing type and color)
- **Physical Features** (hair length, gender)

To handle these diverse tasks efficiently, a **shared backbone** is used, with a pre-trained visual encoder feeding into a multi-head classification layer. This architecture improves the model’s flexibility and reusability across various attributes.

Additionally, an **Attribute Re-weighting Module** is employed, which refines predictions by considering the interrelationships between attributes. For example, age might influence the likelihood of certain clothing types or accessories. This calibration is based on the idea presented in the paper "[Improving Person Re-identification by Attribute and Identity Learning](https://arxiv.org/pdf/1703.07220.pdf)", where relationships between tasks are factored into the loss function.

Given that the difficulty of each task can vary, a **Dynamic Task Prioritization** approach is used. This method assigns a **Key Performance Indicator (KPI)** to each task, guiding the adjustment of loss weightings over time. Tasks that are more challenging are given higher loss weights, while simpler tasks are downweighted. This dynamic strategy ensures that the model adapts throughout training, improving convergence on harder tasks.

### People Re-Identification
For re-identification, the model extracts **embedding vectors** from the shared backbone network, which act as compact representations of individuals. These embeddings allow for **person matching** by calculating cosine similarity between images. This approach ensures that similar individuals, even with different poses, lighting, or backgrounds, can be matched accurately.

To improve the quality of these embeddings, the model incorporates a **Centroid Triplet Loss**, **Central Loss**, and **Island Loss**. These losses help cluster embeddings in feature space, making it easier to distinguish between different individuals while minimizing intra-person variance. Specifically:
- **Centroid Triplet Loss** ensures that embeddings are well-separated in the feature space, favoring individuals with distinct representations.
- **Central Loss** encourages embeddings to be close to their respective class centroids, enhancing intra-class compactness.
- **Island Loss** helps preserve uniqueness by maintaining consistent intra-class separation.

## Installation and Usage
For setting up the model and running the training pipeline, follow the instructions provided in the Jupyter Notebook.
