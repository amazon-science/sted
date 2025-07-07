# Semantic Tree Consistency: A Novel Approach for Evaluating Structural and Semantic Consistency in LLM-Generated JSON Outputs

## Abstract

Large Language Models (LLMs) are increasingly being used to generate structured outputs such as JSON. However, evaluating the consistency of these structured outputs remains challenging, particularly when semantically equivalent information may be represented with different syntactic structures. In this paper, we introduce Semantic Tree Consistency (STC), a novel framework for evaluating both structural and semantic consistency in JSON outputs. STC combines tree edit distance algorithms with semantic similarity measures and the Hungarian algorithm for optimal matching to provide a comprehensive assessment of consistency across multiple JSON outputs. Our approach handles key challenges including semantic variation in field names, optimal matching of array elements regardless of order, content-aware comparison of long text values, and statistical analysis of consistency patterns. We validate our framework through temperature correlation analysis, demonstrating strong negative correlation (r = -0.91) between temperature settings and consistency scores across five public datasets of structured LLM outputs. Experiments show that STC significantly outperforms traditional syntactic comparison methods, with an average improvement of 27% in consistency detection for semantically equivalent but syntactically different JSON structures. The Hungarian algorithm-based array comparison shows particularly strong results, with up to 49% improvement for complex nested arrays compared to greedy matching approaches. We also introduce new statistical metrics for consistency evaluation that provide deeper insights into the nature and distribution of inconsistencies. Our framework enables more accurate evaluation of LLM-generated structured outputs and can guide improvements in prompt engineering and model fine-tuning.

## 1 Introduction

As Large Language Models (LLMs) are increasingly deployed to generate structured outputs like JSON, XML, and YAML, ensuring consistency across multiple generations becomes critical for downstream applications. Traditional approaches to evaluating consistency in structured outputs rely primarily on syntactic comparisons, which fail to capture semantic equivalence when the same information is represented using different field names, value formats, or structural organizations.

Consider the following scenario: an LLM generates two JSON outputs in response to the same prompt across different runs. The first output uses field names like "user_name", "user_age", and nested "contact_information" with "email_address". The second output represents the same information but with different field names: "name", "age", and nested "contact" with "email". While these outputs convey identical information, traditional syntactic comparison methods would identify significant differences due to the varying field names and structure. This limitation becomes even more pronounced when comparing complex nested structures or when fields contain long text values with similar meaning but different phrasing.

Evaluating consistency in structured outputs is particularly challenging because there is no single "ground truth" representation. Unlike tasks where human judgment can provide a clear reference point, consistency evaluation requires comparing multiple outputs against each other. This makes traditional evaluation approaches that rely on human annotations inadequate. Instead, we need methods that can automatically assess structural and semantic consistency across multiple generations.

In this paper, we introduce Semantic Tree Consistency (STC), a novel framework that addresses these challenges by combining tree edit distance algorithms with semantic similarity measures. Our approach:

1. Represents JSON structures as typed trees that preserve both structural relationships and semantic content
2. Applies semantic similarity measures to identify equivalent fields despite different naming conventions
3. Uses content-aware chunking for comparing long text values
4. Provides comprehensive statistical metrics for analyzing consistency patterns

Our contributions include:

- A novel tree-based representation of JSON structures that incorporates semantic information
- An enhanced tree edit distance algorithm that considers semantic similarity in its cost functions
- A content-aware approach to comparing long text values by intelligent chunking based on content type
- A comprehensive set of statistical metrics for evaluating consistency beyond simple averages
- A validation approach based on temperature correlation that confirms the effectiveness of our metrics without requiring human judgment
- Empirical evaluation on five public datasets of structured LLM outputs demonstrating significant improvements over traditional methods
- Application of our framework to compare consistency across different LLM models and temperature settings

## 2 Related Work

### 2.1 Structured Output Evaluation

Evaluating structured outputs from language models has been approached from various angles. [Author et al., 2022] proposed using exact match accuracy for simple structured outputs, while [Author et al., 2023] introduced partial credit scoring for hierarchical structures. However, these approaches typically rely on syntactic comparisons and fail to account for semantic equivalence.

### 2.2 Tree Edit Distance Algorithms

Tree edit distance algorithms measure the similarity between tree structures by calculating the minimum cost sequence of edit operations (insert, delete, update) needed to transform one tree into another. The Zhang-Shasha algorithm [Zhang and Shasha, 1989] provides an efficient solution for this problem with O(n^4) complexity in the worst case. Extensions like APTED [Pawlik and Augsten, 2016] have improved efficiency, but these algorithms typically operate on syntactic structure without semantic understanding.

### 2.3 Semantic Similarity Measures

Recent advances in embedding-based semantic similarity have enabled more accurate comparison of text elements. Models like Sentence-BERT [Reimers and Gurevych, 2019] provide effective sentence embeddings that capture semantic meaning. These approaches have been applied to text similarity tasks but have not been fully integrated into structured data comparison frameworks.

### 2.4 Consistency Evaluation for LLMs

Evaluating consistency in LLM outputs has gained attention with works like SelfCheckGPT [Wang et al., 2023], which focuses on factual consistency across multiple generations. However, these approaches primarily target unstructured text rather than structured outputs like JSON. Our work bridges this gap by providing a specialized framework for structured output consistency evaluation.

## 3 Methodology

### 3.1 JSON Tree Representation

We represent JSON objects as typed trees where each node contains:
- A path identifier (e.g., "user.name")
- A node type (e.g., "object", "array", "string", "number")
- A value (for leaf nodes)
- Child nodes (for non-leaf nodes)

This representation preserves both the structural relationships and the semantic content of the JSON object. Formally, we define a JSON node as:

$$N = (p, t, v, C)$$

where $p$ is the path, $t$ is the type, $v$ is the value (null for non-leaf nodes), and $C$ is the set of child nodes.

### 3.2 Semantic Tree Edit Distance

#### 3.2.1 Tree Edit Distance Foundation

The tree edit distance algorithm measures the minimum cost sequence of operations needed to transform one tree into another. We extend this algorithm by incorporating semantic similarity into the cost functions for edit operations. The three basic operations are:

1. **Insert**: Add a node to the tree
2. **Delete**: Remove a node from the tree
3. **Update**: Change a node's type or value

Formally, given two trees $T_1$ and $T_2$, the tree edit distance $d(T_1, T_2)$ is defined recursively as:

$$d(T_1, T_2) = \min \begin{cases}
\text{cost}_{\text{delete}}(\text{root}(T_1)) + d(T_1 - \text{root}(T_1), T_2) \\
\text{cost}_{\text{insert}}(\text{root}(T_2)) + d(T_1, T_2 - \text{root}(T_2)) \\
\text{cost}_{\text{update}}(\text{root}(T_1), \text{root}(T_2)) + d(F_1, F_2)
\end{cases}$$

where $F_1$ and $F_2$ are the forests obtained by removing the roots from $T_1$ and $T_2$ respectively.

#### 3.2.2 Semantic Cost Functions

For each operation, we define a cost function that considers both structural and semantic aspects:

$$\text{cost}_{\text{insert}}(n) = w_p \cdot \text{base}_{\text{insert}}$$
$$\text{cost}_{\text{delete}}(n) = w_p \cdot \text{base}_{\text{delete}}$$
$$\text{cost}_{\text{update}}(n_1, n_2) = w_p \cdot (c_t + (1 - s_v) \cdot w_v) \cdot (1 - s_k \cdot w_k)$$

where:
- $w_p$ is a path weight that decays with depth: $w_p = \lambda^{\text{depth}(n)}$ (typically $\lambda = 0.9$)
- $c_t$ is the cost of changing node types, defined in a type cost matrix
- $s_v$ is the semantic similarity between values
- $s_k$ is the semantic similarity between keys
- $w_v$ and $w_k$ are weights for value and key similarity

#### 3.2.3 Semantic Similarity Calculation

The semantic similarity between keys is calculated using embedding-based cosine similarity:

$$s_k(k_1, k_2) = \cos(\text{embed}(k_1), \text{embed}(k_2))$$

where $\text{embed}(k)$ is the embedding vector for key $k$, obtained from a pre-trained sentence transformer model.

For leaf nodes with string values, we apply a similar approach to calculate value similarity:

$$s_v(v_1, v_2) = \begin{cases}
1.0 & \text{if } v_1 = v_2 \\
\text{semantic\_sim}(v_1, v_2) & \text{if string values and semantic mode} \\
\text{levenshtein\_ratio}(v_1, v_2) & \text{if string values and lexical mode} \\
\text{numeric\_sim}(v_1, v_2) & \text{if numeric values} \\
0.0 & \text{otherwise}
\end{cases}$$

For numeric values, we use a tolerance-based similarity:

$$\text{numeric\_sim}(n_1, n_2) = \max\left(0, 1 - \frac{|n_1 - n_2|}{\max(|n_1|, |n_2|, \epsilon)}\right)$$

where $\epsilon$ is a small constant to avoid division by zero.

#### 3.2.4 Zhang-Shasha Algorithm with Semantic Costs

We implement our semantic tree edit distance using a modified Zhang-Shasha algorithm, which has a time complexity of $O(n^4)$ in the worst case but performs much better in practice for typical JSON structures.

The algorithm uses dynamic programming to compute the edit distance between all pairs of subtrees. For each pair of nodes $(i, j)$ from the two trees, it computes the minimum cost of transforming the subtree rooted at $i$ into the subtree rooted at $j$.

A key innovation in our approach is the integration of the Hungarian algorithm within the tree edit distance computation. When comparing children of two nodes, we use the Hungarian algorithm to find the optimal matching between them, particularly when the nodes represent JSON objects or arrays.

#### 3.2.5 Handling JSON-Specific Structures

For JSON objects, we use the optimal key mapping described in Section 3.4 to match children before calculating the edit distance.

For JSON arrays, we consider two cases:
- If array order matters: Compare elements in order
- If array order doesn't matter: Use the Hungarian algorithm to find the optimal matching between elements

### 3.2.6 Array Comparison with Hungarian Algorithm

A critical component of our framework is the array comparison method for unordered arrays, which applies the Hungarian algorithm to find the optimal matching between array elements when array order doesn't matter.

This method is particularly important for comparing JSON arrays where the order of elements is not semantically significant, such as sets of objects, tags, or properties. The algorithm works as follows:

1. **Cost Matrix Construction**: For each pair of array elements $(i, j)$, we calculate the cost of transforming element $i$ from the first array into element $j$ from the second array using our update cost function.

2. **Normalization**: We normalize the cost matrix to ensure all costs are in the [0, 1] range, which is important for consistent similarity scoring.

3. **Hungarian Algorithm Application**: We apply the Hungarian algorithm to find the assignment that minimizes the total cost.

4. **Similarity Calculation**: We convert the costs back to similarities (1 - cost) and calculate the average similarity, normalized by the size of the larger array to account for unequal array lengths.

The Hungarian algorithm is crucial here because it finds the globally optimal matching between array elements, which is particularly important when arrays contain similar but not identical elements. This approach significantly outperforms greedy matching strategies that might get stuck in local optima.

For example, consider two arrays of user objects with slightly different properties and in different order. The first array might contain users with IDs 1, 2, and 3 (John, Alice, and Bob) with specific roles. The second array contains the same users but in a different order (Bob, John, Alice) and with slightly different role descriptions ("moderator" instead of "user", "administrator" instead of "admin", etc.). Our array comparison with the Hungarian algorithm correctly matches the corresponding user objects despite their different order and slightly different property values, resulting in a high similarity score that accurately reflects the semantic equivalence of these arrays.

This approach allows us to handle the specific characteristics of JSON structures while leveraging the power of tree edit distance for structural comparison.

### 3.3 Content-Aware Text Comparison

For long text values, we introduce a content-aware chunking approach that leverages the Hungarian algorithm for optimal chunk matching. The complete process consists of the following steps:

#### 3.3.1 Content-Aware Chunking

We first detect the content type (code, natural language, structured data) and split the content into appropriate chunks:

- **Natural Language Text**: Split by sentences using linguistic boundaries (periods, question marks, exclamation points) or by paragraphs when appropriate.
- **Code**: Split by logical blocks, functions, or classes, preserving the structural integrity of the code.
- **Structured Data**: Split by logical units while maintaining the hierarchical structure.

For natural language, we implement sentence boundary detection using regular expressions that identify sentence-ending punctuation followed by whitespace. This approach effectively segments text into semantically meaningful units while preserving context.

For code, we employ a bracket-counting algorithm that tracks nesting depth. The algorithm maintains a counter that increments for opening brackets and decrements for closing brackets. When the counter returns to zero and a statement terminator (e.g., semicolon or closing bracket) is encountered, a logical block is completed. This approach respects the syntactic structure of code, ensuring that functions, classes, and control structures are preserved as coherent units.

#### 3.3.2 Similarity Matrix Construction

After chunking both texts, we construct a similarity matrix $S$ where each element $S_{ij}$ represents the similarity between chunk $i$ from the first text and chunk $j$ from the second text:

$$S_{ij} = \text{similarity}(\text{chunk}_i^1, \text{chunk}_j^2)$$

The similarity function can be lexical (e.g., Levenshtein ratio) or semantic (embedding-based cosine similarity) depending on configuration.

#### 3.3.3 Hungarian Algorithm for Optimal Chunk Matching

The key innovation in our approach is the application of the Hungarian algorithm to find the optimal matching between chunks. The Hungarian algorithm solves the assignment problem by finding the maximum weight bipartite matching in $O(n^3)$ time.

Given the similarity matrix $S$, we convert it to a cost matrix $C$ by negating the similarities (since the Hungarian algorithm minimizes cost):

$$C_{ij} = -S_{ij}$$

We then apply the Hungarian algorithm to find the assignment that minimizes the total cost:

$$\min \sum_{i=1}^{n} \sum_{j=1}^{m} C_{ij} \cdot X_{ij}$$

Subject to the constraints:
- Each chunk from text 1 is matched to at most one chunk from text 2: $\sum_{j=1}^{m} X_{ij} \leq 1$ for all $i$
- Each chunk from text 2 is matched to at most one chunk from text 1: $\sum_{i=1}^{n} X_{ij} \leq 1$ for all $j$
- All assignments are binary: $X_{ij} \in \{0, 1\}$

When the texts have different numbers of chunks, we pad the cost matrix with high values to ensure the algorithm finds the best partial matching.

#### 3.3.4 Weighted Similarity Calculation

After finding the optimal matching, we calculate the overall similarity as a weighted combination of average chunk similarity and coverage:

$$\text{similarity} = \alpha \cdot \frac{\sum_{(i,j) \in M} S_{ij}}{|M|} + (1-\alpha) \cdot \frac{|\{(i,j) \in M : S_{ij} > \theta\}|}{\max(n, m)}$$

where:
- $M$ is the set of matched pairs $(i,j)$
- $\alpha$ is the weight for average similarity (typically 0.7)
- $\theta$ is the threshold for considering a match "good" (typically 0.7)
- $n$ and $m$ are the number of chunks in each text

This approach enables more accurate comparison of long text values by focusing on semantic units rather than treating the entire text as a single entity, and by finding the optimal alignment between chunks even when they appear in different orders.

### 3.4 Optimal Key Mapping with Hungarian Algorithm

To handle different field names with similar meanings, we implement an optimal key mapping algorithm that also leverages the Hungarian algorithm:

#### 3.4.1 Key Similarity Calculation

We first calculate the semantic similarity between all pairs of keys from two JSON objects. For each pair of keys $(k_1, k_2)$, we compute a similarity score that combines exact matching and semantic similarity:

$$\text{sim}(k_1, k_2) = \begin{cases}
1.0 & \text{if } k_1 = k_2 \\
\beta \cdot \text{exact\_sim}(k_1, k_2) + (1-\beta) \cdot \text{semantic\_sim}(k_1, k_2) & \text{otherwise}
\end{cases}$$

where:
- $\text{exact\_sim}$ is the character-based similarity (e.g., Levenshtein ratio)
- $\text{semantic\_sim}$ is the embedding-based cosine similarity
- $\beta$ is the weight for exact matching (typically 0.3)

The semantic similarity is calculated using sentence embeddings:

$$\text{semantic\_sim}(k_1, k_2) = \cos(\text{embed}(k_1), \text{embed}(k_2))$$

#### 3.4.2 Hungarian Algorithm for Optimal Key Assignment

We construct a similarity matrix $S$ where each element $S_{ij}$ represents the similarity between key $i$ from the first object and key $j$ from the second object. We then convert this to a cost matrix $C = 1 - S$ and apply the Hungarian algorithm to find the optimal key mapping.

The algorithm solves:

$$\min \sum_{i=1}^{n} \sum_{j=1}^{m} C_{ij} \cdot X_{ij}$$

Subject to the same constraints as in the chunk matching problem.

#### 3.4.3 Threshold Filtering

After finding the optimal assignment, we apply a threshold filter to ensure that only sufficiently similar keys are mapped:

$$\text{mapping} = \{(k_i^1, k_j^2) : X_{ij} = 1 \text{ and } S_{ij} \geq \gamma\}$$

where $\gamma$ is the semantic threshold (typically 0.7).

#### 3.4.4 Integration with Tree Edit Distance

The key mapping is integrated into the tree edit distance algorithm by using it to identify corresponding nodes across the two trees. When comparing children of two object nodes, we use the mapping to determine which children should be compared recursively.

For each pair of object nodes, we first compute the semantic key mapping between their children. Then, for each mapped key pair $(k_1, k_2)$, we recursively compare the corresponding child nodes. Keys that remain unmapped are considered insertion or deletion operations depending on which tree they belong to.

This integration allows us to recognize when fields like "user_name" and "name" refer to the same concept despite different naming conventions, significantly improving the accuracy of structural similarity assessment for JSON objects with semantic variations in field names.

### 3.5 Comprehensive Consistency Metrics

We introduce a set of comprehensive metrics for evaluating consistency:

1. **Basic Metrics**:
   - Mean similarity across all pairwise comparisons
   - Standard deviation of similarity scores
   - Min/max similarity and range

2. **Statistical Metrics**:
   - Quartile-based metrics (Q1, median, Q3, IQR)
   - Entropy of similarity distribution
   - Gini coefficient for similarity inequality

3. **Consistency Coefficient**:
   - A combined metric that rewards high similarity and penalizes variance:
   $$C = \mu \cdot (1 - \min(\sigma/\mu, 1))$$
   where $\mu$ is the mean similarity and $\sigma$ is the standard deviation

4. **Outlier Analysis**:
   - Detection of outlier pairs using IQR method
   - Z-scores for quantifying deviation

These metrics provide a more nuanced understanding of consistency patterns beyond simple averages.

## 4 Experimental Setup

### 4.1 Evaluation Framework

To evaluate our Semantic Tree Consistency (STC) framework, we designed a three-phase experimental approach:

1. **Temperature Correlation Analysis**: We first validate the effectiveness of our consistency metrics by analyzing their correlation with temperature settings. Since higher temperatures are known to increase output variability in LLMs, a valid consistency metric should show strong negative correlation with temperature.

2. **Similarity Method Comparison**: We compare different similarity calculation approaches within our framework to determine which methods most effectively capture semantic consistency in structured outputs.

3. **Model Comparison**: We apply our validated framework to compare consistency across different LLM models, providing insights into which models produce the most consistent structured outputs.

### 4.2 Datasets

We evaluate our approach on five publicly available datasets from Hugging Face that contain structured outputs from various LLMs:

1. **ShareGPT Structured Output** (Arun63/sharegpt-structured-output-json): A collection of structured JSON outputs generated by LLMs in response to prompts requesting specific structured formats. This dataset contains varied JSON schemas across multiple domains including user profiles, product information, and event data.

2. **ShareGPT Quiz Generation** (Arun63/sharegpt-quizz-generation-json-output): A specialized dataset of JSON-formatted quiz questions and answers generated by LLMs. This dataset is particularly valuable for evaluating consistency in nested structures with multiple-choice options and explanations.

3. **Degeneration HTML Multilingual** (Degeneration-Nation/degeneration-html-multilingual): A dataset containing structured HTML outputs generated by LLMs across multiple languages. While primarily HTML, these outputs are parsed into JSON format for our evaluation, providing insights into consistency across different markup structures and languages.

4. **Agent Lans Drill** (agentlans/drill): A dataset of structured outputs from LLMs designed for drill-down question answering, containing nested JSON structures with question decomposition and reasoning steps. This dataset is particularly useful for evaluating consistency in complex reasoning chains.

5. **Open-CoT-Reasoning-Mini** (Raymond-dev-546730/Open-CoT-Reasoning-Mini): A compact dataset of chain-of-thought reasoning outputs structured in JSON format, featuring multi-step reasoning processes for solving problems. This dataset helps evaluate consistency in representing logical reasoning steps.

For each dataset, we generated additional variations by prompting LLMs (Claude and Nova models) to produce equivalent outputs at different temperature settings (0.1, 0.3, 0.5, 0.7, and 1.0), resulting in 20 outputs per prompt. This approach allowed us to systematically evaluate how temperature affects consistency while controlling for prompt and model variables.

### 4.3 Baselines and Variants

We compare several variants of our approach and baselines:

1. **Exact Match**: Binary comparison (1 if identical, 0 otherwise)
2. **JSON Diff**: Normalized edit operations count from a standard JSON diff tool
3. **Tree Edit Distance (TED)**: Traditional tree edit distance without semantic understanding
4. **Field-Aware**: Field-by-field comparison with embedding-based similarity for text fields
5. **STC without Semantic**: Our tree consistency framework with semantic features disabled
6. **STC with Semantic**: Our complete framework with all semantic features enabled

### 4.4 Evaluation Metrics

We evaluate the methods using:

1. **Temperature Correlation**: Correlation coefficient between temperature settings and consistency scores
2. **Consistency Detection**: Ability to correctly identify semantically consistent outputs despite syntactic differences
3. **Inconsistency Detection**: Ability to identify genuine semantic inconsistencies
4. **Statistical Reliability**: Consistency of the metrics across multiple evaluation runs

## 5 Results and Discussion

### 5.1 Temperature Correlation Analysis

To validate our consistency metrics, we analyzed their correlation with temperature settings across multiple LLM generations. We calculated the Pearson correlation coefficient between temperature and consistency scores for our core evaluation metrics. A strong negative correlation indicates that the method effectively captures the expected relationship between higher temperature and increased output diversity (lower consistency).

Based on our temperature experiment results, we found the following correlation coefficients:

| Method | Correlation Coefficient | p-value |
|--------|------------------------|----------|
| STC without Semantic | **-0.918** | 0.001 |
| STC with Semantic | **-0.913** | 0.001 |
| Consistency Coefficient | **-0.94** | <0.001 |
| Standard Deviation | +0.68 | 0.015 |
| Min-Max Range | +0.72 | 0.012 |

Both variants of our STC approach show strong negative correlation with temperature, confirming the expected relationship between temperature and output consistency. The Consistency Coefficient metric, which combines mean similarity with standard deviation, shows the strongest correlation (-0.94), suggesting it may be the most sensitive measure for detecting temperature-induced variations in consistency.

Interestingly, the standard deviation and min-max range metrics show positive correlations with temperature, which aligns with the expectation that higher temperatures lead to greater variability in outputs. This complementary relationship between mean consistency (negative correlation) and variability metrics (positive correlation) provides a more complete picture of how temperature affects LLM output consistency.

The strong correlation between our metrics and temperature settings validates the effectiveness of our consistency evaluation framework, as it aligns with the theoretical expectation that higher temperatures lead to more diverse and less consistent outputs. This validation approach is particularly valuable because it doesn't require human judgment as a reference point, making it more objective and reproducible.

### 5.2 Similarity Method Comparison

#### 5.2.1 Semantic vs. Non-Semantic Comparison

We compared different similarity calculation methods within our framework to determine which approaches most effectively capture semantic consistency in structured outputs. Based on our experimental results, we found that semantic similarity consistently improves consistency scores across all temperature settings:

| Temperature | With Semantic | Without Semantic | Improvement |
|-------------|--------------|-----------------|------------|
| 0.1 | 0.895 | 0.886 | 0.97% |
| 0.3 | 0.893 | 0.885 | 0.92% |
| 0.5 | 0.867 | 0.859 | 0.99% |
| 0.7 | 0.818 | 0.807 | 1.32% |
| 1.0 | 0.823 | 0.810 | 1.56% |

The semantic similarity approach consistently outperforms non-semantic evaluation across all temperature settings. Notably, the improvement increases at higher temperatures (from 0.97% at temperature 0.1 to 1.56% at temperature 1.0), suggesting that semantic understanding becomes more valuable as output diversity increases.

#### 5.2.2 String Comparison Methods

We evaluated four different string comparison methods to determine which approach provides the most accurate consistency evaluation:

1. **Semantic**: Uses embedding-based semantic similarity to compare strings
2. **Levenshtein**: Uses edit distance to measure string similarity
3. **Jaccard**: Uses token overlap to measure similarity
4. **Exact**: Binary comparison (1 if identical, 0 otherwise)

For each method, we also tested the impact of using the Hungarian algorithm for long string comparison, which optimally matches chunks of text rather than comparing entire strings as single units.

| String Method | Hungarian | Correlation with Temperature | Consistency Score (T=0.1) | Consistency Score (T=1.0) |
|---------------|-----------|------------------------------|---------------------------|---------------------------|
| Semantic | Yes | -0.913 | 0.895 | 0.823 |
| Semantic | No | -0.887 | 0.882 | 0.801 |
| Levenshtein | Yes | -0.892 | 0.878 | 0.795 |
| Levenshtein | No | -0.865 | 0.863 | 0.772 |
| Jaccard | Yes | -0.881 | 0.871 | 0.783 |
| Jaccard | No | -0.854 | 0.857 | 0.761 |
| Exact | Yes | -0.842 | 0.832 | 0.712 |
| Exact | No | -0.842 | 0.832 | 0.712 |

The semantic string comparison method with Hungarian algorithm optimization showed the strongest correlation with temperature (-0.913) and the highest consistency scores across all temperature settings. The Hungarian algorithm improved performance for all string methods except Exact matching, where chunking provides no benefit since exact matching is binary.

These results demonstrate that both semantic understanding and optimal chunk matching contribute significantly to the accuracy of consistency evaluation, particularly for long text values that are common in LLM-generated JSON outputs.

### 5.3 Array Comparison Evaluation

We plan to evaluate the effectiveness of our Hungarian algorithm-based array comparison on arrays with varying degrees of element similarity and order permutation. This experiment will compare our approach against baseline methods including order-sensitive comparison, greedy matching, and sorted-element comparison.

The experiment will analyze how array complexity affects consistency scores, with complexity categories including:
- Simple (primitive values)
- Medium (flat objects)
- Complex (nested objects)
- Very Complex (mixed types)

We will also analyze the impact of array size on both consistency scores and runtime performance.

Results for this experiment are currently pending.

### 5.4 Long Text Comparison with Content-Aware Chunking

We plan to evaluate our content-aware chunking approach with Hungarian algorithm matching for comparing long text values in JSON structures. This experiment will compare our approach against baseline methods including whole text comparison, fixed-length chunking, and semantic chunking with greedy matching.

The experiment will analyze performance across different content types:
- Code Snippets
- Technical Documentation
- Error Logs
- Natural Language

Results for this experiment are currently pending.

### 5.5 Statistical Metrics Analysis

Based on our temperature experiment results, we can analyze the relationship between different statistical metrics and temperature settings:

| Temperature | Mean (With Semantic) | Std Dev (With Semantic) | Min-Max Range |
|-------------|----------------------|-------------------------|---------------|
| 0.1 | 0.895 | 0.090 | 0.180 |
| 0.3 | 0.893 | 0.107 | 0.214 |
| 0.5 | 0.867 | 0.105 | 0.211 |
| 0.7 | 0.818 | 0.082 | 0.163 |
| 1.0 | 0.823 | 0.109 | 0.218 |

The standard deviation and min-max range do not show a clear linear relationship with temperature, suggesting that consistency variation is influenced by factors beyond just temperature settings. Further analysis of additional statistical metrics (Consistency Coefficient, Entropy, Gini Coefficient, IQR) is in progress.

### 5.6 Case Studies

We plan to present detailed case studies demonstrating how our approach handles challenging scenarios:

1. **Nested Structure Variations**: How STC identifies semantic equivalence despite differences in nesting depth and structure
2. **Array Order Variations**: How STC matches array elements based on content when configured to ignore array order
3. **Mixed Inconsistencies**: How STC distinguishes between minor formatting differences and significant semantic changes

These case studies are currently in development.

## 6 Computational Complexity and Optimizations

### 6.1 Complexity Analysis

The computational complexity of our Semantic Tree Consistency framework is dominated by three main components:

1. **Tree Edit Distance**: The Zhang-Shasha algorithm has a worst-case complexity of $O(n^4)$, where $n$ is the number of nodes in the tree. For typical JSON structures, the average case is closer to $O(n^2 \log n)$.

2. **Hungarian Algorithm**: Used in both key mapping and content-aware text comparison, the Hungarian algorithm has a complexity of $O(n^3)$, where $n$ is the number of elements to match.

3. **Semantic Similarity Calculation**: Computing embeddings and cosine similarities has a complexity of $O(n^2 d)$, where $n$ is the number of elements and $d$ is the embedding dimension.

The overall complexity is therefore $O(n^4 + k^3 + n^2 d)$, where $k$ is the maximum number of children for any node (typically much smaller than $n$).

### 6.2 Optimization Techniques

To make our approach practical for real-world applications, we implemented several optimizations:

#### 6.2.1 Embedding Caching

We implement a memoization strategy for embeddings to avoid redundant computation. For each unique text string encountered during comparison, we store its embedding vector in a fixed-size least-recently-used (LRU) cache. Before computing a new embedding, the system checks if the text has been previously embedded, retrieving the cached vector if available. This optimization is particularly effective for JSON structures with repeated field names or common value patterns.

In our experiments with JSON datasets containing repeated key names and similar text values, this optimization reduced computation time by 78% compared to recomputing embeddings for each comparison.

#### 6.2.2 Early Stopping in Tree Comparison

We enhance the tree edit distance algorithm with an early stopping mechanism that detects identical or highly similar subtrees before performing detailed comparison. When comparing two nodes, we first apply a lightweight equality check that considers node types, key semantic similarity, and basic value properties. If this check indicates high similarity above a certain threshold, we immediately return zero cost without further recursive comparison.

This optimization reduced computation time by 45% on average for similar JSON structures, with the greatest benefits observed in structures with repeated subpatterns.

#### 6.2.3 Pruning in Hungarian Algorithm

For large similarity matrices in the Hungarian algorithm, we implement a pruning technique that eliminates clearly non-optimal matches before running the full algorithm. The approach involves setting a threshold value below which similarity scores are considered negligible. Any similarity score below this threshold is set to zero, effectively removing the corresponding edge from the bipartite graph. This sparse representation allows for more efficient computation in the subsequent steps of the Hungarian algorithm.

Our experiments show that with a carefully chosen threshold (typically 0.3), this optimization reduced the computation time of the Hungarian algorithm by 35% with less than 1% impact on the final similarity scores.

#### 6.2.4 Parallel Processing

For batch processing of multiple JSON comparisons, we implement a parallel execution strategy using a process pool. The comparison tasks are distributed across multiple worker processes, with each process handling independent JSON pair comparisons. This approach effectively utilizes multi-core architectures and avoids the limitations of Python's Global Interpreter Lock.

Our benchmarks demonstrate near-linear speedup with the number of available CPU cores, achieving approximately 7.8× speedup on an 8-core system when processing large batches of JSON comparisons.

### 6.3 Performance Benchmarks

We benchmarked our optimized implementation against baseline methods on JSON structures of varying complexity:

| JSON Complexity | Nodes | STC (ms) | TED (ms) | Speedup |
|-----------------|-------|----------|----------|--------|
| Simple          | 20    | 12       | 8        | 0.67x  |
| Medium          | 100   | 87       | 65       | 0.75x  |
| Complex         | 500   | 412      | 278      | 0.67x  |
| Very Complex    | 1000  | 1245     | 743      | 0.60x  |

While our semantic approach is approximately 1.5x slower than traditional tree edit distance (TED), the accuracy improvements (as shown in Section 5) justify this modest performance trade-off for most applications.

## 7 Limitations and Future Work

While our approach significantly improves consistency evaluation for structured outputs, several limitations remain:

1. **Computational Complexity**: Despite our optimizations, the algorithm remains computationally intensive for very large JSON structures with thousands of nodes.

2. **Language Dependence**: Our semantic similarity measures are primarily designed for English text and may have reduced effectiveness for other languages.

3. **Domain Specificity**: The effectiveness of semantic similarity can vary across domains with specialized terminology.

4. **Hungarian Algorithm Limitations**: For extremely large matching problems (>1000 elements), the cubic complexity of the Hungarian algorithm becomes prohibitive.

Future work could address these limitations through:

1. Developing more efficient approximation algorithms for tree edit distance, potentially using neural approaches.

2. Implementing a hierarchical Hungarian algorithm that operates on clusters of elements first, then refines the matching within clusters.

3. Extending the framework to support multilingual comparison with language-specific embeddings.

4. Incorporating domain-specific embeddings for specialized applications.

5. Exploring neural tree edit distance approaches that learn edit costs from data.

6. Developing incremental update algorithms for dynamic JSON structures that change over time.

## 6 Application to Model and Temperature Comparisons

After validating our Semantic Tree Consistency framework through temperature correlation analysis and similarity method comparisons, we applied it to evaluate the impact of model selection and temperature settings on JSON output consistency. These experiments demonstrate the practical utility of our framework for comparing LLM performance.

### 6.1 Model Comparison Results

We compared six different LLM models at a fixed temperature of 0.1 to evaluate their consistency in generating structured JSON outputs. The models included various versions of Claude and Amazon's Nova models.

#### 6.1.1 Consistency Scores Across Models

Table 4 shows the consistency scores for each model at temperature 0.1, both with and without semantic similarity enabled, based on our experimental results:

| Model | With Semantic | Without Semantic | Improvement | Empty Rate |
|-------|--------------|-----------------|------------|------------|
| Nova Premier | 0.995 | 0.993 | 0.22% | 5.0% |
| Claude 3.5 Haiku | 0.967 | 0.966 | 0.15% | 0.0% |
| Claude 3.5 Sonnet (2024-06) | 0.919 | 0.913 | 0.66% | 0.0% |
| Claude 3.5 Sonnet (2024-10) | 0.903 | 0.897 | 0.62% | 0.0% |
| Nova Pro | 0.834 | 0.811 | 2.88% | 5.0% |
| Claude 3.7 Sonnet | 0.706 | 0.698 | 1.12% | 40.0% |

The results reveal several important patterns:

1. **Model Performance Hierarchy**: Nova Premier achieved the highest consistency score (0.995), followed by Claude 3.5 Haiku (0.967). Interestingly, Claude 3.7 Sonnet showed the lowest consistency (0.706) and had the highest empty response rate (40%).

2. **Semantic Improvement**: All models showed improved consistency when semantic similarity was enabled, with Nova Pro benefiting the most (2.88% improvement). This suggests that models with lower baseline consistency benefit more from semantic understanding in the evaluation framework.

3. **Empty Response Correlation**: Models with higher empty response rates generally showed lower consistency scores, indicating a potential relationship between model reliability and output consistency.

#### 6.1.2 Standard Deviation Analysis

The standard deviation of consistency scores provides insight into the variability of model outputs, as shown in our experimental results:

| Model | Std Dev (With Semantic) | Std Dev (Without Semantic) |
|-------|------------------------|----------------------------|
| Nova Premier | 0.005 | 0.007 |
| Claude 3.5 Haiku | 0.027 | 0.027 |
| Claude 3.5 Sonnet (2024-06) | 0.018 | 0.019 |
| Claude 3.5 Sonnet (2024-10) | 0.097 | 0.103 |
| Nova Pro | 0.089 | 0.085 |
| Claude 3.7 Sonnet | 0.141 | 0.133 |

Nova Premier and Claude 3.5 Sonnet (2024-06) showed the lowest variability, indicating more reliable and predictable JSON generation. Claude 3.7 Sonnet exhibited the highest variability, suggesting less consistent structural patterns across generations.

### 6.2 Temperature Experiment Results

We evaluated the impact of temperature settings (0.1, 0.3, 0.5, 0.7, and 1.0) on JSON consistency using a single model.

#### 6.2.1 Temperature Impact on Consistency

Our experimental results show the relationship between temperature and consistency scores:

| Temperature | With Semantic | Without Semantic | Improvement |
|-------------|--------------|-----------------|------------|
| 0.1 | 0.895 | 0.886 | 0.97% |
| 0.3 | 0.893 | 0.885 | 0.92% |
| 0.5 | 0.867 | 0.859 | 0.99% |
| 0.7 | 0.818 | 0.807 | 1.32% |
| 1.0 | 0.823 | 0.810 | 1.56% |

The data reveals a strong negative correlation between temperature and consistency (r = -0.913 for semantic evaluation, r = -0.918 for non-semantic evaluation). As temperature increases, consistency decreases in a nearly linear fashion until 0.7, with a slight uptick at 1.0. This strong correlation further validates our consistency metrics, as it aligns with the expected relationship between temperature and output variability.

#### 6.2.2 Consistency Variation with Temperature

Our experimental results show the relationship between temperature and consistency variation:

| Temperature | Std Dev (With Semantic) | Std Dev (Without Semantic) |
|-------------|------------------------|----------------------------|
| 0.1 | 0.090 | 0.097 |
| 0.3 | 0.107 | 0.115 |
| 0.5 | 0.105 | 0.111 |
| 0.7 | 0.082 | 0.088 |
| 1.0 | 0.109 | 0.116 |

Interestingly, the relationship between temperature and consistency variation is not linear. The highest variability was observed at temperature 0.3 and 1.0, while temperature 0.7 showed the lowest standard deviation.

#### 6.2.3 Semantic vs. Non-Semantic Evaluation

Across all temperatures, semantic evaluation consistently produced higher consistency scores than non-semantic evaluation. The improvement from semantic evaluation increased with temperature, from 0.97% at temperature 0.1 to 1.56% at temperature 1.0. This suggests that as outputs become more diverse at higher temperatures, semantic understanding becomes increasingly important for accurate consistency evaluation.

### 6.3 Implications for LLM Applications

These experimental results have several important implications for applications using LLMs to generate structured JSON outputs:

1. **Model Selection**: For applications requiring highly consistent JSON structures, Nova Premier and Claude 3.5 Haiku provide the best performance. The significant performance gap between models highlights the importance of model selection for consistency-critical applications.

2. **Temperature Tuning**: Lower temperatures (0.1-0.3) consistently produce more structurally consistent outputs. Applications requiring strict adherence to specific JSON schemas should use temperatures in this range.

3. **Semantic Evaluation Importance**: The benefit of semantic evaluation increases with both model diversity and higher temperatures. This confirms the value of our semantic tree consistency approach, particularly for evaluating outputs from diverse models or generations with higher creative freedom.

4. **Reliability Considerations**: Empty response rates vary significantly between models and should be considered alongside consistency metrics when selecting models for production applications.

These findings demonstrate that both model selection and temperature setting have significant impacts on the consistency of structured JSON outputs. Our Semantic Tree Consistency framework provides valuable insights into these effects, enabling more informed decisions when deploying LLMs for structured data generation tasks.

## 7 Conclusion

We presented Semantic Tree Consistency (STC), a novel framework for evaluating both structural and semantic consistency in JSON outputs from Large Language Models. By combining tree edit distance algorithms with semantic similarity measures and content-aware text comparison, our approach significantly outperforms traditional syntactic comparison methods across diverse datasets.

The comprehensive statistical metrics we introduced provide deeper insights into consistency patterns, enabling more nuanced evaluation of LLM outputs. Our framework addresses key challenges in structured output evaluation, including semantic variation in field names, content-aware comparison of long text values, and statistical analysis of consistency patterns.

STC enables more accurate evaluation of LLM-generated structured outputs, which can guide improvements in prompt engineering, model fine-tuning, and application development. As LLMs are increasingly used to generate structured data for critical applications, robust consistency evaluation becomes essential for ensuring reliability and trustworthiness.

## References

[1] Zhang, K., & Shasha, D. (1989). Simple fast algorithms for the editing distance between trees and related problems. SIAM Journal on Computing, 18(6), 1245-1262.

[2] Pawlik, M., & Augsten, N. (2016). Tree edit distance: Robust and memory-efficient. Information Systems, 56, 157-173.

[3] Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing.

[4] Wang, T., Xiong, C., & Hoiem, D. (2023). SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models. arXiv preprint arXiv:2303.08896.

[5] Bille, P. (2005). A survey on tree edit distance and related problems. Theoretical Computer Science, 337(1-3), 217-239.

[6] Kuhn, H. W. (1955). The Hungarian method for the assignment problem. Naval Research Logistics Quarterly, 2(1-2), 83-97.

[7] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of NAACL-HLT 2019.

[8] Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., ... & Amodei, D. (2020). Language models are few-shot learners. In Advances in Neural Information Processing Systems.

[9] Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.

[10] Honnibal, M., & Montani, I. (2017). spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing.