# Post-hoc Orthogonalization for Mitigation of Protected Feature Bias in CXR Embeddings

Code for the paper Post-hoc Orthogonalization for Mitigation of Protected Feature Bias in CXR Embeddings.

>**Purpose**: To analyze and remove protected feature effects in chest radiograph embeddings of deep learning models.
>**Materials** and Methods: An orthogonalization is utilized to remove the influence of protected features (e.g., age, sex, race) in chest radiograph embeddings, ensuring feature-independent results. To validate the efficacy of the approach, we retrospectively study the MIMIC and CheXpert datasets using three pre-trained models, namely a supervised contrastive, a self-supervised contrastive, and a baseline classifier model. Our statistical analysis involves comparing the original versus the orthogonalized embeddings by estimating protected feature influences and evaluating the ability to predict race, age, or sex using the two types of embeddings.
>**Results**: Our experiments reveal a significant influence of protected features on predictions of pathologies. Applying orthogonalization removes these feature effects. Apart from removing any influence on pathology classification, while maintaining competitive predictive performance, orthogonalized embeddings further make it infeasible to directly predict protected attributes and mitigate subgroup disparities.
>**Conclusion**: The presented work demonstrates the successful application and evaluation of the orthogonalization technique in the domain of chest X-ray classification. 


### A geometric visualization of the orthogonalization method

<p align="center">
<img src=assets/ortho.png />
</p>

The column space of the protected features (col($X$)) contains all the possible vectors that can be formed by taking 
linear combinations of the respective features, i.e. the hypothesis space of a linear model.
For an embedding vector $e \in E$, the orthogonalization is equivalent to the residual between $e$ and its projection 
onto col($X$). With $\mathcal{P}_X^\bot e$ being perpendicular to col($X$), the influence of protected features
in $e$ is neutralized.


## Requirements

Required packages are listed in the `requirements.txt` file, which can be install
e.g. over `pip`:

```shell
pip install -r requirements.txt
```

We use Python 3.11.

## Orthogonalization

The actual orthogonalization procedure is a rather simple procedure.
As a requirement, you need a target matrix $E \in \mathbb{R}^{n \times d}$ that you want to cleanse.
Here, $n$ is the number of samples and $d$ the dimension of the embedding, weight, etc.
The second necessity is the actual feature matrix $X$ with protected characteristics, whose information 
you want to remove from $E$.

In case your protected data is provided in a pandas dataframe including categorical variables, a handy tool to transform
it into a numeric matrix is `dmatrices` from the `patsy` library.
For example, in our case, we construct $X$ as follows:

```python
from patsy import dmatrices

formula = '1 ~ age + sex + race'
_, x_mat = dmatrices(formula, data=x_df)
```

The orthogonalization itself can then be conducted conveniently by our `Orthogonalizator` class.

```python
e_ortho = Orthogonalizator().fit_transform(x_mat, e_org)
```

## Datasets

Our analyses involved the two datasets [MIMIC](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)
and [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/), which can be downloaded under their respective
links after accreditation and signing licenses.
The metadata for CheXpert can be obtained 
[here](https://stanfordaimi.azurewebsites.net/datasets/192ada7c-4d43-466e-b8bb-b81992bb80cf).

## Obtaining Embeddings

We utilize three models to obtain various embeddings that are used in the paper.

#### Chest X-ray Foundation Model

The raw MIMIC embeddings can be 
[downloaded from physionet](https://physionet.org/content/image-embeddings-mimic-cxr/1.0/).
Using the `0_preprocess_mimic_emb.py` in `scripts`, the `.tfrecords` files is parsed to numpy.
Simultaneously, the script serves to preprocess and collect the necessary MIMIC meta data.

#### CheSS

The CheSS model can be downloaded from the [original repository](https://github.com/mi2rl/CheSS).
The responsible script is `1_compute_chess_embs.py`. Adjust the file paths and name accordingly.

#### Chest X-ray Classifier

The basis for our embeddings are the pretrained classifiers from 
[torchxrayvision](https://github.com/mlmed/torchxrayvision).
With the same interface as above, computing the embeddings can be triggered over `2_compute_classifier_embs.py`.

## Notebooks

We supply two notebooks as a measure to reproduce our analysis.
In the first notebook `01_analyze_embeddings.ipynb` the influence of protected characteristics on model predictions is 
estimated and the performance of downstream classifiers is evaluated.
The second notebook `02_predict_protected.ipynb` shows how to derive protected characteristics directly from embeddings
and investigate whether orthogonalization is able to render this task infeasible.

## Reproduction of Tables in Paper

**Table 1:** Make sure the datasets and metadata are downloaded correctly.
For MIMIC, additionally call `scripts/0_preprocess_mimic_emb.py` to trigger the meta preprocessing.
The dataframes containing the MIMIC and CheXpert subsets are then obtained by calling `get_mimic_meta_data(f_path: str)`
and `get_chexpert_meta_data(d_path: str)` in `utils.py` respectively.

**Table 2 & 3:** The coefficents and p-values are obtained by the `eval_classifier` method of the `EmbeddingEvaluator`.
The downstream task classification metrics are computed by `get_classifier_metrics` of the same object.
Both functions and their usage are showcased in `01_analyze_embeddings.ipynb`.

**Table 4:** Execute `02_predict_protected.ipynb` to investigate the prediction of protected features from the embeddings.
