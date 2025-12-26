Federated learning (FL) for Video Action Recognition (VAR) faces significant challenges in balancing privacy preservation, communication efficiency, and model performance.
This paper introduces FLAaMeST Federated Learning for Action Recognition with Multimodal embeddings and Spacio-Temporal Fusion, a FL framework that synergizes Vision-Language Models (VLMs) and spatiotemporal CNNs to address these challenges.
Unlike existing works that use BLIP (VLM) solely for caption generation, $\flamest$ leverages BLIP in a dual manner.
To enhance temporal modeling, complementary spatiotemporal features are extracted using a pre-trained 3D CNN (Slow network). 
These semantic (BLIP) and motion (Slow) embeddings are concatenated into a unified representation to train a lightweight Multi-Layer Perceptron (MLP). 
Within the FL paradigm, only the MLP parameters are shared with the server, ensuring raw video data and generated captions remain local.
$\flamest$ employs the FedAvg algorithm for model aggregation, achieving $99\%(\downarrow)$ lower communication overhead compared to full-model training. 
Experiments on UCF101 and HMDB51 datasets demonstrate the framework’s robustness, achieving improved accuracies of $5.13\%(\uparrow) \text{ and } 2.71\%(\uparrow)$, respectively, against the baseline.
