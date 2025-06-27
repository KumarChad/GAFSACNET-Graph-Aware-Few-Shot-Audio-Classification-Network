# 📄 GAFSACNet

## 💻 Introduction

**GAFSACNet** (Graph-Aware Few-Shot Audio Classification Network) is an ongoing deep learning research project for multi-label and few-shot audio tagging. It combines a pretrained ResNet for feature extraction to generate embeddings, Graph Attention Networks (GAT) for hierarchical modeling of the embedding space, and a modified Prototypical Network for classification. The model is trained on Mel spectrograms extracted from a subset of the FSD50K dataset.

## 📖 Table of Contents

- [💻 Introduction](#introduction)
- [🔥 Features](#features)
- [👥 Contributors](#contributors)

## ♣️ Features

- **Hybrid and Novel Architecture:**: Combines ResNet, GAT, and Prototypical Networks for robust multi-label and few-shot audio tagging.

- **Data Augmentation:**: Employs augmentation techniques such as MixUp and SpecAugment on Mel spectrograms to enhance generalization of the model.

- **Scalable Data Pipeline:**: Includes a scalable and efficient custom dataset implementation for handling large scale data and the episodic training needs of the model.

## 👥 Contributors

* [Anirudh Vignesh](https://github.com/crystallyen)
* [Divyesh Dileep](https://github.com/Divyesh48960)
* [Kumaradithya Chadalavada](https://github.com/KumarChad)