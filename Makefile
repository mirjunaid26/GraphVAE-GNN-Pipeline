# Makefile for GNN Pipeline

.PHONY: all prepare train_gnn

all: prepare train_gnn

prepare:
	python data_preparation.py

train_gnn:
	python train_gnn.py