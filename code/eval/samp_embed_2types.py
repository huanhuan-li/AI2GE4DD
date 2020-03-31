# coding=UTF-8
'''
    TODO:
        GAN的生成样本是否与测试集中样本聚为一类;
    data:
        1) Generated samples from GAN (900); 2) Test set (900); 3) original data (130000) 
        2) scanpy (for single cell data) clustering and embedding
'''
import os
import json
import argparse
import numpy as np
import pandas as pd
import scipy.sparse
import scanpy.api as sc
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='GAN generated samples embedding.')
parser.add_argument('-path_px', type=str, help='path to GAN generated file')
parser.add_argument('-file_px', type=str, help='GAN generated file, npy format')
parser.add_argument('-path_test', type=str, help='path to test set')
parser.add_argument('-file_test', type=str, help='test file')
parser.add_argument('-path_save', type=str, default='', help='save path')
parser.add_argument('-HVG_MODE', action='store_true', default=False, help='whether to use highly variable genes.')
parser.add_argument('-HP_FLAG', action='store_true', default=False, help='whether to plot heatmap.')
parser.add_argument('-n_pcs', type=int, default=500, help='pcs used for clustering etc.')
parser.add_argument('-hvg_pcs', type=int, default=200, help='pcs used for clustering etc.')
parser.add_argument('-o', type=str, default='', help='output name')
args = parser.parse_args()
sys.setrecursionlimit(100000)

sc.settings.autosave=True
sc.settings.autoshow=False
sc.settings.figdir = args.path_save
sc.settings.verbosity=0

'''read in data'''
def dataload(path, filename):
    # return: nparray
    if filename.split('.')[1]=='npy':
        data = np.load(os.path.join(path, filename))
    elif filename.split('.')[1]=='npz':
        data = scipy.sparse.load_npz(os.path.join(path, filename))
        data = data.toarray()
    else:
        print("TypeError: Invalid file type, pls check!")
    return data
    
def get_genenames(path, filename):
    # return list
    with open(os.path.join(path, filename), 'r') as f:
        gene_idx = json.load(f)
    gene_idx_ = zip(gene_idx.values(), gene_idx.keys())
    gene_idx = sorted(gene_idx_)
    gene_list_byorder = [g[1] for g in gene_idx]
    return gene_list_byorder

# GENE NAMES by order
colnames = get_genenames('/home/hli/vae2dd/data/Microwell_MCA/', 'gene_idx.json')

# gen data
gen_data = dataload(args.path_px, args.file_px)

N, xdim = gen_data.shape
gen_adata = sc.AnnData(X = gen_data, var = {}, obs = {})
gen_adata.var.index = colnames[0:xdim]
gen_adata.obs['sample'] = 'gen_'+gen_adata.obs.index
gen_adata.obs['batch_ids']='Generated'

# test data
test_data = dataload(args.path_test, args.file_test)
test_data = test_data[:,0:xdim]
test_adata = sc.AnnData(X = test_data, var = {}, obs = {})
test_adata.var.index = colnames[0:xdim]
test_adata.obs['sample'] = 'test_'+test_adata.obs.index
test_adata.obs['batch_ids']='Test_set'

# concatenate
concat_adata = test_adata.concatenate(gen_adata) #org_data
print("Shape of gen data:{}".format(gen_adata.shape))
print("Shape of real data:{}".format(test_adata.shape))
print("Shape of concat data:{}".format(concat_adata.shape))
concat_adata.obs['total_counts']=concat_adata.X.sum(axis=1)

if args.HVG_MODE:
    adata_hvg = concat_adata
    sc.pp.filter_genes(adata_hvg, min_cells=50)
    print("Shape of concat data after genes filter:{}".format(adata_hvg.shape))
    filter_result = sc.pp.filter_genes_dispersion(adata_hvg.X, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata_hvg = adata_hvg[:, filter_result.gene_subset]
    print("Shape of concat data:{}".format(adata_hvg.shape))

'''clustering and plot'''
# heatmap
if args.HVG_MODE:
    sub = adata_hvg[0:2*N:2,:]
else:
    sub = concat_adata[0:2*N:2,:]
if args.HP_FLAG:
    clust = sc.pl.clustermap(sub, vmin=0, vmax=1, cmap='Reds')
    sub = sub[:, clust.data2d.columns]
    sc.pl.heatmap(sub, 
                var_names=sub.var.index, 
                groupby='batch_ids', 
                save='_{}'.format(args.o), 
                vmin=0, vmax=0.7, 
                cmap='viridis', 
                figsize=(6,3))

# clustering
print('pca...')
PCA_COMS=args.n_pcs
if args.HVG_MODE:
    sc.tl.pca(adata_hvg, n_comps=args.hvg_pcs, svd_solver='arpack', use_highly_variable=None)
    print ("HVG_PCA variance_ratio for {} dims: {}".format(args.hvg_pcs, adata_hvg.uns['pca']['variance_ratio'].sum()))
sc.tl.pca(concat_adata, n_comps=PCA_COMS, svd_solver='arpack', use_highly_variable=None)
print ("ALL_PCA variance_ratio for {} dims: {}".format(PCA_COMS, concat_adata.uns['pca']['variance_ratio'].sum()))

print('clustering...')
if args.HVG_MODE:
    sc.pp.neighbors(adata_hvg, n_pcs=args.hvg_pcs, method='umap') # method : {‘umap’, ‘gauss’}
    sc.tl.louvain(adata_hvg)
sc.pp.neighbors(concat_adata, n_pcs=PCA_COMS, method='umap') # method : {‘umap’, ‘gauss’}
sc.tl.louvain(concat_adata)

print('embedding...')
if args.HVG_MODE:
    sc.tl.umap(adata_hvg)
    sc.pl.umap(adata_hvg, color=['louvain', 'batch_ids'], title=['louvain', '{}_hvg_pca_{}_variancce'.format(args.o, adata_hvg.uns['pca']['variance_ratio'].sum())], ncols=2, save='_genX_{}'.format(args.o))
sc.tl.umap(concat_adata)
sc.pl.umap(concat_adata, color=['louvain', 'batch_ids'], title=['louvain', '{}_all_pca_{}_variancce'.format(args.o, concat_adata.uns['pca']['variance_ratio'].sum())], ncols=2, save='_genX_{}'.format(args.o))
