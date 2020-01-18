please feel free to add/edit  
# AI2GE4DD
Model for predicting/screening small molecules with ability to achieve specific gene expression profile  

## TODO List
#### **Towards learning disentangled explanatory factors (2020.01.19 - 2020.02):**  
ref: https://github.com/google-research/disentanglement_lib  
1. models: β-VAE, FactorVAE, β-TCVAE, DIP-VAE
2. metrics: BetaVAE score, FactorVAE score, Mutual Information Gap(MIG), DCI disentanglement
3. dataset: L1000 gene expression subset.
4. performance:  
&ensp; - different number of dimensions, and biological meaning of each dimension;  
&ensp; - robustness, for example, whether different parameter initializations will lead to shuffle of dimension meaning;    

#### **Building of our model**  

#### **Transfering to novel cell types**  

## Done List  

## Under Discussion
**Dataset**: to check performance on test set 1) from the same distribution with training set; 2) novel cell type/small molecule;
**Evaluation**: metrics design    
*maybe worth trying: Does disentanglement performance correlated with our prediction result?*
