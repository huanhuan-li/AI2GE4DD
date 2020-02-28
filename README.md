please feel free to add/edit  
# AI2GE4DD
Model for predicting/screening small molecules with ability to achieve specific gene expression profile  

## TODO List
#### **Towards learning disentangled explanatory factors:**   
(2020.03 - 2020.04)    
&emsp;Performance:  
1. different number of dimensions, and biological meaning of each dimension;  
2. robustness, for example, whether different parameter initializations will lead to shuffle of dimension meaning;    

#### **Transfering to novel cell types**  

## Done List  
#### **Towards learning disentangled explanatory factors:**   
(2020.01.19 - 2020.02)  
&emsp;ref: https://github.com/google-research/disentanglement_lib  
1. models: β-VAE, FactorVAE, β-TCVAE, DIP-VAE
2. metrics: BetaVAE score, FactorVAE score, Mutual Information Gap(MIG), DCI disentanglement
3. dataset: L1000 gene expression subset.  

#### **Initial Building of our model**    
Done by Pengcheng.  

## Under Discussion
**Dataset**: to check performance on test set 1) from the same distribution with training set; 2) novel cell type/small molecule;
**loss**: E(logp(yi,xi,zi)|zx,zd) = E(logp(yi,xi,zi|yi^,xi^,zi^)) ~ E(MSE). concat or not before MSE?  => equivalent  
**Evaluation**: metrics design    
*maybe worth trying: Does disentanglement performance correlated with our prediction result?*
