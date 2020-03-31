please feel free to add/edit  
# AI2GE4DD
Model for predicting/screening small molecules with ability to achieve specific gene expression profile  

## TODO List
#### **Towards learning disentangled explanatory factors:**   
(2020.04 - 2020.05)    
&emsp;Performance:  
1. model selection, and biological meaning interperation;    

#### **Transfering to novel cell types**  

## Done List  
#### **Towards learning disentangled explanatory factors:**   
(2020.01.19 - 2020.02)  
&emsp;ref: https://github.com/google-research/disentanglement_lib  
1. models: β-VAE, FactorVAE, β-TCVAE, DIP-VAE
2. metrics: BetaVAE score, FactorVAE score, Mutual Information Gap(MIG), DCI disentanglement
3. dataset: L1000 gene expression subset.  

(2020.02 - 2020.03)  
&emsp;model conducted: beta-VAE, AnnealedVAE, beta-TCVAE, FactorVAE, DIP-VAE;  
&emsp;metrics used:  
1. Independence between the latent variables - total correlation/correlation between variables;  
2. capacity of latent variable - KL(q(z_i|x)||p(z_i));  
3. Mutual information between the latent variables and the data variable;  

#### **Initial Building of our model**    
Done by Pengcheng.  

## Under Discussion
**Dataset**: to check performance on test set 1) from the same distribution with training set; 2) novel cell type/small molecule;
**loss**: E(logp(yi,xi,zi)|zx,zd) = E(logp(yi,xi,zi|yi^,xi^,zi^)) ~ E(MSE). concat or not before MSE?  => equivalent  
**Evaluation**: metrics design    
*maybe worth trying: Does disentanglement performance correlated with our prediction result?*
