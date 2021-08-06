import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
class Lab4(object):
    
    def expectation_maximization(self,read_mapping,tr_lengths,n_iterations) :
        #start code here
        N = len(read_mapping)
        K = len(tr_lengths)
        rhos = np.zeros((K,n_iterations+1))
        z = np.zeros((N,K))
        rhos[:,0] = np.ones(K)/K
        for t in range(n_iterations):
            for i in range(N):
                for k in range(K):
                    S_i = read_mapping[i]
                    if k in S_i:
                        z[i][k] = rhos[k][t]/sum(rhos[S_i,t])
                    else:
                        z[i][k] = 0
            theta = np.mean(z,axis=0)
            rhos[:,t+1] = theta/tr_lengths/np.sum(theta/tr_lengths)
        return rhos
        #end code here

    def prepare_data(self,lines_genes) :
        '''
        Input - list of strings where each string corresponds to expression levels of a gene across 3005 cells
        Output - gene expression dataframe
        '''
        #start code here
        dic = {}
        i = 0
        for gene in lines_genes:
            dic[f"Gene_{i}"] = np.round(np.log(np.array(gene.split()).astype("float")+1),5)
            i += 1
        df = pd.DataFrame(dic)
        return df
        #end code here
    
    def identify_less_expressive_genes(self,df) :
        '''
        Input - gene expression dataframe
        Output - list of column names which are expressed in less than 25 cells
        '''
        #start code here
        names = []
        col = df.columns
        n_expressed = np.sum(df.values > 0, axis=0)
        for i in range(len(n_expressed)):
            if n_expressed[i] < 25:
                names.append(col[i])
        return names
        #end code here
    
    
    def perform_pca(self,df) :
        '''
        Input - df_new
        Output - numpy array containing the top 50 principal components of the data.
        '''
        #start code here
        return np.round(PCA(n_components=50,random_state=365).fit_transform(df.values),5)
        #end code here
    
    def perform_tsne(self,pca_data) :
        '''
        Input - pca_data
        Output - numpy array containing the top 2 tsne components of the data.
        '''
        #start code here
        return TSNE(n_components=2,perplexity=50, random_state=1000).fit_transform(pca_data)
        #end code here