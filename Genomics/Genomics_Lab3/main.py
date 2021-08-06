import pandas as pd
import statsmodels.api as sm
import numpy as np
import statsmodels

class Lab3(object):
    
    def create_data(self,snp_lines) :
        '''
        Input - the snp_lines parsed at the beginning of the notebook
        Output - You should return the 53 x 3902 dataframe
        '''
        #start code here
        data = np.zeros((53,3902))
        header = []
        for i in range(len(snp_lines)):
            line = snp_lines[i].replace('\n','').split('\t')
            header.append(line[0]+':'+line[1])
            for dogi in range(9,len(line)):
                xy = line[dogi].split('/')
                if xy[0] != '.' and xy[1] != '.':
                    data[dogi-9][i] = int(xy[0])+int(xy[1])
                elif xy[0] == '.' and xy[1] != '.':
                    data[dogi-9][i] = int(xy[1])
                elif xy[0] != '.' and xy[1] == '.':
                    data[dogi-9][i] = int(xy[0])
                else:
                    data[dogi-9][i] = np.nan
        df = pd.DataFrame(data,columns=header)
        return df
        #end code here

    def create_target(self,header_line) :
        '''
        Input - the header_line parsed at the beginning of the notebook
        Output - a list of values(either 0 or 1)
        '''
        #start code here
        pheno = header_line.replace('\n','').split('\t')[9:]
        res = []
        for i in pheno:
            if 'dark' in i:
                res.append(0)
            else:
                res.append(1)
        return res
        #end code here
    
    def logistic_reg_per_snp(self,df) :
        '''
        Input - snp_data dataframe
        Output - list of pvalues and list of betavalues
        '''
        #start code here
        pvalues = []
        betavalues = []
        for i in range(df.shape[1]-1):
            x = sm.add_constant(df.iloc[:,i],False)
            res = sm.Logit(df['target'],x,missing='drop').fit(method='bfgs',disp=False)
            pvalues.append(round(res.pvalues[0],9))
            betavalues.append(round(res.params[0],5))
        return pvalues,betavalues
        #end code here
    
    
    def get_top_snps(self,snp_data,p_values) :
        '''
        Input - snp dataframe with target column and p_values calculated previously
        Output - list of 5 tuples, each with chromosome and position
        '''
        #start code here
        res = []
        tmp = snp_data.columns[np.argpartition(p_values,5)[:5]]
        for i in tmp:
            chro,pos = i.split(':')
            res.append((chro,pos))
        return res
        #end code here