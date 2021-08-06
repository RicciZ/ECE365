import numpy as np
from collections import OrderedDict

class Lab2(object):
    
    def smith_waterman_alignment(self,s1,s2,penalties) :
        '''
        Input - two sequences and a dictionary with penalities for match, mismatch and gap
        Output - an integer value which is the maximum smith waterman alignment score
        '''
        #start code here
        l1 = len(s1)
        l2 = len(s2)
        H = np.zeros((l1+1,l2+1),dtype=int)
        max_score = 0
        for i in range(1,l1+1):
            for j in range(1,l2+1):
                if s1[i-1] == s2[j-1]:
                    m = H[i-1][j-1] + penalties['match']
                    s = 0
                else:
                    m = 0
                    s = H[i-1][j-1] + penalties['mismatch']
                H[i][j] = max(m,s,H[i-1][j]+penalties['gap'],H[i][j-1]+penalties['gap'])
                if H[i][j] > max_score:
                    max_score = H[i][j]
        return max_score
        #end code here

    def print_smith_waterman_alignment(self,s1,s2,penalties) :
        '''
        Input - two sequences and a dictionary with penalities for match, mismatch and gap
        Output - a tuple with two strings showing the two sequences with '-' representing the gaps
        '''
        #start code here
        l1 = len(s1)
        l2 = len(s2)
        H = np.zeros((l1+1,l2+1),dtype=int)
        pos = np.zeros((l1+1,l2+1),dtype=int)
        max_score = 0
        max_pos = (0,0)

        up = 1
        left = 2
        upleft = 3

        # build the dp map
        for i in range(1,l1+1):
            for j in range(1,l2+1):
                if s1[i-1] == s2[j-1]:
                    d = H[i-1][j-1] + penalties['match']
                else:
                    d = H[i-1][j-1] + penalties['mismatch']
                score = 0
                if d > score:
                    score = d
                    pos[i][j] = upleft
                if H[i-1][j]+penalties['gap'] > score:
                    score = H[i-1][j]+penalties['gap']
                    pos[i][j] = up
                if H[i][j-1]+penalties['gap'] > score:
                    score = H[i][j-1]+penalties['gap']
                    pos[i][j] = left
                H[i][j] = score
                if H[i][j] > max_score:
                    max_score = H[i][j]
                    max_pos = (i,j)

        # find the sequences
        (i,j) = max_pos
        res1 = ''
        res2 = ''
        while 1:
            if pos[i][j] == up:
                res1 = s1[i-1] + res1
                res2 = '-' + res2
                i -= 1
            elif pos[i][j] == left:
                res1 = '-' + res1
                res2 = s2[j-1] + res2
                j -= 1
            elif pos[i][j] == upleft:
                res1 = s1[i-1] + res1
                res2 = s2[j-1] + res2
                i -= 1
                j -= 1
            else:
                break
        return res1, res2
        #end code here

    def find_exact_matches(self,list_of_reads,genome):
        
        '''
        Input - list of reads of the same length and a genome fasta file (converted into a single string)
        Output - a list with the same length as list_of_reads, where the ith element is a list of all locations (starting positions) in the genome where the ith read appears. The starting positions should be specified using the "chr2:120000" format
        '''
        #start code here
        res = []
        if len(list_of_reads) < 1:
            return res
        reads_len = len(list_of_reads[0])
        genome_dict = {}
        

        # build the list
        i = 0
        temp = genome.split('\n')
        genome_list = []
        while 1:
            if temp[i] == '':
                genome_list.append(temp[i])
                break
            if temp[i][0] == '>':
                genome_list.append(temp[i])
                i += 1
                continue
            else:
                if temp[i-1][0] == '>':
                    genome_list.append(temp[i])
                    i += 1
                    continue
                else:
                    genome_list[-1] += temp[i]
                    i += 1
                    continue

        # build the dict
        i = 0
        while (genome_list[i] != ''):
            chro = genome_list[i][1:]
            bases = genome_list[i+1]
            genome_dict[chro] = {}
            for j in range(0,len(bases)-reads_len+1):
                if bases[j:j+reads_len] not in genome_dict[chro]:
                    genome_dict[chro][bases[j:j+reads_len]] = [j]
                else:
                    genome_dict[chro][bases[j:j+reads_len]].append(j)
            i += 2

        # search for the reads
        for i in list_of_reads:
            find_one_read = []
            for j in genome_dict:
                if i in genome_dict[j]:
                    for k in genome_dict[j][i]:
                        find_one_read.append(j+':'+str(k+1))
            res.append(find_one_read)
        return res
        #end code here
       
    
    def find_approximate_matches(self,list_of_reads,genome):
        '''
        Input - list of reads of the same length and a genome fasta file (converted into a single string)
        Output -  a list with the same length as list_of_reads, where the ith element is a list of all locations (starting positions) in the genome which have the highest smith waterman alignment score with ith read in list_of_reads
        '''
        
        #start code here
        res = []
        if len(list_of_reads) < 1:
            return res
        L = len(list_of_reads[0])
        k = L // 4
        genome_dict = {}
        penalties = {'match':1,'mismatch':-1,'gap':-1}

        # build the list
        i = 0
        temp = genome.split('\n')
        genome_list = []
        while 1:
            if temp[i] == '':
                genome_list.append(temp[i])
                break
            if temp[i][0] == '>':
                genome_list.append(temp[i])
                i += 1
                continue
            else:
                if temp[i-1][0] == '>':
                    genome_list.append(temp[i])
                    i += 1
                    continue
                else:
                    genome_list[-1] += temp[i]
                    i += 1
                    continue

        # build the dict
        i = 0
        while (genome_list[i] != ''):
            chro = genome_list[i][1:]
            bases = genome_list[i+1]
            genome_dict[chro] = {}
            for j in range(0,len(bases)-k+1):
                if bases[j:j+k] not in genome_dict[chro]:
                    genome_dict[chro][bases[j:j+k]] = [j]
                else:
                    genome_dict[chro][bases[j:j+k]].append(j)
            i += 2
        
        # find the matches
        for X in list_of_reads:
            max_score = 0
            max_pos = []
            for i_X in range(0,L-k+1):
                sub_X = X[i_X:i_X+k]
                for chro in genome_dict:
                    if sub_X in genome_dict[chro]:
                        for i_chr in genome_dict[chro][sub_X]:
                            # build Y
                            chr_sequence = genome_list[int(chro.replace('chr',''))*2-1]
                            if len(chr_sequence) < L:
                                continue
                            left = max(i_chr-i_X,0)
                            right = min(i_chr+L-i_X,len(chr_sequence))
                            Y = chr_sequence[left:right]
                            # compute score and check if it is the best
                            score = self.smith_waterman_alignment(X,Y,penalties)
                            pos = chro+':'+str(left+1)
                            if score > max_score:
                                max_score = score
                                max_pos = [pos]
                            elif score == max_score:
                                if pos not in max_pos:
                                    max_pos.append(pos)
            res.append(max_pos)
        
        return res
        #end code here
