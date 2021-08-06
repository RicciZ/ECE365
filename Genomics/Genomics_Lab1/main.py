import numpy as np
from collections import OrderedDict

class Lab1(object):
    def parse_reads_illumina(self,reads) :
        '''
        Input - Illumina reads file as a string
        Output - list of DNA reads
        '''
        #start code here
        reads_split = reads.split("\n")
        res = [reads_split[i] for i in range(len(reads_split)) if i%4==1]
        return res
        #end code here

    def unique_lengths(self,dna_reads) :
        '''
        Input - list of dna reads
        Output - set of counts of reads
        '''
        #start code here
        return set([len(i) for i in dna_reads])
        #end code here

    def check_impurity(self,dna_reads) :
        '''
        Input - list of dna reads
        Output - list of reads which have impurities, a set of impure chars 
        '''
        #start code here
        impure_reads = []
        impure_char = set()
        for i in dna_reads:
            impure = 0
            for j in i:
                if j not in "ACGTacgt":
                    impure_char.add(j)
                    impure = 1
            if impure:
                impure_reads.append(i)
        return impure_reads,impure_char
        #end code here

    def get_read_counts(self,dna_reads) :
        '''
        Input - list of dna reads
        Output - dictionary with key as read and value as the no. of times it occurs
        '''
        #start code here
        res = {}
        for i in dna_reads:
            if i not in res:
                res[i] = 1
            else:
                res[i] += 1
        return res
        #end code here

    def parse_reads_pac(self,reads_pac) :
        '''
        Input - pac bio reads file as a string
        Output - list of dna reads
        '''
        #start code here
        reads_pac_split = reads_pac.split("\n")[1:-1]
        res = []
        read = ''
        for i in reads_pac_split:
            if i[0] == ">":
                res.append(read)
                read = ''
                continue
            read += i
        res.append(read)
        return res
        #end code here