import numpy as np
import scipy.special as sp

import numpy as np

class Decoder:
    def __init__(self, sequences, profhmm):


        self.sequences = []

        with open(sequences, "r") as file:
            current_sequence_id = ""
            current_sequence = ""

            for line in file:
                line = line.strip()

                if line.startswith(">"):
                    # New sequence header
                    if current_sequence_id:
                        self.sequences.append((current_sequence_id, current_sequence))
                    current_sequence_id = line[1:]
                    current_sequence = ""
                else:
                    # Sequence content
                    current_sequence += line

            # Add the last sequence
            if current_sequence_id:
                self.sequences.append((current_sequence_id, current_sequence))

        self.symbols =  ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


        #now the hard part, grab the stuff in the hmm file

        counter_null = 0
        self.tranM = np.array([])
        self.emM = np.array([])
        self.inemM = np.array([])
        counter_states = 0
        with open(profhmm, 'r') as hmm_file:
            
            for line in hmm_file:
                line = line.split() #what we do here is removing all blank spaces, each string surronded by whitespaces will be an element to the list'
                if line == ["//"]: #just in case
                    break 
                if "LENG" in line:
                    self.leng = int(line[1]) #how many match states
                
                if "COMPO" in line: #this is where the info is
                    self.nullem = [float(i) if i != "*" else np.inf for i in line[1:]]

                    counter_null += 1
                    continue
                if  counter_null == 1:
                    self.inemM = [float(i) if i != "*" else np.inf for i in line]
                    self.emM = [float(i) if i != "*" else np.inf for i in line] #removed later, just for matrix initiation
                    counter_states += 1
                    counter_null -= 1
                    continue
                if counter_states == 1:
                    self.tranM = [float(i) if i != "*" else np.inf for i in line]
                    counter_states += 1
                    continue
                if counter_states == 2:
                    temp = [float(i) if i != "*" else np.inf for i in line[1:21]]
                    self.emM = np.vstack([self.emM, temp])
                    counter_states += 1
                    continue
                if counter_states == 3:
                    temp = np.array([float(i) if i != "*" else np.inf for i in line])
                    self.inemM = np.vstack([self.inemM, temp])
                    counter_states += 1
                    continue
                if counter_states == 4:
                    temp = np.array([float(i) if i != "*" else np.inf for i in line])
                    self.tranM = np.vstack([self.tranM, temp])
                    counter_states =2
                    continue
        self.emM = self.emM[1:]

        self.emM = np.exp(-self.emM)
        self.inemM = np.exp(-self.inemM)

        
        np.savetxt('tranM.txt', self.tranM, fmt='%8.6s', delimiter='\t')
        np.savetxt('emM.txt', self.emM, fmt='%8.6s', delimiter='\t')
        np.savetxt('imemM.txt', self.inemM, fmt='%8.6s', delimiter='\t')
        
        #keep in mind, inemM has one extra row, because of the state type INSERT has one extra state, 0. Same goes for the tranM, has one extra row (self.leng + 1)
        #I will remove the first row from tranM because it only contains how transition from M0 and transitions from I0 is and that is not relevant in the plan 7 model
        
        self.amm = np.exp(-self.tranM[:,0] )
        self.ami = np.exp(-self.tranM[:,1] )
        self.amd = np.exp(-self.tranM[:,2] )
        #self.amd[-2] = 0 #This might be used, it is really unclear but some sources states that the Plan7viterbi does NOT use the last delete state, as I understand it, it is enough to remove the first delete state in the core model so each search always matches one residue to match
        self.aim = np.exp(-self.tranM[:,3])
        self.aii = np.exp(-self.tranM[:,4])
        self.adm = np.exp(-self.tranM[:,5])
        self.add = np.exp(-self.tranM[:,6]) #decided to seperate them (mm = match to match etc, a for transition as durbin does)
        #self.add[-2] = 0
        #print(self.add)
        #print(np.log2(self.amd))

        

        self.states = []
        for i in range(1, self.leng+1):
            for j in ["M","D","I"]:
                self.states.append(j + str(i))

        self.states = ["IO"] + self.states
        self.states = ["D0"] + self.states #look at the comment below, in addition this state is actually nonexistant
        self.states = ["START"] + self.states #same as M0
        self.states = self.states + ["END"] #will not be used but needed to releate the index in the dp_matrix row with the state
        #print(self.states)
        self.nullem0 = [0.0787945, 0.0151600, 0.0535222, 0.0668298, 0.0397062, 0.0695071, 0.0229198, 0.0590092, 0.0594422, 0.0963728, 0.0237718, 0.0414386, 0.0482904, 0.0395639, 0.0540978, 0.0683364, 0.0540687, 0.0673417, 0.0114135, 0.0304133]
        #self.nullem0 = [i/100 for i in self.nullem0]


    def viterbi(self):
        
        for id_seq, stuff in self.sequences:

            #print("Alignment of", id_seq)
            seq = list(stuff)
            
            self.symb_index = [self.symbols.index(j) for j in seq] #easier usage later
            

            #now for some transition probabilities, in line with what sean R. EDDY said
            #negative log transforming it all 
            anb = act = ajb = 3/(len(seq)+3) #? #pretty much the same as hmmer if 3 is replaced with 6?????? wtf
            ann = ajj = acc = 1-act

            aej = aec = 0.5
            abm = 2/((self.leng*(self.leng + 1)))
            ame = ade = 1
            nullscore = 0
            arr = len(seq)/(len(seq)+1) #the transition prob for going from R to R in the null model



            dp_m = np.array([[{}]*(self.leng+1+5) for _ in range(len(seq)+1)]) #the last 3 columns are the special states in the order E, J, C and first 2 are N and B
            #initiating
            for seqi in range(dp_m.shape[0]):
                #print(seqi)
                for statej in range(dp_m.shape[1]):
                    dp_m[seqi,statej] = {"log-odds": -np.inf, "prev": None}
            #print(dp_m[:,3])
            for seqi in range(dp_m.shape[0]):
                for statej in range(2, dp_m.shape[1]-3):
                    dp_m[seqi,statej] = {"log-odds M": -np.inf,"log-odds I": -np.inf,"log-odds D": -np.inf, "prev": None}
            
            #Start is from dp[0,N] = 0 (nat log transformed so prob 1)
            dp_m[0,0] = {"log-odds": 0, "prev": None}
            #dp[0,B] = the transistion from N to B
            dp_m[0,1] = {"log-odds": np.log2(anb), "prev": ("N", 0, 0)}
            #print(dp_m[0,3]["log-odds M"])
            for seqi in range(1, dp_m.shape[0]):
                #print(self.symbols[self.symb_index[seqi-1]])
                em_rowind = 0
                inem_rowin = 1
                for statej in range(3, dp_m.shape[1]-3):
                    #alot of overflows.....looking back into durbin et al into chapter 3.6 and using the trick they showed
                    # R = log(q) + log(p) = log(exp(Q) + exp(P))
                    # R = P + log(1 + exp(Q-P))
                    #OH YEAH feels like evil number hacking
                    #switched it with numpys function
                    #m
                    T = np.log2(np.exp2(dp_m[seqi-1,statej-1]["log-odds M"]) * self.amm[em_rowind])
                    Q = np.log2(np.exp2(dp_m[seqi-1,statej-1]["log-odds I"]) * self.aim[em_rowind])
                    S = np.log2(np.exp2(dp_m[seqi-1,statej-1]["log-odds D"]) * self.adm[em_rowind])
                    P = np.log2(np.exp2(dp_m[seqi-1,1]["log-odds"]) * abm)
                    
                    log_tot = np.logaddexp2.reduce([P,Q,T,S])
                    dp_m[seqi,statej]["log-odds M"] = log_tot + np.log2((self.emM[em_rowind, self.symb_index[seqi-1]]/self.nullem0[self.symb_index[seqi-1]]))

                    #i
                    P = np.log2(np.exp2(dp_m[seqi-1,statej]["log-odds M"]) * self.ami[em_rowind])
                    Q = np.log2(np.exp2(dp_m[seqi-1,statej]["log-odds I"]) * self.aii[em_rowind])
                    #temp = np.exp2(dp_m[seqi-1,statej]["log-odds M"]) * self.ami[em_rowind-1] + np.exp2(dp_m[seqi-1,statej]["log-odds I"]) * self.aii[em_rowind-1]
                    PQ = np.logaddexp2(P,Q)        
                    #print(PQ)
                    dp_m[seqi,statej]["log-odds I"] = PQ + np.log2((self.inemM[inem_rowin, self.symb_index[seqi-1]]/self.nullem0[self.symb_index[seqi-1]]))

                    #d
                    P = np.log2(np.exp2(dp_m[seqi,statej-1]["log-odds M"]) * self.amd[em_rowind])
                    Q = np.log2(np.exp2(dp_m[seqi,statej-1]["log-odds D"]) * self.add[em_rowind])
                    #print(P, Q, PQ)
                    PQ = np.logaddexp2(P,Q)
                    dp_m[seqi,statej]["log-odds D"] = PQ 

                    em_rowind += 1
                    inem_rowin += 1
                    
                #e
                holdit = []
                for statej in range(3, dp_m.shape[1]-3):
                    holdit.append(np.log(np.exp2(dp_m[seqi,statej]["log-odds M"]) * ame))
                    holdit.append(np.log(np.exp2(dp_m[seqi,statej]["log-odds D"]) * ade))

                log_tot = sp.logsumexp(holdit)
                log_tot = log_tot/np.log(2)
                #log_tot = np.logaddexp2.reduce(holdit)
                #print(log_tot)

                dp_m[seqi,-3]["log-odds"] = log_tot

                #n
                dp_m[seqi,0]["log-odds"] = dp_m[seqi-1,0]["log-odds"] + np.log2(ann) #np.log2(np.exp2(dp_m[seqi-1,0]["log-odds"]) * ann)

                #j
                P = np.log2(np.exp2(dp_m[seqi,-3]["log-odds"]) * aej)
                Q = np.log2(np.exp2(dp_m[seqi-1,-2]["log-odds"]) * ajj)

                dp_m[seqi,-2]["log-odds"] = np.logaddexp2(P,Q)
                
                #c
                P = np.log2(np.exp2(dp_m[seqi,-3]["log-odds"]) * aec)
                Q = np.log2(np.exp2(dp_m[seqi-1,-1]["log-odds"]) * acc)

                dp_m[seqi,-1]["log-odds"] = np.logaddexp2(P,Q)

                #b
                P = np.log2(np.exp2(dp_m[seqi,0]["log-odds"]) * anb)
                Q = np.log2(np.exp2(dp_m[seqi,-2]["log-odds"]) * ajb)

                dp_m[seqi,1]["log-odds"] = np.logaddexp2(P,Q)

            #print(id_seq, (dp_m[-1,-1]["log-odds"] + np.log2(act) - (np.logaddexp2(-len(seq)*np.log2(arr), np.log2(1-arr)))))
            print(id_seq, (dp_m[-1,-1]["log-odds"] + np.log2(act)) - (len(seq)*np.log2(arr) + np.log2(1-arr)))
            #print(dp_m[-1,-1]["log-odds"] - len(seq)*np.log2(act))

            





                

                
test = Decoder("Testse.txt","globins4.hmm")
test.viterbi()
