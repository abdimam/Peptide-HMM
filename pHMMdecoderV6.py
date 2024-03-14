import numpy as np
import scipy.special as sp
import math

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
        self.amd[-2] = 0 #This might be used, it is really unclear but some sources states that the Plan7viterbi does NOT use the last delete state, as I understand it, it is enough to remove the first delete state in the core model so each search always matches one residue to match
        self.aim = np.exp(-self.tranM[:,3])
        self.aii = np.exp(-self.tranM[:,4])
        self.adm = np.exp(-self.tranM[:,5])
        self.add = np.exp(-self.tranM[:,6]) #decided to seperate them (mm = match to match etc, a for transition as durbin does)
        self.add[-2] = 0
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


    def forward(self):

        
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
  



    def inverse(self, alg_base="forward", inverse_mode = False):
        if alg_base not in ["forward", "viterbi"]:
            raise ValueError("Error: Not a valid algorithm")
        if inverse_mode not in [True, False]:
            raise SyntaxError("Error: Please give a valid boolean value for inverse_mode")
        #elif inverse_mode == True:
        #    ir = 0.12/len(seq) #inversion ratio, the fraction of transitions that should have gone to M that instead goes to MM
                   #this value is based on the frequency of getting a incorrect peptide multiplied with the fraction of the errors that are classifed as an inversion error of 2 residues
                   #We assume that the probability of such inversion to occour in a peptide is UNIFORM across the whole peptide, so the value is divided by the length of the peptide!
                   #Another assumtion would be that we expect only one inversion in a peptide (if it is present) because of how previous research have hinted that such errors typically only happens once
                    #ir = 0.12/len(seq) obs: the error type rate seems to differs greatly between what algorithm where used to generete the peptides so this is a really rough assumtion of eyeballing the average (pepnovo seems to make this kind of error for every 6 peptide genereted....)
        #else:
        #    ir = 0
        #iir = 1-ir #inverted inversion ratio, the fraction that continues to go to M instead of MM
            
        
        
        for id_seq, stuff in self.sequences:
                #print("Alignment of", id_seq)
                seq = list(stuff)
                
                self.symb_index = [self.symbols.index(j) for j in seq] #easier usage later
                

                #now for some transition probabilities, in line with what sean R. EDDY said
                anb = act = ajb = 3/(len(seq)+3) #? #pretty much the same as hmmer if 3 is replaced with 6?????? wtf
                ann = ajj = acc = 1-act

                aej = aec = 0.5
                abm = 2/((self.leng*(self.leng + 1)))
                ame = ade = 1
                nullscore = 0
                arr = len(seq)/(len(seq)+1) #the transition prob for going from R to R in the null model


                #some comments before implementing the MM state
                #MM state in DP programming requires to look back 2 steps in both row and col, to not get an index error we extand the row (seq) and col (states), these will be nonexistant Seq(-1) state(-1)
                #MM1 will be seen as the same "level" as M1 and same for MM2 and M2, however the transition probabilities from MM(k) will be the same as for M(k+1)
                #the transition from other states into MM are the same as into M but multiplied with ir

                dp_m = np.array([[{}]*(self.leng+2+5) for _ in range(len(seq)+2)]) #the last 3 columns are the special states in the order E, J, C and first 2 are N and B, the 2 columns efter B are nonexistant states and the same 
                #initiating
                for seqi in range(dp_m.shape[0]):
                    #print(seqi)
                    for statej in range(dp_m.shape[1]):
                        dp_m[seqi,statej] = {"log-odds": -np.inf, "prev": None}
                #print(dp_m[:,3])
                for seqi in range(dp_m.shape[0]):
                    for statej in range(2, dp_m.shape[1]-3):
                        dp_m[seqi,statej] = {"log-odds M": -np.inf,"log-odds I": -np.inf,"log-odds D": -np.inf, "prev": None, "log-odds /MM/": -np.inf, "prev": None}
                
                #Start is from dp[0,N] = 0 (nat log transformed so prob 1)
                dp_m[1,0] = {"log-odds": 0, "prev": None}
                #dp[0,B] = the transistion from N to B
                dp_m[1,1] = {"log-odds": np.log2(anb), "prev": ("N", [0,0])}
                #print(dp_m[0,3]["log-odds M"])
                for seqi in range(2, dp_m.shape[0]):
                    #print(self.symbols[self.symb_index[seqi-1]])
                    em_rowind = 0
                    inem_rowin = 1
                    for statej in range(4, dp_m.shape[1]-3):
                        
                        #because of how MM state cant exist when there is only one residue left OR when we are evaluating the last state
                        if inverse_mode == True:
                            if seqi == dp_m.shape[0]-1 or statej == dp_m.shape[1]-4:
                                ir = 0
                                iir = 1-ir
                            else:
                                ir = 0.02564102564102564
                                                             #disregard the wall of comment above, the new value is based on the frequency of WRONG residue times the fraction of the errors that are within this type
                                        
                                iir = 1-ir 
                        else:
                            ir = 0
                            iir = 1-ir

                        #m
                        T = np.log2(np.exp2(dp_m[seqi-1,statej-1]["log-odds M"]) * self.amm[em_rowind]  * iir)
                        Q = np.log2(np.exp2(dp_m[seqi-1,statej-1]["log-odds I"]) * self.aim[em_rowind]  * iir)
                        S = np.log2(np.exp2(dp_m[seqi-1,statej-1]["log-odds D"]) * self.adm[em_rowind]  * iir)
                        P = np.log2(np.exp2(dp_m[seqi-1,1]["log-odds"]) * abm * iir)
                        R = np.log2(np.exp2(dp_m[seqi-2,statej-2]["log-odds /MM/"]) * self.amm[em_rowind]  * iir) 
                        if alg_base == "forward":
                            log_tot = np.logaddexp2.reduce([P,Q,T,S,R])
                        elif alg_base == "viterbi":
                            log_tot, prev_ind = np.max([T,Q,S,P]), np.argmax([T,Q,S,P,R])
                            
                            prev_state = ["M","I","D","B","/MM/"][prev_ind]
                            #print(prev_state)
                            prev_pos = [[seqi-1,statej-1], [seqi-1,statej-1], [seqi-1,statej-1], [seqi-1,1], [seqi-2,statej-2]][prev_ind]
                            dp_m[seqi,statej]["prev M"] = (prev_state, prev_pos)
                            #print(dp_m[seqi,statej]["prev"])
                        dp_m[seqi,statej]["log-odds M"] = log_tot + np.log2((self.emM[em_rowind, self.symb_index[seqi-2]]/self.nullem0[self.symb_index[seqi-2]]))

#kolla med käll, osäker om jag borde applicera irr här eller om det räcker med att transition probs in till MM och M är behandlade och därmed blir summan av alla probs till 1 fortfarande
#uses the transition prob of the outer MM (that is, if it is MM1 is it a combination of M1 and M2, M2 is the other one)

                        #i
                        P = np.log2(np.exp2(dp_m[seqi-1,statej]["log-odds M"]) * self.ami[em_rowind])
                        Q = np.log2(np.exp2(dp_m[seqi-1,statej]["log-odds I"]) * self.aii[em_rowind])
                        R = np.log2(np.exp2(dp_m[seqi-1,statej-1]["log-odds /MM/"]) * self.ami[em_rowind]) #This is because Ik depends on the previous MM(k-1) state
                        #temp = np.exp2(dp_m[seqi-1,statej]["log-odds M"]) * self.ami[em_rowind-1] + np.exp2(dp_m[seqi-1,statej]["log-odds I"]) * self.aii[em_rowind-1]
                        if alg_base == "forward":    
                            PQ = np.logaddexp2.reduce([P,Q,R])
                        elif alg_base == "viterbi":
                            PQ, prev_ind = np.max([P,Q,R]), np.argmax([P,Q,R])
                            prev_state = ["M","I","/MM/"][prev_ind]
                            prev_pos = [[seqi-1,statej], [seqi-1,statej],[seqi-1,statej-1]][prev_ind]
                            dp_m[seqi,statej]["prev I"] = (prev_state, prev_pos)

                        #print(PQ)
                        dp_m[seqi,statej]["log-odds I"] = PQ + np.log2((self.inemM[inem_rowin, self.symb_index[seqi-2]]/self.nullem0[self.symb_index[seqi-2]]))

                        #d
                        P = np.log2(np.exp2(dp_m[seqi,statej-1]["log-odds M"]) * self.amd[em_rowind])
                        Q = np.log2(np.exp2(dp_m[seqi,statej-1]["log-odds D"]) * self.add[em_rowind])
                        R = np.log2(np.exp2(dp_m[seqi,statej-2]["log-odds /MM/"]) * self.amd[em_rowind]) #Easier to understand if one look at the graph diagram but transitions to delete from MM is a jump of 2 states (k) in the diagram
                        if alg_base == "forward":    
                            PQ = np.logaddexp2.reduce([P,Q,R])
                        elif alg_base == "viterbi":
                            PQ, prev_ind = np.max([P,Q,R]), np.argmax([P,Q,R])
                            prev_state = ["M","D","/MM/"][prev_ind]
                            prev_pos = [[seqi,statej-1], [seqi,statej-1],[seqi,statej-2]][prev_ind]
                            dp_m[seqi,statej]["prev D"] = (prev_state, prev_pos)
                        dp_m[seqi,statej]["log-odds D"] = PQ 


                        #mm, 
                        T = np.log2(np.exp2(dp_m[seqi-1,statej-1]["log-odds M"]) * self.amm[em_rowind] * ir)
                        Q = np.log2(np.exp2(dp_m[seqi-1,statej-1]["log-odds I"]) * self.aim[em_rowind] * ir)
                        S = np.log2(np.exp2(dp_m[seqi-1,statej-1]["log-odds D"]) * self.adm[em_rowind] * ir)
                        P = np.log2(np.exp2(dp_m[seqi-1,1]["log-odds"]) * abm * ir)
                        R = np.log2(np.exp2(dp_m[seqi-2,statej-2]["log-odds /MM/"]) * self.amm[em_rowind] * ir) 
                        if alg_base == "forward":
                            log_tot = np.logaddexp2.reduce([P,Q,T,S,R])
                        elif alg_base == "viterbi":
                            log_tot, prev_ind = np.max([T,Q,S,P]), np.argmax([T,Q,S,P,R])
                            
                            prev_state = ["M","I","D","B","/MM/"][prev_ind]
                            #print(prev_state)
                            prev_pos = [[seqi-1,statej-1], [seqi-1,statej-1], [seqi-1,statej-1], [seqi-1,1], [seqi-2,statej-2]][prev_ind]
                            dp_m[seqi,statej]["prev /MM/"] = (prev_state, prev_pos)
                            #print(dp_m[seqi,statej]["prev"])

                        #the emission must account for the average emission of the first M emitts residue i and the second M emitts residue i+1 with the reverse event
                        #index error will happen if this is evaluated at the last state k and last residue, normally we expect the emission score to be 0 in such cases so we make the following condition
                        if seqi == dp_m.shape[0]-1 or statej == dp_m.shape[1]-4:
                            dp_m[seqi,statej]["log-odds /MM/"] = log_tot #+ -np.inf
                        else:
                            dp_m[seqi,statej]["log-odds /MM/"] = log_tot + (np.log2(self.emM[em_rowind, self.symb_index[seqi-2]]/self.nullem0[self.symb_index[seqi-2]] 
                                                                                * self.emM[em_rowind+1, self.symb_index[seqi-1]]/self.nullem0[self.symb_index[seqi-1]] #the joined probability of M(k) emits x(i) and M(k+1) emits x(i+1)
                                                                                 + self.emM[em_rowind, self.symb_index[seqi-1]]/self.nullem0[self.symb_index[seqi-1]]  # addition of these two events, addition because these are alternative events so addition is the right operation
                                                                                * self.emM[em_rowind+1, self.symb_index[seqi-2]]/self.nullem0[self.symb_index[seqi-2]] #the joined probability of M(k) emits x(i+1) and M(k+1) emits x(i)
                                                                                ) - np.log2(2)) #division by two to average out these events!
                            




                        em_rowind += 1
                        inem_rowin += 1 #hej 
                        
                    #e
                    holdit = []
                    state_remeber = []
                    stateind_remember = []
                    for statej in range(4, dp_m.shape[1]-3):
                        holdit.append(np.log(np.exp2(dp_m[seqi,statej]["log-odds M"]) * ame))
                        holdit.append(np.log(np.exp2(dp_m[seqi,statej]["log-odds D"]) * ade))
                        holdit.append(np.log(np.exp2(dp_m[seqi,statej]["log-odds /MM/"]) * ame)) #assuming that we may end at MM
                        state_remeber += ["M", "D", "/MM/"]
                        stateind_remember += [statej, statej, statej]
                    if alg_base == "forward":
                        log_tot = sp.logsumexp(holdit)
                    elif alg_base == "viterbi":
                        log_tot, prev_ind = np.max(holdit), np.argmax(holdit)
                        prev_state = state_remeber[prev_ind]
                        prev_pos = stateind_remember[prev_ind]
                        #print(prev_pos)
                        #print(seqi)
                        #prev_pos = np.where(dp_m["log-odds M"] == log_tot)
                        #print(prev_pos)
                        #prev_pos = [index.tolist() for index in prev_pos]
                        dp_m[seqi,-3]["prev"] = (prev_state, [seqi,prev_pos])
                    log_tot = log_tot/np.log(2)
                    #log_tot = np.logaddexp2.reduce(holdit)
                    #print(log_tot)

                    dp_m[seqi,-3]["log-odds"] = log_tot

                    #n
                    dp_m[seqi,0]["log-odds"] = dp_m[seqi-1,0]["log-odds"] + np.log2(ann) #np.log2(np.exp2(dp_m[seqi-1,0]["log-odds"]) * ann)
                    dp_m[seqi,0]["prev"] = ("N", [seqi-1,0])

                    #j
                    P = np.log2(np.exp2(dp_m[seqi,-3]["log-odds"]) * aej)
                    Q = np.log2(np.exp2(dp_m[seqi-1,-2]["log-odds"]) * ajj)
                    if alg_base == "forward":
                        PQ = np.logaddexp2(P,Q)
                    elif alg_base == "viterbi":
                        PQ, prev_ind = np.max([P,Q]), np.argmax([P,Q])
                        prev_state = ["E", "J"][prev_ind]
                        prev_pos = [[seqi,-3], [seqi-1,-2]][prev_ind]
                        dp_m[seqi,-2]["prev"] = (prev_state, prev_pos)
                    dp_m[seqi,-2]["log-odds"] = PQ
                    
                    #c
                    P = np.log2(np.exp2(dp_m[seqi,-3]["log-odds"]) * aec)
                    Q = np.log2(np.exp2(dp_m[seqi-1,-1]["log-odds"]) * acc)
                    if alg_base == "forward":
                        PQ = np.logaddexp2(P,Q)
                    elif alg_base == "viterbi":
                        PQ, prev_ind = np.max([P,Q]), np.argmax([P,Q])
                        prev_state = ["E", "C"][prev_ind]
                        prev_pos = [[seqi,-3], [seqi-1,-1]][prev_ind]
                        dp_m[seqi,-1]["prev"] = (prev_state, prev_pos)

                    dp_m[seqi,-1]["log-odds"] = PQ

                    #b
                    P = np.log2(np.exp2(dp_m[seqi,0]["log-odds"]) * anb)
                    Q = np.log2(np.exp2(dp_m[seqi,-2]["log-odds"]) * ajb)
                    if alg_base == "forward":
                        PQ = np.logaddexp2(P,Q)
                    elif alg_base == "viterbi":
                        PQ, prev_ind = np.max([P,Q]), np.argmax([P,Q])
                        prev_state = ["N", "J"][prev_ind]
                        prev_pos = [[seqi,0], [seqi,-2]][prev_ind]
                        dp_m[seqi,1]["prev"] = (prev_state, prev_pos)

                    dp_m[seqi,1]["log-odds"] = PQ

                #print(id_seq, (dp_m[-1,-1]["log-odds"] + np.log2(act) - (np.logaddexp2(-len(seq)*np.log2(arr), np.log2(1-arr)))))
                print(id_seq, (dp_m[-1,-1]["log-odds"] + np.log2(act)) - (len(seq)*np.log2(arr) + np.log2(1-arr)))
                #print(dp_m[:,5])
                #print(dp_m[-1,-1]["log-odds"] - len(seq)*np.log2(act))
                if alg_base == "viterbi":
                    path = []
                    #print(dp_m[50,104]["prev"])
                    path.append(dp_m[-1,-1]["prev"])
                    while path[-1] is not None:
                        #print(path[-1])
                        #print(path[-1][1])
                        #print([path[-1][1][0], path[-1][1][1]])
                        if path[-1][0] in ["M","I","D","/MM/"]:
                            if path[-1][0] == "M":
                                path.append(dp_m[path[-1][1][0], path[-1][1][1]]["prev M"])
                            elif path[-1][0] == "I":
                                path.append(dp_m[path[-1][1][0], path[-1][1][1]]["prev I"])
                            elif path[-1][0] == "D":
                                path.append(dp_m[path[-1][1][0], path[-1][1][1]]["prev D"])
                            elif path[-1][0] == "/MM/":
                                path.append(dp_m[path[-1][1][0], path[-1][1][1]]["prev /MM/"])
                        else:
                            path.append(dp_m[path[-1][1][0], path[-1][1][1]]["prev"])
                        #print(path)
                    capital_letters_only = ["S" if item is None else item[0] for item in path]
                    capital_letters_only.reverse()
                    capital_letters_only = ''.join(capital_letters_only)
                    print(capital_letters_only)
                    print(len(seq))
                    print(len(capital_letters_only))



                
            





                

                
test = Decoder("globins45.fa","globins4.hmm")
#test.forward()
test.inverse(alg_base="viterbi", inverse_mode = True)
