import numpy as np
import scipy.special as sp
from tabulate import tabulate
from tqdm import tqdm
import numpy as np


class Decoder:
    import warnings

    # Filter out RuntimeWarnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
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
        self.consseq = ""
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
                    self.consseq += line[22]
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
        #self.nullem0 = np.log2(self.nullem0)


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
                    T = np.log2(np.exp2(dp_m[seqi-1,statej-1]["log-odds M"]) * self.amm[em_rowind]) # = dp_m[seqi-1,statej-1]["log-odds M"] + np.log2(self.amm[em_rowind])
                    Q = np.log2(np.exp2(dp_m[seqi-1,statej-1]["log-odds I"]) * self.aim[em_rowind]) # = dp_m[seqi-1,statej-1]["log-odds I"] + np.log2(self.adm[em_rowind])
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
  



    def inverse(self, alg_base="forward", inverse_mode = False, consensus_alignment = False, mspeptide_alignment = True):
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
        if alg_base == "viterbi" and consensus_alignment == True and mspeptide_alignment == True:
            print("Consensus sequence:       ",self.consseq)
            
        
        total_sequences = len(self.sequences)
        progress_bar = tqdm(total=total_sequences, desc="Sequences decoded", position=0)
        modified_list_mspepalign = [self.consseq]
        id_list = ["Consensus seq"]
        state_list,full_state_list = [],[]
        for id_seq, stuff in self.sequences:
                
                #print("Alignment of", id_seq)
                seq = list(stuff)
                len_seq = len(seq)
                
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
                        T = dp_m[seqi-1,statej-1]["log-odds M"] + np.log2(self.amm[em_rowind]  * iir)
                        Q = dp_m[seqi-1,statej-1]["log-odds I"] + np.log2(self.adm[em_rowind] * iir)
                        S = dp_m[seqi-1,statej-1]["log-odds D"] + np.log2(self.adm[em_rowind]  * iir)
                        P = dp_m[seqi-1,1]["log-odds"] + np.log2( abm * iir)
                        R = dp_m[seqi-2,statej-2]["log-odds /MM/"] + np.log2(self.amm[em_rowind]  * iir)
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
                        P = dp_m[seqi-1,statej]["log-odds M"] + np.log2(self.ami[em_rowind])
                        Q = dp_m[seqi-1,statej]["log-odds I"] + np.log2(self.aii[em_rowind])
                        R = dp_m[seqi-1,statej-1]["log-odds /MM/"] + np.log2(self.ami[em_rowind]) #This is because Ik depends on the previous MM(k-1) state
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
                        P = dp_m[seqi,statej-1]["log-odds M"] + np.log2(self.amd[em_rowind])
                        Q = dp_m[seqi,statej-1]["log-odds D"] + np.log2(self.add[em_rowind])
                        R = dp_m[seqi,statej-2]["log-odds /MM/"] + np.log2(self.amd[em_rowind]) #Easier to understand if one look at the graph diagram but transitions to delete from MM is a jump of 2 states (k) in the diagram
                        if alg_base == "forward":    
                            PQ = np.logaddexp2.reduce([P,Q,R])
                        elif alg_base == "viterbi":
                            PQ, prev_ind = np.max([P,Q,R]), np.argmax([P,Q,R])
                            prev_state = ["M","D","/MM/"][prev_ind]
                            prev_pos = [[seqi,statej-1], [seqi,statej-1],[seqi,statej-2]][prev_ind]
                            dp_m[seqi,statej]["prev D"] = (prev_state, prev_pos)
                        dp_m[seqi,statej]["log-odds D"] = PQ 


                        #mm, 
                        T = dp_m[seqi-1,statej-1]["log-odds M"] + np.log2(self.amm[em_rowind] * ir)
                        Q = dp_m[seqi-1,statej-1]["log-odds I"] + np.log2(self.aim[em_rowind] * ir)
                        S = dp_m[seqi-1,statej-1]["log-odds D"] + np.log2(self.adm[em_rowind] * ir)
                        P = dp_m[seqi-1,1]["log-odds"] + np.log2(abm * ir)
                        R = dp_m[seqi-2,statej-2]["log-odds /MM/"] + np.log2(self.amm[em_rowind] * ir) 
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
                        inem_rowin += 1 
                        
                    #e
                    holdit = []
                    state_remeber = []
                    stateind_remember = []
                    for statej in range(4, dp_m.shape[1]-3):
                        holdit.append(dp_m[seqi,statej]["log-odds M"] + np.log2(ame))
                        holdit.append(dp_m[seqi,statej]["log-odds D"] + np.log2(ade))
                        holdit.append(dp_m[seqi,statej]["log-odds /MM/"] + np.log2(ame)) #assuming that we may end at MM
                        state_remeber += ["M", "D", "/MM/"]
                        stateind_remember += [statej, statej, statej]
                    if alg_base == "forward":
                        #logsumexp function is fast.....but only does it on the natural log, transform it to natural log and back into log 2, might not be efficient
                        holdit = holdit / np.log2(np.e)
                        log_tot = sp.logsumexp(holdit)
                        log_tot = log_tot * np.log2(np.e)
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
                    #log_tot = log_tot/np.log(2)
                    #log_tot = np.logaddexp2.reduce(holdit)
                    #print(log_tot)

                    dp_m[seqi,-3]["log-odds"] = log_tot

                    #n
                    dp_m[seqi,0]["log-odds"] = dp_m[seqi-1,0]["log-odds"] + np.log2(ann) #np.log2(np.exp2(dp_m[seqi-1,0]["log-odds"]) * ann)
                    dp_m[seqi,0]["prev"] = ("N", [seqi-1,0])

                    #j
                    P = dp_m[seqi,-3]["log-odds"] + np.log2(aej)
                    Q = dp_m[seqi-1,-2]["log-odds"] + np.log2(ajj)
                    if alg_base == "forward":
                        PQ = np.logaddexp2(P,Q)
                    elif alg_base == "viterbi":
                        PQ, prev_ind = np.max([P,Q]), np.argmax([P,Q])
                        prev_state = ["E", "J"][prev_ind]
                        prev_pos = [[seqi,-3], [seqi-1,-2]][prev_ind]
                        dp_m[seqi,-2]["prev"] = (prev_state, prev_pos)
                    dp_m[seqi,-2]["log-odds"] = PQ
                    
                    #c
                    P = dp_m[seqi,-3]["log-odds"] + np.log2(aec)
                    Q = dp_m[seqi-1,-1]["log-odds"] + np.log2(acc)
                    if alg_base == "forward":
                        PQ = np.logaddexp2(P,Q)
                    elif alg_base == "viterbi":
                        PQ, prev_ind = np.max([P,Q]), np.argmax([P,Q])
                        prev_state = ["E", "C"][prev_ind]
                        prev_pos = [[seqi,-3], [seqi-1,-1]][prev_ind]
                        dp_m[seqi,-1]["prev"] = (prev_state, prev_pos)

                    dp_m[seqi,-1]["log-odds"] = PQ

                    #b
                    P = dp_m[seqi,0]["log-odds"] + np.log2(anb)
                    Q = dp_m[seqi,-2]["log-odds"] + np.log2(ajb)
                    if alg_base == "forward":
                        PQ = np.logaddexp2(P,Q)
                    elif alg_base == "viterbi":
                        PQ, prev_ind = np.max([P,Q]), np.argmax([P,Q])
                        prev_state = ["N", "J"][prev_ind]
                        prev_pos = [[seqi,0], [seqi,-2]][prev_ind]
                        dp_m[seqi,1]["prev"] = (prev_state, prev_pos)

                    dp_m[seqi,1]["log-odds"] = PQ

                #print(id_seq, (dp_m[-1,-1]["log-odds"] + np.log2(act) - (np.logaddexp2(-len(seq)*np.log2(arr), np.log2(1-arr)))))
                with open("output.txt", "a") as f:
                    result = id_seq + "\n"  # Start building the result string with id_seq
                    result += str((dp_m[-1,-1]["log-odds"] + np.log2(act)) - (len(seq)*np.log2(arr) + np.log2(1-arr))) + "\n"  # Append the computation result
                    bitscore = (dp_m[-1,-1]["log-odds"] + np.log2(act)) - (len(seq)*np.log2(arr) + np.log2(1-arr))
                    if alg_base == "viterbi" and consensus_alignment == True and bitscore >= -5:
                        path = []
                        path.append(dp_m[-1,-1]["prev"])
                        while path[-1] is not None:
                            if path[-1][0] in ["M", "I", "D", "/MM/"]:
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
                        path.reverse()
                        capital_letters_only = ''.join(capital_letters_only)
                        capital_letters_only = capital_letters_only.replace("/MM/", "ÖÖ")
                        result += capital_letters_only + " "  # Append the computed result
                        #we need to add a functionallity to align the optimum states to the consesnsus structure of the profile turn them into the corresponding residues
                        #consensus sequences is already parsed, contained in self.consseq
                        #we need to know where the found optimum states are located in the model
                        #we also need to carry over the functionalitty to pinpoint inverted residues
                        aligned_seq = []
                        previous_char = None
                        envelop_list = []
                        for index, element in enumerate(path):
                            if element == None:
                                aligned_seq.append(False)
                            elif element[0] == 'N':
                                if previous_char == 'N':
                                    aligned_seq.append(True)
                                else:
                                    aligned_seq.append(False)
                            elif element[0] == "B":
                                aligned_seq.append(False)
                                envelop_list.append(index)
                            elif element[0] == "M":
                                aligned_seq.append(True)
                            elif element[0] == "I":
                                aligned_seq.append(True)
                            elif element[0] == "D":
                                aligned_seq.append(True)
                            elif element[0] == "/MM/":
                                aligned_seq.append(True)
                            elif element[0] == "E":
                                aligned_seq.append(False)
                                envelop_list.append(index)
                            elif element[0] == 'J':
                                if previous_char == 'J':
                                    aligned_seq.append(True)
                                else:
                                    aligned_seq.append(False)
                            previous_char = element[0] if element is not None else element
                        pairs = []
                        for i in range(0, len(envelop_list), 2):
                            if i + 1 < len(envelop_list):
                                pairs.append((envelop_list[i], envelop_list[i + 1]))
                        #print(pairs)
                        intervals = []
                        model_interval = []
                        seq_interval = []
                        for envelop in pairs:
                            intervals.append((path[envelop[0]+1][1],path[envelop[1]-1][1]))
                        for inter in intervals:
                            model_interval.append((inter[0][1],inter[1][1]))
                            seq_interval.append((inter[0][0],inter[1][0]))
                        #print(seq_interval)
                        #print(model_interval)
                        
                        domain_list = []
                        aligned_list = []
                        sub_seq = ""
                        for inter in model_interval:
                            domain_list.append(self.consseq[inter[0]-4:inter[1]-3])
                        for inter in seq_interval:
                            for i in range(inter[0]-2, inter[1]-1):
                                sub_seq += seq[i]
                            aligned_list.append(sub_seq)
                            sub_seq = ""
                        #print(domain_list, len(domain_list[0]))
                        #print(aligned_list, len(aligned_list[0]))

                        ## Initialize list to store extracted substrings
                        substring_list = []

                        # Initialize flag to indicate whether to start storing substring
                        store_substring = False

                        # Iterate over characters in the input string
                        for char in capital_letters_only:
                            # If current character is "B", set flag to start storing substring
                            if char == "B":
                                store_substring = True
                                substring = ""
                            # If current character is "E", set flag to stop storing substring
                            elif char == "E":
                                store_substring = False
                                substring_list.append(substring)
                            # If flag is True, store current character in substring
                            elif store_substring:
                                substring += char
                        #print(capital_letters_only)
                        #print(substring_list)
                        modified_list = []

                        for aligned_string, substring_string in zip(aligned_list, substring_list):
                            modified_string = ""
                            substring_iterator = iter(substring_string)
                            print("hej",aligned_string, substring_string)

                            stopper = 0  # Initialize the stopper flag
                            aligned_iterator = iter(aligned_string)
                            deleter = 0
                            while True:
                                try:
                                    if deleter == 0:
                                        aligned_char = next(aligned_iterator)
                                    else:
                                        print("jag e bög", aligned_char)
                                    
                                    substring_char = next(substring_iterator)
                                    if stopper == 1:
                                        print("big bög", substring_char)
                                except StopIteration:
                                    break

                                #if stopper == 1:  # Check if stopper flag is set
                                #    stopper = 0  # Reset the stopper flag
                                #    continue  # Stay on the same iteration if stopper is 1
                                if deleter == 1:
                                    deleter = 0
                                #    continue

                                if substring_char == "D":
                                    modified_string += "-" 
                                    deleter = 1
                                    #modified_string += aligned_char  # Include aligned_char # Add "-"
                                elif substring_char == "M":
                                    stopper = 0
                                    modified_string += aligned_char.upper()
                                elif substring_char == "I":
                                    stopper = 0
                                    #modified_string += aligned_char.lower()
                                elif substring_char == "Ö":
                                    #stopper = 1
                                    modified_string += aligned_char.lower()
                                    #modified_string += aligned_char
                            print("DÅ", modified_string)
                            modified_list.append(modified_string)

                        #print(modified_list)



                            


                                





                            




                    
                        

                    result += "\n"  # Start a new line for the next result


                    print(result)
                    if bitscore >= -5:
                        full_state_list.append(capital_letters_only)        
                        state_list.append(substring_string)
                    
                    if alg_base == "viterbi" and consensus_alignment == True and mspeptide_alignment == False and bitscore >= -5:
                        full_state_list.append(capital_letters_only)        
                        state_list.append(substring_string)
                        for consensus, mod_ali in zip(domain_list, modified_list):
                            alignment_str = ""
                            for consensus_char, mod_ali_char in zip(consensus, mod_ali):
                                consensus_char = consensus_char.lower()  # Convert to lowercase
                                mod_ali_char = mod_ali_char.lower()      # Convert to lowercase
                                if consensus_char == mod_ali_char:
                                    alignment_str += "\033[92m" + consensus_char + "\033[0m"  # Green for conserved residues
                                elif consensus_char == "-" or mod_ali_char == "-":
                                    alignment_str += "\033[91m" + "*" + "\033[0m"  # Red asterisk for gaps
                                elif mod_ali_char == "/":
                                    alignment_str += "\033[93m" + "/" + "\033[0m"  # Yellow for "/"
                                else:
                                    alignment_str += mod_ali_char  # Display modified alignment
                            print("Consensus:       ", consensus)
                            print("Target sequence: ", alignment_str)
                            print("//////////////////////")
                    #f.write(result)
                        #print(len(seq))
                        #print(len(capital_letters_only))
                if alg_base == "viterbi" and consensus_alignment == True and mspeptide_alignment == True and bitscore >= -5:
                    for i, mod_ali in enumerate(modified_list):
                        empty_seq = list(" " * len(self.consseq))
                        mod_ali_list = list(mod_ali)
                        empty_seq[model_interval[i][0]-4:model_interval[i][1]-3] = mod_ali_list
                        modified_list_mspepalign.append("".join(empty_seq))
                        id_list.append(id_seq)
                    
            
                progress_bar.update(1)
        progress_bar.close()

        def find_blank_spaces(input_string): #needed for checking for gaps, for the sake of finding the minimum amount of needed peptides to give full coverages
            blank_spaces_indices = []
            for index, char in enumerate(input_string):
                if char.isspace():
                    blank_spaces_indices.append(index)
            return blank_spaces_indices
        def compare_lists(list1, list2):
            # Convert lists to sets to perform set operations
            set1 = set(list1)
            set2 = set(list2)
            
            # Find common values
            common_values = set1.intersection(set2)
            
            # Create a new list with common values
            new_list = list(common_values)
            
            return new_list
        full_peptide_cov = None
        merged = []  # Initialize the list of merged indices
        gaps_check = True
        taken_list = []
        estimation_peptides_cov = 0
        i = 1
        while i < len(modified_list_mspepalign):
            if i in taken_list:
                i += 1
                #print("SKIPPED", taken_list)
                continue   
            seq = modified_list_mspepalign[i][:len(self.consseq)]
            j = 1
            while j < len(modified_list_mspepalign):
                start_index = None
                end_index = None
                seq_comp = modified_list_mspepalign[j]
                for idx, char in enumerate(seq_comp):
                    if char != ' ':
                        start_index = idx
                        break
                for idx in range(len(seq_comp) - 1, -1, -1):
                    if seq_comp[idx] != ' ':
                        end_index = idx
                        break
                if seq[start_index:end_index+1] == (" " * (end_index + 1 - start_index)) and j not in taken_list: #seq_comp[start_index:end_index] not in merged:
                    modified_seq = seq[:start_index] + seq_comp[start_index:end_index+1] + seq[end_index+1:len(self.consseq)]
                    seq = modified_seq
                    merged.append(seq_comp[start_index:end_index+1])
                    taken_list.append(j)
                    estimation_peptides_cov += 1
                j += 1
                modified_list_mspepalign[i] = seq
                if gaps_check == True:
                    if i == 1 and find_blank_spaces(seq) == []: #in the case full coverage is achived just from the first "assembly" of aligned peptides, REALLY unlikely
                        full_peptide_cov = len(set(taken_list)) + i #amount of peptide sequences that have been appended plus the amount of peptides that these peptides have been appended to
                        gaps_check == False
                    elif i == 1:
                        gaps_prev = find_blank_spaces(seq)
                        gaps = compare_lists(find_blank_spaces(" " * len(self.consseq)), gaps_prev)
                    else:
                        gaps_curr = find_blank_spaces(seq)
                        gaps = compare_lists(gaps_curr, gaps_prev) #the amount of gaps present so far
                        print("these are the gaps", gaps)
                        if gaps == []:
                            full_peptide_cov = len(set(taken_list)) + i
                            print("borde ej ske", i, gaps)
                            estimation_peptides_cov = i
                            gaps_check = False
                        else:
                            gaps_prev = gaps

                
            i += 1


        #Remove elements from modified_list_mspepalign using indices in merged
        for index in sorted(taken_list, reverse=True):
            if 0 <= index < len(modified_list_mspepalign):
                del modified_list_mspepalign[index]

        # Remove duplicates from modified_list_mspepalign while preserving order
        #modified_list_mspepalign = list(dict.fromkeys(modified_list_mspepalign))
        
        
        if alg_base == "viterbi" and consensus_alignment == True and mspeptide_alignment == True:
            # ANSI escape code for blue color
            blue_color = "\033[94m"
            # ANSI escape code to reset color
            reset_color = "\033[0m"
            
            # Print table header
            print(f"{'ID':<20} {'Alignment'}")
            
            
            
            def most_occuring_letter(sequences):
                # Initialize the resulting sequence
                result_sequence = ''
                # Iterate over each position in the sequences
                for i in range(len(sequences[0])):
                    # Initialize a dictionary to count occurrences of characters at this position
                    char_count = {}
                    # Iterate over each sequence
                    for sequence in sequences[1:]:
                        # Get the character at this position in the sequence
                        char = sequence[i]
                        if char == char.lower:
                            char_inv = sequence[i+1]
                        # Update the count for this character
                        if char.strip():  # Check if the character is not whitespace
                            if char == char.lower and char_inv == char_inv.lower:
                                char_count[char.upper] = char_count.get(char.upper, 0) + 1
                                char_count[char_inv.upper] = char_count.get(char_inv.upper, 0) + 1
                            else:
                                char_count[char] = char_count.get(char, 0) + 1
                    # Get the most occurring non-whitespace character at this position
                    most_occuring_char = max(char_count, key=char_count.get, default=' ')
                    # Append the most occurring character to the result sequence
                    result_sequence += most_occuring_char
                return result_sequence
            def count_letters(strings_list, letters):
                total_residues = sum(len(peptide) for peptide in strings_list)
                total_letter_count = 0
                for peptide in strings_list:
                    total_residues -= 4
                    if "C" in peptide:
                        total_residues -= 1
                
                for peptide in strings_list:
                    for letter in letters:
                        total_letter_count += peptide.count(letter)
                if letters == ["Ö"]:
                    print("Ö", total_letter_count)
                
                if letters == ["N"]:
                    for peptide in strings_list:
                        if "N" in peptide:
                            total_letter_count -= 1
                if letters == ["C"]:
                    for peptide in strings_list:
                        if "C" in peptide:
                            total_letter_count -= 1
                


                return total_letter_count / total_residues
            
            
            print("Estimated procentage of peptide residues predicted to be homologous to the model:\n",count_letters(full_state_list, ["M","I","D","Ö"]))
            print("Estimated procentage of peptide residues predicted to be non-homologous to the model, N-terminal side:\n", count_letters(full_state_list, ["N"]))
            print("Estimated procentage of peptide residues predicted to be non-homologous to the model, C-terminal side:\n", count_letters(full_state_list, ["C"]))
            print("Estimated procentage of peptide residues predicted to be encompass inversion uncertainty:\n", count_letters(full_state_list, ["Ö"]))
             # ANSI escape code for blue color
            blue_color = "\033[94m"
            # ANSI escape code to reset color
            reset_color = "\033[0m"
            red_color = "\033[91m"  # ANSI escape code for red color
            ref_seq = most_occuring_letter(modified_list_mspepalign)
            # Print table header
            print(f"{'ID':<20} {'Alignment'}")
            print(f"{'resulting sequence:':<26} {ref_seq}")
            mismatch = 0
            for i, mod_ali in enumerate(modified_list_mspepalign):
                if i == 0:
                    # Print the first mod_ali in blue
                    print(f"{'Conensus':<20} {i:<5} {blue_color}{mod_ali}{reset_color}")
                else:
                    highlighted_mod_ali = ''
                    for mod_char, ref_char in zip(mod_ali, ref_seq):
                        if mod_char == ref_char or mod_char == ' ':
                            highlighted_mod_ali += mod_char
                        else:
                            # Highlight differences in red
                            highlighted_mod_ali += red_color + mod_char + reset_color
                            mismatch += 1
                    print(f"{'Alignment':<20} {i:<5} {highlighted_mod_ali}")

            if full_peptide_cov is not None:
                print("Estimated amount of peptides needed for full coverages:", estimation_peptides_cov)
                print("Procentage of mismatched residues in the peptide alignment:", (mismatch/sum(len(peptide) for peptide in state_list)))
                print("test", len(set(taken_list)) + i)
            else:
                print("Coverages to the model estimated to (%):", (len(self.consseq) - len(gaps)) / len(self.consseq))
                print("Procentage of mismatched residues in the peptide alignment:", (mismatch/sum(len(peptide) for peptide in state_list)))

                print(mismatch, sum(len(peptide) for peptide in state_list), len(self.consseq), len(gaps))



            #Evaluating the coverage and estimating the best guess for the full protein sequence
            #We will estimate the frequencies of amino acids in our alignment, residues generated from MM state will count two corresponding residues to
                    

                                
                            





                

                
test = Decoder("sequences.FASTA","abdihmm")
#test.forward()
test.inverse(alg_base="viterbi", inverse_mode = False, consensus_alignment = True, mspeptide_alignment = True)
