import numpy as np
import scipy.special as sp
from tabulate import tabulate
from tqdm import tqdm
import numpy as np
from functools import reduce
import math
from numba import jit
from numba.core.errors import NumbaPendingDeprecationWarning
import timeit
from numba.experimental import jitclass
import warnings
import copy

# Filter out RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)



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

        self.emM = np.log2(np.exp(-self.emM))
        self.inemM = np.log2(np.exp(-self.inemM))

        
        np.savetxt('tranM.txt', self.tranM, fmt='%8.6s', delimiter='\t')
        np.savetxt('emM.txt', self.emM, fmt='%8.6s', delimiter='\t')
        np.savetxt('imemM.txt', self.inemM, fmt='%8.6s', delimiter='\t')
        
        #keep in mind, inemM has one extra row, because of the state type INSERT has one extra state, 0. Same goes for the tranM, has one extra row (self.leng + 1)
        #I will remove the first row from tranM because it only contains how transition from M0 and transitions from I0 is and that is not relevant in the plan 7 model
        
        self.amm = np.log2(np.exp(-self.tranM[:,0] ))
        self.ami = np.log2(np.exp(-self.tranM[:,1] ))
        self.amd = np.log2(np.exp(-self.tranM[:,2] ))
        self.amd[-2] = -np.inf #This might be used, it is really unclear but some sources states that the Plan7viterbi does NOT use the last delete state, as I understand it, it is enough to remove the first delete state in the core model so each search always matches one residue to match
        self.aim = np.log2(np.exp(-self.tranM[:,3]))
        self.aii = np.log2(np.exp(-self.tranM[:,4]))
        self.adm = np.log2(np.exp(-self.tranM[:,5]))
        self.add = np.log2(np.exp(-self.tranM[:,6]) )#decided to seperate them (mm = match to match etc, a for transition as durbin does)
        self.add[-2] = -np.inf
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
        self.nullem0 = np.log2(self.nullem0)


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
            #print(id_seq, (dp_m[-1,-1]["log-odds"] + np.log2(act)) - (len(seq)*np.log2(arr) + np.log2(1-arr)))
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
        def find_max_index(arr):
            """
            Find the index of the maximum value in a list.

            Parameters:
                arr (list): The list of numbers.

            Returns:
                int: The index of the maximum value.
            """
            max_value = float('-inf')
            max_index = None
          
        
        total_sequences = len(self.sequences)


        progress_bar = tqdm(total=total_sequences, desc="Sequences decoded", position=0, miniters=0)
        modified_list_mspepalign = [self.consseq]
        id_list = ["Consensus seq"]
        state_list,full_state_list = [],[]

        for id_seq, stuff in self.sequences:
                
                
                #print("Alignment of", id_seq)
                seq = list(stuff)
                len_seq = len(seq)
                
                self.symb_index = [self.symbols.index(j) for j in seq] #easier usage later
                

                #now for some transition probabilities, in line with what sean R. EDDY said
                anb = act = ajb = np.log2(3/(len(seq)+3)) #? #pretty much the same as hmmer if 3 is replaced with 6?????? wtf
                ann = ajj = acc = np.log2(len(seq)/(len(seq) +3))

                aej = aec = np.log2(0.5)
                abm = np.log2(2/((self.leng*(self.leng + 1))))
                ame = ade = np.log2(1)
                nullscore = 0
                arr = np.log2(len(seq)/(len(seq)+1)) #the transition prob for going from R to R in the null model
                exp2arr = np.exp2(arr)
                divide_two = np.log2(2)


                #some comments before implementing the MM state
                #MM state in DP programming requires to look back 2 steps in both row and col, to not get an index error we extand the row (seq) and col (states), these will be nonexistant Seq(-1) state(-1)
                #MM1 will be seen as the same "level" as M1 and same for MM2 and M2, however the transition probabilities from MM(k) will be the same as for M(k+1)
                #the transition from other states into MM are the same as into M but multiplied with ir

                dp_m = np.array([[{}]*(self.leng+2+5) for _ in range(len(seq)+2)]) #the last 3 columns are the special states in the order E, J, C and first 2 are N and B, the 2 columns efter B are nonexistant states and the same
                #print(dp_m.shape)
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
                dp_m[1,1] = {"log-odds": (anb), "prev": ("N", [0,0])}
                #print(dp_m[0,3]["log-odds M"])
                if inverse_mode == True:
                    ir = 0.02564102564102564
                                                            #disregard the wall of comment above, the new value is based on the frequency of WRONG residue times the fraction of the errors that are within this type
                                        
                    iir = 1-ir
                    ir, iir = math.log2(ir), math.log2(iir)
                else:
                        ir = -np.inf
                        iir = 0
                def compute_scorev(self, ir, iir, alg_base = "forward", inverse_mode = False): #SUCCESS but for the moment only does viterbi

                    dp_mv = np.empty((len(seq)+2, (self.leng)+7, 4)) #times 4 because we have 4 states, add 7 because we have 5 extra states and we need 2 fake states, 2 fake sequence positions too
                    btrace = np.empty((len(seq)+2, (self.leng)+7, 4), dtype=object)


                    #print(index_m[index_m[2,155,2][0],index_m[2,155,2][1],index_m[2,155,2][2]])

                    

                    

                    #initiate the things brotha
                    dp_mv[:,:,:] = -np.inf

                    #init N state

                    dp_mv[1,0,0] = 0
                    btrace[1,0,0] = None
                    for i in range(2, dp_mv.shape[0]):
                        dp_mv[i,0,0] = dp_mv[i-1,0,0] + ann
                        btrace[i,0,0] = (i-1,0,0)

                    #init B state
                    dp_mv[1,1,0] = anb
                    btrace[1,1,0] = (1,0,0)

                    #start!
                    em_rowind = 0
                    inem_rowind = 1
                    for seqi in range(2, dp_mv.shape[0]):
                        if seqi == dp_mv.shape[0] or inverse_mode == False:
                            ir = -np.inf
                            iir = 0
                        #M glön inte ir samt att korregera för de tillfällen MM inte finns borde korregera för sista state och seq
                        #print(len(self.amm[1:]/np.log2(np.e))) #OMG TA INTE FÖRSTA ELEMENTET UR TRANSITIONS FÖR DEN ÄR FUCKING START AAAAAAh
                        #print(len(dp_mv[seqi-1,4:-3,1])) #normal
                        #print(len(dp_mv[seqi-1,3:-4,1])) #state back 1 #one save for matrix holder
                        #print("HUJBFU", type(sliced_ind[3,148][0]))
                        part = np.array([dp_mv[seqi-1,3:-4,0] + self.amm[1:],
                                                dp_mv[seqi-1,3:-4,1] + self.aim[1:],
                                                dp_mv[seqi-1,3:-4,2] + self.adm[1:],
                                                dp_mv[seqi-2,2:-5,3] + self.amm[1:],
                                                dp_mv[seqi-1,1,0] + (abm + np.zeros(self.leng))])
                        if alg_base == "forward":

                            winner = np.logaddexp2.reduce((dp_mv[seqi-1,3:-4,0] + self.amm[1:],
                                                dp_mv[seqi-1,3:-4,1] + self.aim[1:],
                                                dp_mv[seqi-1,3:-4,2] + self.adm[1:],
                                                dp_mv[seqi-2,2:-5,3] + self.amm[1:],
                                                dp_mv[seqi-1,1,0] + abm + np.zeros(self.leng)), axis = 0)
                        else:
                            winner = np.max(part,axis = 0)
                            ind = np.argmax(part, axis=0)
                            new_list = [(seqi-2 if Z== 3 else seqi-1, i+2 if Z == 3 else i+3, Z) for i, Z in enumerate(ind)]
                            new_list = [(seqi-1, 1, 0) if tpl[2] == 4 else tpl for tpl in new_list]
                            btrace[seqi, 4:-3, 0] = new_list
                            #print(btrace[seqi, 4:-3, 0])



                        #print(winnerarg2[0])
                        
                        
                        #print(helper)
                        winner[-1] = winner[-1] - iir
                        dp_mv[seqi, 4:-3, 0] = winner + iir + self.emM[:, self.symb_index[seqi-2]] - self.nullem0[self.symb_index[seqi-2]]






                        #I
                        part = np.array([dp_mv[seqi-1,4:-3,0] + self.ami[1:],
                                                dp_mv[seqi-1,4:-3,1] + self.aii[1:],
                                                dp_mv[seqi-1,3:-4,3] + self.ami[1:]])
                        if alg_base == "forward":
                            winner = np.logaddexp2.reduce(part, axis=0)
                        else:
                            winner = np.max(part,axis = 0)
                            ind = np.argmax(part, axis=0)
                            new_list = [(seqi-1, i+3 if Z == 3 else i+4, Z) for i, Z in enumerate(ind)]
                            btrace[seqi, 4:-3, 1] = new_list
                        dp_mv[seqi, 4:-3, 1] = winner + self.inemM[1:, self.symb_index[seqi-2]] - self.nullem0[self.symb_index[seqi-2]]



                        #print(dp_mv[seqi, 3:-4, 0])
                        #D
                        part = np.array([dp_mv[seqi,3:-4,0] + self.amd[1:],
                                                dp_mv[seqi,3:-4,2] + self.add[1:],
                                                dp_mv[seqi,2:-5,3] + self.amd[1:]])
                        
                        if alg_base == "forward":
                            winner = np.logaddexp2.reduce(part, axis=0)
                        else:
                            winner = np.max(part,axis = 0)
                            ind = np.argmax(part, axis=0)
                            new_list = [(seqi, i+2 if Z == 3 else i+3, Z) for i, Z in enumerate(ind)]
                            btrace[seqi, 4:-3, 2] = new_list
                        dp_mv[seqi, 4:-3, 2] = winner



                        #MM
                        part = np.array([dp_mv[seqi-1,3:-4,1] + self.amm[:-1] + ir,
                        dp_mv[seqi-1,3:-4,1] + self.aim[1:] + ir,
                        dp_mv[seqi-1,3:-4,2] + self.adm[1:] + ir,
                        dp_mv[seqi-2,2:-5,3] + self.amm[1:] + ir,
                        dp_mv[seqi-1,1,0] + (abm + np.zeros(self.leng)) + ir])
                        if alg_base == "forward":
                            winner = np.logaddexp2.reduce(part, axis=0)
                        else:
                            winner = np.max(part,
                            axis = 0)
                            ind = np.argmax(part, axis=0)
                            new_list = [(seqi-2 if Z== 3 else seqi-1, i+2 if Z == 3 else i+3, Z) for i, Z in enumerate(ind)]
                            new_list = [(seqi-1, 1, 0) if tpl[2] == 4 else tpl for tpl in new_list]
                            btrace[seqi, 4:-3, 3] = new_list

                        winner[-1] = -np.inf
                        if seqi == dp_mv.shape[0]-1:
                            dp_mv[seqi, 4:-3, 3] = winner
                        else:
                            dp_mv[seqi,4:-3,3] = winner + (self.emM[:, self.symb_index[seqi-2]] - self.nullem0[self.symb_index[seqi-2]] + np.roll(self.emM[:, self.symb_index[seqi-1]], -1) - self.nullem0[self.symb_index[seqi-1]] + self.emM[:, self.symb_index[seqi-1]] - self.nullem0[self.symb_index[seqi-1]] + np.roll(self.emM[:, self.symb_index[seqi-2]], -1) - self.nullem0[self.symb_index[seqi-2]]) - divide_two
                                #print((self.emM[em_rowind, self.symb_index[seqi-2]] - self.nullem0[self.symb_index[seqi-2]] + self.emM[em_rowind+1, self.symb_index[seqi-1]] - self.nullem0[self.symb_index[seqi-1]] + self.emM[em_rowind, self.symb_index[seqi-1]] - self.nullem0[self.symb_index[seqi-1]] + self.emM[em_rowind+1, self.symb_index[seqi-2]] - self.nullem0[self.symb_index[seqi-2]]) - divide_two)

                        #E
                        part = [dp_mv[seqi,4:-3,0] + (ame), dp_mv[seqi,4:-3,2] + (ade), dp_mv[seqi,4:-3,3] + (ame)]
                        if alg_base == "forward":
                            winner = np.logaddexp2.reduce(part)
                            winner = np.logaddexp2.reduce(winner)
                        else:
                            winner = np.max(part)
                            ind=np.argmax(part)
                            ind2=np.argmax(part, axis=0)
                            #print(ind)
                            #print("WTF", ind)
                            
                            #this one got tricky, we need to check at what level of state that is most probable to exit the core model, as well as what type of state it was, I cant figure out any other way than to use argmax two times
                            if ind // self.leng == 0:
                                #print("JAG KÖRDES")
                                btrace[seqi,-3,0] = (seqi, ind % self.leng + 4, 0)
                            elif ind // self.leng == 1:
                                btrace[seqi,-3,0] = (seqi, ind % self.leng +4, 2)
                            elif ind // self.leng == 2:
                                btrace[seqi,-3,0] = (seqi, ind % self.leng +4, 3)
                        #print(winner)
                        dp_mv[seqi,-3,0] = winner

                        
                        #J
                        if alg_base == "forward":
                            winner = np.logaddexp2(dp_mv[seqi,-3,0] + (aej), dp_mv[seqi-1,-2,0] + (ajj))
                        else:
                            winner = np.max([
                                dp_mv[seqi,-3,0] + (aej + np.zeros(self.leng)), dp_mv[seqi-1,-2,0] + (ajj + np.zeros(self.leng))
                            ])
                            
                            slices = ((seqi, -3, 0), (seqi-1, -2, 0))
                            winnerarg = np.argmax([
                                np.max(dp_mv[seqi,-3,0] + (aej + np.zeros(self.leng))), np.max(dp_mv[seqi-1,-2,0] + (ajj + np.zeros(self.leng)))
                            ])
                            btrace[seqi,-2,0] = slices[winnerarg]
                        dp_mv[seqi,-2,0] = winner
                        #C
                        if alg_base == "forward":
                            winner = np.logaddexp2(
                            dp_mv[seqi,-3,0] + (aec), dp_mv[seqi-1,-1,0] + (acc))
                        else:
                            winner = np.max([
                                dp_mv[seqi,-3,0] + (aec), dp_mv[seqi-1,-1,0] + (acc)])
                            
                            slices = ((seqi, -3, 0), (seqi-1, -1, 0))
                            winnerarg = np.argmax([
                                np.max(dp_mv[seqi,-3,0] + (aec)), np.max(dp_mv[seqi-1,-1,0] + (acc))
                            ])
                            #print("UH",winnerarg )
                            btrace[seqi,-1,0] = slices[winnerarg]
                        dp_mv[seqi,-1,0] = winner
                        #b
                        if alg_base == "forward":
                            winner = np.logaddexp2(
                            dp_mv[seqi,0,0] + (anb), dp_mv[seqi,-2,0] + (ajb))
                        else:
                            winner = np.max([
                                dp_mv[seqi,0,0] + (anb), dp_mv[seqi,-2,0] + (ajb )
                            ])
                            
                            slices = ((seqi, 0, 0), (seqi, -2, 0))
                            winnerarg = np.argmax([
                                np.max(dp_mv[seqi,0,0] + (anb + np.zeros(self.leng))), np.max(dp_mv[seqi,-2,0] + (ajb + np.zeros(self.leng)))
                            ])
                            btrace[seqi,1,0] = slices[winnerarg]
                        dp_mv[seqi,1,0] = winner
                        
                    
                    result = (dp_mv[-1,-1,0] + (act)) - (len(seq)*((arr)) + np.log2(1-exp2arr))
                    #print(id_seq, result)
                    if alg_base == "viterbi":
                        going = (-1,-1,0)
                        curr = (-1,-1,0)
                        envelop = []
                        holder = []
                        envelop_res = []
                        enve_check = False
                        while going != (1,0,0):
                            curr = btrace[going]
                            #print("LESSGO", curr)
                            #print(curr)
                            if curr[2] == 0:
                                if curr[1] > 3 and curr[1] < len(dp_mv[1])-3:
                                    holder.append("M")
                                    if enve_check == True:
                                        envelop.append(curr[1])
                                        envelop_res.append(curr[0])
                                        enve_check = False
                                elif curr[1] == 0:
                                    holder.append("N")
                                elif curr[1] == 1:
                                    holder.append("B")
                                    envelop.append(going[1])
                                    #print("fghvkytvvgkv",curr[1])
                                    envelop_res.append(curr[0])
                                elif curr[1] == -3:
                                    holder.append("E")
                                    enve_check = True

                                elif curr[1] == -2:
                                    holder.append("J")
                                elif curr[1] == -1:
                                    holder.append("C")
                            #elif curr[2] == 1:
                            #    state_list.append("I")
                            elif curr[2] == 2:
                                holder.append("D")
                                if enve_check == True:
                                        envelop.append(curr[1])
                                        envelop_res.append(curr[0])
                                        enve_check = False
                            elif curr[2] == 3:
                                holder.append("Ö")
                                if enve_check == True:
                                        envelop.append(curr[1])
                                        envelop_res.append(curr[0])
                                        enve_check = False
                            going = curr


                        full_state_list.append(holder)
                        #print(full_state_list)
                        envelop = [(envelop[i]-4, envelop[i + 1]-4) for i in range(0, len(envelop), 2)]
                        envelop_res = [(envelop_res[i]-1, envelop_res[i + 1]-1) for i in range(0, len(envelop_res), 2)]
                        envelop = [tuple(reversed(t)) for t in envelop]
                        envelop_res = [tuple(reversed(t)) for t in envelop_res]
                        elements_to_remove = ["N", "C", "B", "E"]
                        state_list = [element for element in holder if element not in elements_to_remove]
                        state_list.reverse()
                        if "J" in state_list:
                            Jind = state_list.index("J")

                            state_list = [state_list[:Jind], state_list[Jind+1:]]
                        else:
                            state_list = [state_list]

                        


                        #print("LESSSGO",state_list)

                        #print(state_list)
                        #print(envelop)
                        #print(envelop_res)
                        #print(seq)


                        elements_to_remove = ["J"]
                        for ind, bruh in enumerate(state_list):
                            jcount = bruh.count("J")
                            holder = [element for element in bruh if element not in elements_to_remove]
                            state_list[ind] = holder
                        envelop_res.reverse()
                        envelop.reverse()


                            

                            #print(envelop[ind][0],envelop[ind][1])
                            #print(envelop_res[ind][0],envelop_res[ind][1])
                            #print(seq[envelop_res[ind][0]:envelop_res[ind][1]])

                        #print(full_state_list)
                        #print(state_list)
                        #print(envelop_res)
                        for ind2, char in enumerate(state_list):
                            #print(char)
                            i=0
                            holder2 = []
                            empty_seq = [" "] * self.leng
                            for charchar in char: #I have no ide WHY I need to have 3 forloops but am stupid mode rizzler gyaaaaat hunter so let it be
                                empty_seq = [" "] * (self.leng)
                            # print("SUG MIN KUK VRE",len(state_list), len(seq))
                                #print("HALLÅ",charchar)
                                #print(seq)
                                #print(ind2)
                                if charchar == "M":
                                    #print(envelop_res[ind2][0] + i)
                                    if ind2 == 0:
                                        #print(i)
                                        holder2.append(seq[envelop_res[ind2][0]-jcount + i])
                                    else:
                                        holder2.append(seq[envelop_res[ind2][0] + i])  

                                elif charchar == "D":
                                    holder2.append("-")
                                    i -= 1
                                    
                                elif charchar == "Ö":
                                    holder2.append(seq[envelop_res[ind2][0] + i].lower())
                                    holder2.append(seq[envelop_res[ind2][0] + i+1].lower())
                                    i += 1
                                i += 1
                            #print(holder2)

                            
                            empty_seq[envelop[ind2][0]:envelop[ind2][1]] = holder2

                            #print("PLS PLS PLS", empty_seq)
                            if result > -7:
                                modified_list_mspepalign.append(empty_seq)
                    #print(result)






                    





                        



                        
                    
                        



                #def wrapper():
                    #compute_scorev(self, ir, iir, alg_base, inverse_mode)

                # Time your function
                #execution_time = timeit.timeit(wrapper, number=1)
                #print(execution_time)
                compute_scorev(self, ir, iir, alg_base, inverse_mode)




                




                
                
                def compute_score(self, alg_base = "forward", inverse_mode = False):
                    def find_index(arr: 'numpy.ndarray', value: float) -> int:
                        for i, element in enumerate(arr):
                            if element == value:
                                return i
                        return None 
                    for seqi in range(2, dp_m.shape[0]):
                        #print(self.symbols[self.symb_index[seqi-1]])
                        em_rowind = 0
                        inem_rowin = 1
                        for statej in range(4, dp_m.shape[1]-3):
                            
                            #because of how MM state cant exist when there is only one residue left OR when we are evaluating the last state
                            if inverse_mode == True:
                                if seqi == dp_m.shape[0]-1 or statej == dp_m.shape[1]-4:
                                    ir = -np.inf
                                    iir = 0
                                else:
                                    ir = 0.02564102564102564
                                                                #disregard the wall of comment above, the new value is based on the frequency of WRONG residue times the fraction of the errors that are within this type
                                            
                                    iir = 1-ir
                                    ir, iir = math.log2(ir), math.log2(iir) 
                            else:
                                ir = -np.inf
                                iir = 0

                            #m
                            T = dp_m[seqi-1,statej-1]["log-odds M"] + (self.amm[em_rowind]  + iir)
                            Q = dp_m[seqi-1,statej-1]["log-odds I"] + (self.adm[em_rowind] + iir)
                            S = dp_m[seqi-1,statej-1]["log-odds D"] + (self.adm[em_rowind]  + iir)
                            P = dp_m[seqi-1,1]["log-odds"] + ( abm + iir)
                            R = dp_m[seqi-2,statej-2]["log-odds /MM/"] + (self.amm[em_rowind]  + iir)
                            comb_array = np.array([T,Q,S,P,R])
                            if alg_base == "forward":
                                log_tot = np.logaddexp2.reduce([P,Q,T,S,R])
                            elif alg_base == "viterbi":
                                log_tot = max(comb_array) #reduce(np.maximum, comb_array)
                                prev_ind = find_index(comb_array,log_tot)#max(range(len(comb_array)), key=lambda i: comb_array[i])
                                #print(prev_ind)
                                prev_state = ["M","I","D","B","/MM/"][prev_ind]
                                #print(prev_state)
                                prev_pos = [[seqi-1,statej-1], [seqi-1,statej-1], [seqi-1,statej-1], [seqi-1,1], [seqi-2,statej-2]][prev_ind]
                                dp_m[seqi,statej]["prev M"] = (prev_state, prev_pos)
                                #print(dp_m[seqi,statej]["prev"])
                            dp_m[seqi,statej]["log-odds M"] = log_tot + self.emM[em_rowind, self.symb_index[seqi-2]] - self.nullem0[self.symb_index[seqi-2]]
                            #print(self.emM[em_rowind, self.symb_index[seqi-2]] - self.nullem0[self.symb_index[seqi-2]])
                            

    #kolla med käll, osäker om jag borde applicera irr här eller om det räcker med att transition probs in till MM och M är behandlade och därmed blir summan av alla probs till 1 fortfarande
    #uses the transition prob of the outer MM (that is, if it is MM1 is it a combination of M1 and M2, M2 is the other one)

                            #i
                            P = dp_m[seqi-1,statej]["log-odds M"] + (self.ami[em_rowind])
                            Q = dp_m[seqi-1,statej]["log-odds I"] + (self.aii[em_rowind])
                            R = dp_m[seqi-1,statej-1]["log-odds /MM/"] + (self.ami[em_rowind]) #This is because Ik depends on the previous MM(k-1) state
                            #temp = np.exp2(dp_m[seqi-1,statej]["log-odds M"]) * self.ami[em_rowind-1] + np.exp2(dp_m[seqi-1,statej]["log-odds I"]) * self.aii[em_rowind-1]
                            comb_array = [P,Q,R]
                            if alg_base == "forward":    
                                PQ = np.logaddexp2.reduce([P,Q,R])
                            elif alg_base == "viterbi":
                                PQ = max(comb_array)
                                prev_ind = find_index(comb_array,PQ)
                                prev_state = ["M","I","/MM/"][prev_ind]
                                prev_pos = [[seqi-1,statej], [seqi-1,statej],[seqi-1,statej-1]][prev_ind]
                                dp_m[seqi,statej]["prev I"] = (prev_state, prev_pos)

                            #print(PQ)
                            dp_m[seqi,statej]["log-odds I"] = PQ + self.inemM[inem_rowin, self.symb_index[seqi-2]] - self.nullem0[self.symb_index[seqi-2]]

                            #d
                            P = dp_m[seqi,statej-1]["log-odds M"] + (self.amd[em_rowind])
                            Q = dp_m[seqi,statej-1]["log-odds D"] + (self.add[em_rowind])
                            R = dp_m[seqi,statej-2]["log-odds /MM/"] + (self.amd[em_rowind]) #Easier to understand if one look at the graph diagram but transitions to delete from MM is a jump of 2 states (k) in the diagram
                            comb_array = [P,Q,R]
                            if alg_base == "forward":    
                                PQ = np.logaddexp2.reduce([P,Q,R])
                            elif alg_base == "viterbi":
                                PQ = max(comb_array)
                                prev_ind = find_index(comb_array,PQ)
                                prev_state = ["M","D","/MM/"][prev_ind]
                                prev_pos = [[seqi,statej-1], [seqi,statej-1],[seqi,statej-2]][prev_ind]
                                dp_m[seqi,statej]["prev D"] = (prev_state, prev_pos)
                            dp_m[seqi,statej]["log-odds D"] = PQ 


                            #mm, 
                            T = dp_m[seqi-1,statej-1]["log-odds M"] + (self.amm[em_rowind] + ir)
                            Q = dp_m[seqi-1,statej-1]["log-odds I"] + (self.aim[em_rowind] + ir)
                            S = dp_m[seqi-1,statej-1]["log-odds D"] + (self.adm[em_rowind] + ir)
                            P = dp_m[seqi-1,1]["log-odds"] + (abm + ir)
                            R = dp_m[seqi-2,statej-2]["log-odds /MM/"] + (self.amm[em_rowind] + ir)
                            comb_array = np.array([T,Q,S,P,R]) 
                            if alg_base == "forward":
                                log_tot = np.logaddexp2.reduce([P,Q,T,S,R])
                            elif alg_base == "viterbi":
                                log_tot = max(comb_array) #reduce(np.maximum, comb_array)
                                prev_ind = find_index(comb_array,log_tot)#max(range(len(comb_array)), key=lambda i: comb_array[i])
                                
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
                                dp_m[seqi,statej]["log-odds /MM/"] = log_tot + (self.emM[em_rowind, self.symb_index[seqi-2]] - self.nullem0[self.symb_index[seqi-2]] + self.emM[em_rowind+1, self.symb_index[seqi-1]] - self.nullem0[self.symb_index[seqi-1]] + self.emM[em_rowind, self.symb_index[seqi-1]] - self.nullem0[self.symb_index[seqi-1]] + self.emM[em_rowind+1, self.symb_index[seqi-2]] - self.nullem0[self.symb_index[seqi-2]]) - divide_two
                                #print((self.emM[em_rowind, self.symb_index[seqi-2]] - self.nullem0[self.symb_index[seqi-2]] + self.emM[em_rowind+1, self.symb_index[seqi-1]] - self.nullem0[self.symb_index[seqi-1]] + self.emM[em_rowind, self.symb_index[seqi-1]] - self.nullem0[self.symb_index[seqi-1]] + self.emM[em_rowind+1, self.symb_index[seqi-2]] - self.nullem0[self.symb_index[seqi-2]]) - divide_two)
                                




                            em_rowind += 1
                            inem_rowin += 1 
                            
                        #e
                        holdit = []
                        state_remeber = []
                        stateind_remember = []
                        for statej in range(4, dp_m.shape[1]-3):
                            holdit.append(dp_m[seqi,statej]["log-odds M"] + (ame))
                            holdit.append(dp_m[seqi,statej]["log-odds D"] + (ade))
                            holdit.append(dp_m[seqi,statej]["log-odds /MM/"] + (ame)) #assuming that we may end at MM
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
                        dp_m[seqi,0]["log-odds"] = dp_m[seqi-1,0]["log-odds"] + (ann) #np.log2(np.exp2(dp_m[seqi-1,0]["log-odds"]) * ann)
                        dp_m[seqi,0]["prev"] = ("N", [seqi-1,0])

                        #j
                        P = dp_m[seqi,-3]["log-odds"] + (aej)
                        Q = dp_m[seqi-1,-2]["log-odds"] + (ajj)
                        if alg_base == "forward":
                            PQ = np.logaddexp2(P,Q)
                        elif alg_base == "viterbi":
                            PQ, prev_ind = reduce(np.maximum, np.array([P,Q])), np.argmax([P,Q])
                            prev_state = ["E", "J"][prev_ind]
                            prev_pos = [[seqi,-3], [seqi-1,-2]][prev_ind]
                            dp_m[seqi,-2]["prev"] = (prev_state, prev_pos)
                        dp_m[seqi,-2]["log-odds"] = PQ
                        
                        #c
                        P = dp_m[seqi,-3]["log-odds"] + (aec)
                        Q = dp_m[seqi-1,-1]["log-odds"] + (acc)
                        if alg_base == "forward":
                            PQ = np.logaddexp2(P,Q)
                        elif alg_base == "viterbi":
                            PQ, prev_ind = reduce(np.maximum, [P,Q]), np.argmax([P,Q])
                            prev_state = ["E", "C"][prev_ind]
                            prev_pos = [[seqi,-3], [seqi-1,-1]][prev_ind]
                            dp_m[seqi,-1]["prev"] = (prev_state, prev_pos)

                        dp_m[seqi,-1]["log-odds"] = PQ

                        #b
                        P = dp_m[seqi,0]["log-odds"] + (anb)
                        Q = dp_m[seqi,-2]["log-odds"] + (ajb)
                        if alg_base == "forward":
                            PQ = np.logaddexp2(P,Q)
                        elif alg_base == "viterbi":
                            PQ, prev_ind = reduce(np.maximum, [P,Q]), np.argmax([P,Q])
                            prev_state = ["N", "J"][prev_ind]
                            prev_pos = [[seqi,0], [seqi,-2]][prev_ind]
                            dp_m[seqi,1]["prev"] = (prev_state, prev_pos)

                        dp_m[seqi,1]["log-odds"] = PQ
                    return dp_m[-1,-1]["log-odds"]
                #computed_score = compute_score(self, alg_base, inverse_mode)  
                # def wrapper1():
                #     compute_score(self, alg_base, inverse_mode)

                # ## Time your function
                # #execution_time = timeit.timeit(wrapper1, number=10000)
                # #print("sämre?", execution_time)              
                # #break
                # #print(id_seq, (dp_m[-1,-1]["log-odds"] + np.log2(act) - (np.logaddexp2(-len(seq)*np.log2(arr), np.log2(1-arr)))))
            
                # result = id_seq + "\n"  # Start building the result string with id_seq
                # result += str((computed_score + (act)) - (len(seq)*(arr) + np.log2(1-np.exp2(arr)))) + "\n"  # Append the computation result
                # print(result)
                # #print(computed_score)
                # bitscore = (dp_m[-1,-1]["log-odds"] + (act)) - (len(seq)*(arr) + np.log2(1-np.exp2(arr)))
                # if alg_base == "viterbi" and consensus_alignment == True and bitscore >= -1000:
                #     path = []
                #     path.append(dp_m[-1,-1]["prev"])
                #     while path[-1] is not None:
                #         if path[-1][0] in ["M", "I", "D", "/MM/"]:
                #             if path[-1][0] == "M":
                #                 path.append(dp_m[path[-1][1][0], path[-1][1][1]]["prev M"])
                #             elif path[-1][0] == "I":
                #                 path.append(dp_m[path[-1][1][0], path[-1][1][1]]["prev I"])
                #             elif path[-1][0] == "D":
                #                 path.append(dp_m[path[-1][1][0], path[-1][1][1]]["prev D"])
                #             elif path[-1][0] == "/MM/":
                #                 path.append(dp_m[path[-1][1][0], path[-1][1][1]]["prev /MM/"])
                #         else:
                #             path.append(dp_m[path[-1][1][0], path[-1][1][1]]["prev"])
                #     #print(path)
                #     capital_letters_only = ["S" if item is None else item[0] for item in path]
                #     capital_letters_only.reverse()
                #     print(capital_letters_only)
                #     path.reverse()
                #     print(path)
                #     capital_letters_only = ''.join(capital_letters_only)
                #     capital_letters_only = capital_letters_only.replace("/MM/", "ÖÖ")
                #     result += capital_letters_only + " "  # Append the computed result
                #     #we need to add a functionallity to align the optimum states to the consesnsus structure of the profile turn them into the corresponding residues
                #     #consensus sequences is already parsed, contained in self.consseq
                #     #we need to know where the found optimum states are located in the model
                #     #we also need to carry over the functionalitty to pinpoint inverted residues
                #     aligned_seq = []
                #     previous_char = None
                #     envelop_list = []
                #     last_MM = None
                #     for index, element in enumerate(path):
                #         if element == None:
                #             aligned_seq.append(False)
                #         elif element[0] == 'N':
                #             if previous_char == 'N':
                #                 aligned_seq.append(True)
                #             else:
                #                 aligned_seq.append(False)
                #         elif element[0] == "B":
                #             aligned_seq.append(False)
                #             envelop_list.append(index)
                #         elif element[0] == "M":
                #             aligned_seq.append(True)
                #             last_MM = None
                #         elif element[0] == "I":
                #             aligned_seq.append(True)
                #             last_MM = None
                #         elif element[0] == "D":
                #             aligned_seq.append(True)
                #             last_MM = None
                #         elif element[0] == "/MM/":
                #             aligned_seq.append(True)
                #             last_MM = True
                #         elif element[0] == "E":
                #             aligned_seq.append(False)
                #             envelop_list.append(index)
                #         elif element[0] == 'J':
                #             if previous_char == 'J':
                #                 aligned_seq.append(True)
                #             else:
                #                 aligned_seq.append(False)
                #         previous_char = element[0] if element is not None else element
                #     pairs = []
                #     for i in range(0, len(envelop_list), 2):
                #         if i + 1 < len(envelop_list):
                #             pairs.append((envelop_list[i], envelop_list[i + 1]))
                #     #print(pairs)
                #     intervals = []
                #     model_interval = []
                #     seq_interval = []
                #     for envelop in pairs:
                #         intervals.append((path[envelop[0]+1][1],path[envelop[1]-1][1]))
                #     #print(intervals)
                #     add_residue = 0
                #     for ind, inter in enumerate(intervals):
                #         if last_MM == True and len(intervals) - 1 == ind:
                #             add_residue = 1
                #         model_interval.append((inter[0][1],inter[1][1]))
                #         seq_interval.append((inter[0][0],inter[1][0] + add_residue))
                #         add_residue = 0
                #     #print(seq_interval)
                #     #print(model_interval)
                    
                #     domain_list = []
                #     aligned_list = []
                #     sub_seq = ""
                #     for inter in model_interval:
                #         domain_list.append(self.consseq[inter[0]-4:inter[1]-3])
                #     for inter in seq_interval:
                #         for i in range(inter[0]-2, inter[1]-1):
                #             sub_seq += seq[i]
                #         aligned_list.append(sub_seq)
                #         sub_seq = ""
                #     #print(domain_list, len(domain_list[0]))
                #     #print(aligned_list, len(aligned_list[0]))

                #     ## Initialize list to store extracted substrings
                #     substring_list = []

                #     # Initialize flag to indicate whether to start storing substring
                #     store_substring = False
                #     #print("WTF", capital_letters_only)
                #     # Iterate over characters in the input string
                #     for char in capital_letters_only:
                #         # If current character is "B", set flag to start storing substring
                #         if char == "B":
                #             store_substring = True
                #             substring = ""
                #         # If current character is "E", set flag to stop storing substring
                #         elif char == "E":
                #             store_substring = False
                #             substring_list.append(substring)
                #         # If flag is True, store current character in substring
                #         elif store_substring:
                #             substring += char
                #     #print(capital_letters_only)
                #     #print(substring_list)
                #     modified_list = []

                #     for aligned_string, substring_string in zip(aligned_list, substring_list):
                #         modified_string = ""
                #         substring_iterator = iter(substring_string)
                #         #print("hej",aligned_string, substring_string)

                #         stopper = 0  # Initialize the stopper flag
                #         aligned_iterator = iter(aligned_string)
                #         deleter = 0
                #         while True:
                #             try:
                #                 if deleter == 0:
                #                     aligned_char = next(aligned_iterator)
                #                 else:
                #                     #print("jag e bög", aligned_char)
                #                     pass
                                
                #                 substring_char = next(substring_iterator)
                #                 if stopper == 1:
                #                     #print("big bög", substring_char)
                #                     pass
                #             except StopIteration:
                #                 break

                #             #if stopper == 1:  # Check if stopper flag is set
                #             #    stopper = 0  # Reset the stopper flag
                #             #    continue  # Stay on the same iteration if stopper is 1
                #             if deleter == 1:
                #                 deleter = 0
                #             #    continue

                #             if substring_char == "D":
                #                 modified_string += "-" 
                #                 deleter = 1
                #                 #modified_string += aligned_char  # Include aligned_char # Add "-"
                #             elif substring_char == "M":
                #                 stopper = 0
                #                 modified_string += aligned_char.upper()
                #             elif substring_char == "I":
                #                 stopper = 0
                #                 #modified_string += aligned_char.lower()
                #             elif substring_char == "Ö":
                #                 #stopper = 1
                #                 modified_string += aligned_char.lower()
                #                 #modified_string += aligned_char
                #         #print("DÅ", modified_string)
                #         modified_list.append(modified_string)

                #     #print(modified_list)



                            


                                





                            




                    
                        

                #     result += "\n"  # Start a new line for the next result
                #     #print(result)
                #     if alg_base == "forward":
                #         print(result)
                #     if bitscore >= -1000 and alg_base == "viterbi":
                #         full_state_list.append(capital_letters_only)        
                #         state_list.append(substring_string)
                
                #     if alg_base == "viterbi" and consensus_alignment == True and mspeptide_alignment == False and bitscore >= -1000:
                #         full_state_list.append(capital_letters_only)        
                #         state_list.append(substring_string)
                #         for consensus, mod_ali in zip(domain_list, modified_list):
                #             alignment_str = ""
                #             for consensus_char, mod_ali_char in zip(consensus, mod_ali):
                #                 consensus_char = consensus_char.lower()  # Convert to lowercase
                #                 mod_ali_char = mod_ali_char.lower()      # Convert to lowercase
                #                 if consensus_char == mod_ali_char:
                #                     alignment_str += "\033[92m" + consensus_char + "\033[0m"  # Green for conserved residues
                #                 elif consensus_char == "-" or mod_ali_char == "-":
                #                     alignment_str += "\033[91m" + "*" + "\033[0m"  # Red asterisk for gaps
                #                 elif mod_ali_char == "/":
                #                     alignment_str += "\033[93m" + "/" + "\033[0m"  # Yellow for "/"
                #                 else:
                #                     alignment_str += mod_ali_char  # Display modified alignment
                #             print("Consensus:       ", consensus)
                #             print("Target sequence: ", alignment_str)
                #             print("//////////////////////")
                #     #f.write(result)
                #         #print(len(seq))
                #         #print(len(capital_letters_only))
                # if alg_base == "viterbi" and consensus_alignment == True and mspeptide_alignment == True and bitscore >= -1000:
                #     for i, mod_ali in enumerate(modified_list):
                #         empty_seq = list(" " * len(self.consseq))
                #         mod_ali_list = list(mod_ali)
                #         empty_seq[model_interval[i][0]-4:model_interval[i][1]-3] = mod_ali_list
                #         modified_list_mspepalign.append("".join(empty_seq))
                #         id_list.append(id_seq)
                    
            
                progress_bar.update(1)
        progress_bar.close()
        def most_occuring_letter(sequences):
                # Initialize the resulting sequence
                result_sequence = ''
                # Iterate over each position in the sequences
                for i in range(len(sequences[0])):
                    try:
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
                    except IndexError:
                        print("WTF",sequence, len(sequence), len(self.consseq))
                return result_sequence

        ref_seq = most_occuring_letter(modified_list_mspepalign)

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
        #print(modified_list_mspepalign)
        #print(modified_list_mspepalign)

        def sort_list2(list1):
            # Transpose list1 to get columns
            columns = zip(*list1)
            
            # Initialize progress bar
            total_iterations = len(list1)
            progress_bar = tqdm(total=total_iterations, desc="Sorting Columns")
            
            # Sort each column separately
            sorted_columns = []
            for col in columns:
                non_whitespace = [x for x in col if x != " "]
                whitespace = [x for x in col if x == " "]
                sorted_col = non_whitespace + whitespace
                sorted_columns.append(sorted_col)
                
                # Update progress bar
                progress_bar.update(1)
            
            # Close progress bar
            progress_bar.close()

            # Transpose back to rows
            sorted_list1 = [list(row) for row in zip(*sorted_columns)]
            
            # Remove elements that are just lists filled with whitespace
            sorted_list1 = [row for row in sorted_list1 if set(row) != {" "}]
            
            # Calculate gaps
            gaps = sorted_list1[1].count(" ")
            sorted_list1[0] = "".join(sorted_list1[0])

            return sorted_list1, gaps
        
        modified_list_mspepalign, gaps = sort_list2(modified_list_mspepalign)

        
        def assambly_time(consseq, modified_list_mspepalign):
            progress_bar2 = tqdm(total=len(modified_list_mspepalign), desc="Sequences visually aligned", position=0, miniters=0)
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
                seq = modified_list_mspepalign[i][:len(consseq)]
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
                    #print("????",seq[start_index:end_index+1], ([" "] * (end_index + 1 - start_index)))
                    if seq[start_index:end_index+1] == (" " * (end_index + 1 - start_index)) and j not in taken_list: #seq_comp[start_index:end_index] not in merged:
                        #print("LESSSGo")
                        modified_seq = seq[:start_index] + seq_comp[start_index:end_index+1] + seq[end_index+1:len(consseq)]
                        seq = modified_seq
                        merged.append(seq_comp[start_index:end_index+1])
                        taken_list.append(j)
                        estimation_peptides_cov += 1
                    j += 1
                    modified_list_mspepalign[i] = seq
                    #print(len(seq))
                    if gaps_check == True:
                        if i == 1 and " " not in seq: #in the case full coverage is achived just from the first "assembly" of aligned peptides, REALLY unlikely
                            full_peptide_cov = len(set(taken_list)) + i #amount of peptide sequences that have been appended plus the amount of peptides that these peptides have been appended to
                            gaps_check == False
                        elif i == 1:
                            gaps_prev = find_blank_spaces(seq)
                            gaps = compare_lists(find_blank_spaces(" " * len(consseq)), gaps_prev)
                        else:
                            gaps_curr = find_blank_spaces(seq)
                            gaps = compare_lists(gaps_curr, gaps_prev) #the amount of gaps present so far
                            if len(gaps) == 0:
                                full_peptide_cov = len(set(taken_list)) + i
                                estimation_peptides_cov = i
                                gaps_check = False
                            else:
                                gaps_prev = gaps
                progress_bar2.update(1)
            
                        
                i += 1

            return gaps, estimation_peptides_cov, full_peptide_cov, taken_list
        #gaps, estimation_peptides_cov, full_peptide_cov, taken_list = assambly_time(self.consseq, modified_list_mspepalign)

        


        #Remove elements from modified_list_mspepalign using indices in merged
        #for index in sorted(taken_list, reverse=True):
        #    if 0 <= index < len(modified_list_mspepalign):
        #        del modified_list_mspepalign[index]
        

        # Remove duplicates from modified_list_mspepalign while preserving order
        #modified_list_mspepalign = list(dict.fromkeys(modified_list_mspepalign))
        
        
        if alg_base == "viterbi" and consensus_alignment == True and mspeptide_alignment == True:
            # ANSI escape code for blue color
            blue_color = "\033[94m"
            # ANSI escape code to reset color
            reset_color = "\033[0m"
            
            # Print table header
            def print_header():
                ID = 'ID'
                Alignment = 'Alignment'
                print("{:<20} {}".format(ID, Alignment))

            # Call the function
            print_header()
            
            
            
            
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
            
            # Print table header
            def print_header():
                ID = 'ID'
                Alignment = 'Alignment'
                print("{:<20} {}".format(ID, Alignment))

            def print_result_sequence(ref_seq):
                print("{:<26} {}".format('resulting sequence:', ref_seq))

            # Call the functions
            print_header()
            print_result_sequence(ref_seq)
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

            if gaps != 0:
                #print("Estimated amount of peptides needed for full coverages:", estimation_peptides_cov)
                print("Coverages to the model estimated to (%):", (len(self.consseq) - gaps) / len(self.consseq))
                print("Procentage of mismatched residues in the peptide alignment:", (mismatch/sum(len(peptide) for peptide in modified_list_mspepalign)))
            else:
                print("Coverages to the model estimated to (%):", (len(self.consseq) - gaps) / len(self.consseq))
                print("Procentage of mismatched residues in the peptide alignment:", (mismatch/sum(len(peptide) for peptide in modified_list_mspepalign)))

                print(mismatch, sum(len(peptide) for peptide in modified_list_mspepalign), len(self.consseq), gaps)
            
        # Open an HTML file in write mode
            # Open an HTML file in write mode
# Open an HTML file in write mode
            with open("output.html", "w") as html_file:
                # Write the HTML content
                html_file.write("<html>\n<head>\n<title>Alignment Results</title>\n</head>\n<body>\n")

                # Write the table header
                html_file.write("<table border='1'>\n<tr><th>ID</th><th>Alignment</th></tr>\n")

                # Write the resulting sequence row
                html_file.write("<tr><td>Resulting sequence</td><td><pre>")
                html_file.write(ref_seq + "</pre></td></tr>\n")

                # Write the consensus row
                html_file.write("<tr><td>Conensus</td><td><pre>")
                html_file.write("<font color='blue'>" + modified_list_mspepalign[0] + "</font></pre></td></tr>\n")

                # Write the alignment rows
                for i, mod_ali in enumerate(modified_list_mspepalign[1:], 1):
                    highlighted_mod_ali = ''
                    for mod_char, ref_char in zip(mod_ali, ref_seq):
                        if mod_char == ref_char or mod_char == ' ':
                            highlighted_mod_ali += mod_char
                        else:
                            # Highlight differences in red
                            highlighted_mod_ali += "<font color='red'>" + mod_char + "</font>"
                    html_file.write("<tr><td>Alignment</td><td><pre>")
                    html_file.write(highlighted_mod_ali + "</pre></td></tr>\n")

                # Write additional information if available
                if gaps != 0:
                    #html_file.write("<tr><td>Estimated amount of peptides needed for full coverage</td><td>")
                    #html_file.write(str(estimation_peptides_cov) + "</td></tr>\n")
                    html_file.write("<tr><td>Coverage to the model estimated to (%)</td><td>")
                    html_file.write(str((len(self.consseq) - gaps) / len(self.consseq)) + "</td></tr>\n")
                    html_file.write("<tr><td>Percentage of mismatched residues in the peptide alignment</td><td>")
                    html_file.write(str((mismatch / sum(len(peptide) for peptide in modified_list_mspepalign))) + "</td></tr>\n")
                else:
                    html_file.write("<tr><td>Coverage to the model estimated to (%)</td><td>")
                    html_file.write(str((len(self.consseq) - gaps) / len(self.consseq)) + "</td></tr>\n")
                    html_file.write("<tr><td>Percentage of mismatched residues in the peptide alignment</td><td>")
                    html_file.write(str((mismatch / sum(len(peptide) for peptide in modified_list_mspepalign))) + "</td></tr>\n")

                # Write additional information
                html_file.write("<tr><td>Estimated procentage of peptide residues predicted to be homologous to the model:</td><td>")
                html_file.write(str(count_letters(full_state_list, ["M","I","D","Ö"])) + "</td></tr>\n")

                html_file.write("<tr><td>Estimated procentage of peptide residues predicted to be non-homologous to the model, N-terminal side:</td><td>")
                html_file.write(str(count_letters(full_state_list, ["N"])) + "</td></tr>\n")

                html_file.write("<tr><td>Estimated procentage of peptide residues predicted to be non-homologous to the model, C-terminal side:</td><td>")
                html_file.write(str(count_letters(full_state_list, ["C"])) + "</td></tr>\n")

                html_file.write("<tr><td>Estimated procentage of peptide residues predicted to encompass inversion uncertainty:</td><td>")
                html_file.write(str(count_letters(full_state_list, ["Ö"])) + "</td></tr>\n")


                # Close the table and HTML tags
                html_file.write("</table>\n</body>\n</html>")












            #Evaluating the coverage and estimating the best guess for the full protein sequence
            #We will estimate the frequencies of amino acids in our alignment, residues generated from MM state will count two corresponding residues to
                    

                                
                            





                
import profile
from line_profiler import LineProfiler
import timeit
                
test = Decoder("sequences.FASTA","super_hmm1.hmm")
#test.forward()
test.inverse(alg_base="viterbi", inverse_mode=False, consensus_alignment=True, mspeptide_alignment=True)
#profile.run('test.inverse(alg_base="viterbi", inverse_mode=True, consensus_alignment=True, mspeptide_alignment=True)')
#callable_func = lambda: test.inverse(alg_base="viterbi", inverse_mode=True, consensus_alignment=True, mspeptide_alignment=True)

# Measure the execution time of the function
#execution_time = timeit.timeit(callable_func, number=3)

# Print the average execution time

#profiler = LineProfiler()
#profiler.add_function(test.inverse)
#profiler.run('test.inverse(alg_base="viterbi", inverse_mode=True, consensus_alignment=True, mspeptide_alignment=True)')
#profiler.print_stats()
