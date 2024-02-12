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

        
        np.savetxt('tranM.txt', self.tranM, fmt='%8.6s', delimiter='\t')
        np.savetxt('emM.txt', self.emM, fmt='%8.6s', delimiter='\t')
        np.savetxt('imemM.txt', self.inemM, fmt='%8.6s', delimiter='\t')
        
        #keep in mind, inemM has one extra row, because of the state type INSERT has one extra state, 0. Same goes for the tranM, has one extra row (self.leng + 1)
        #I will remove the first row from tranM because it only contains how transition from M0 and transitions from I0 is and that is not relevant in the plan 7 model
        
        self.amm = self.tranM[:,0] 
        self.ami = self.tranM[:,1] 
        self.amd = self.tranM[:,2]
        #self.amd[-2] = np.inf #This might be used, it is really unclear but some sources states that the Plan7viterbi does NOT use the last delete state, as I understand it, it is enough to remove the first delete state in the core model so each search always matches one residue to match
        self.aim = self.tranM[:,3]
        self.aii = self.tranM[:,4]
        self.adm = self.tranM[:,5]
        self.add = self.tranM[:,6] #decided to seperate them (mm = match to match etc, a for transition as durbin does)

        

        self.states = []
        for i in range(1, self.leng+1):
            for j in ["M","D","I"]:
                self.states.append(j + str(i))

        self.states = ["IO"] + self.states
        self.states = ["D0"] + self.states #look at the comment below, in addition this state is actually nonexistant
        self.states = ["START"] + self.states #same as M0
        self.states = self.states + ["END"] #will not be used but needed to releate the index in the dp_matrix row with the state
        #print(self.states)

    def viterbi(self):
        for id_seq, stuff in self.sequences:
            #print("Alignment of", id_seq)
            seq = list(stuff)
            self.symb_index = [self.symbols.index(j) for i, j in enumerate(seq)] #easier usage later

            v_m = [[{}]*(self.leng+1) for _ in range(len(seq)+1)]
            v_i = [[{}]*(self.leng+1) for _ in range(len(seq)+1)]
            v_d = [[{}]*(self.leng+1) for _ in range(len(seq)+1)]
            v_e = [[{}]*(self.leng+1)]
            #v_s = [[{}]*(len(seq)+1)] I believe this one is actually not needed in the algorithm
            v_n = [[{}]*(len(seq)+1)]
            v_b = [[{}]*(len(seq)+1)]
            v_e = [[{}]*(len(seq)+1)]
            v_c = [[{}]*(len(seq)+1)]
            v_t = [[{}]*(len(seq)+1)]
            v_j = [[{}]*(len(seq)+1)] #used for multihit local model, will try to make this one work

            #time to initiate!
            #for i in range (1, len(seq)+1):
            #    v_s[i] = {"log-odds": np.inf, "prev": None}

            #v_s[0] = {"log-odds": 0, "prev": None}
            for i in range (0, len(seq)+1):

                v_n[0][i] = {"log-odds": np.inf, "prev": None}
                v_b[0][i] = {"log-odds": np.inf, "prev": None}
                v_b[0][i] = {"log-odds": np.inf, "prev": None}
                v_e[0][i] = {"log-odds": np.inf, "prev": None}
                v_c[0][i] = {"log-odds": np.inf, "prev": None}
                v_t[0][i] = {"log-odds": np.inf, "prev": None}
                v_j[0][i] = {"log-odds": np.inf, "prev": None}
            
            for col in range(0, self.leng+1):
                for row in range(0, len(seq)+1):
                    v_m[row][col] = {"log-odds": np.inf, "prev": None}
                    v_i[row][col] = {"log-odds": np.inf, "prev": None}
                    v_d[row][col] = {"log-odds": np.inf, "prev": None}
            #Initiation done of the various DP matrixes
            
            #now for some transition probabilities, in line with what sean R. EDDY said
            #negative log transforming it all 
            ann = ajj = acc = -np.log(len(seq)/(len(seq)+3))
            anb = act = ajb = -np.log(3/(len(seq)+3)) #?
            aej = aec = -np.log(0.5)
            abm = -np.log(2/((self.leng*(self.leng + 1))))
            ame = ade = -np.log(1)
            #these where not easy to find (for me because I am stupid) but if everything is right this should be enough to have a multi hit smith waterman style hmmsearch

            #I think I will for now not bother with the backtracking because I only want to see the final log odds score
            #just noticed that the v_n can be initiated beforehand so it will be done
            v_n[0][0] = {"log-odds": 0, "prev": "START"}
            v_b[0][0] = {"log-odds": 0 + anb, "prev": ("N,", 0, 0)}
            #for seqi in range(1, len(seq)+1):
                #print(seqi)
            #    temp = v_n[0][seqi-1]["log-odds"] + ann
            #    v_n[0][seqi] = {"log-odds": temp, "prev": ("N", seqi)}
            #print(len(v_m[0][:]))

            #ABDI NORMALIZERA MED qi INTE GJORT 
                #gjorde det nu, tog bort insert emission from match och från sig själv
            for seqi in range(1, len(seq)+1):
                #print("hej")
                for statej in range(1, self.leng+1):

                    
                    
                    #m
                    temp = [v_m[seqi-1][statej-1]["log-odds"] + self.amm[statej-1],
                            v_i[seqi-1][statej-1]["log-odds"] + self.aim[statej-1],
                            v_d[seqi-1][statej-1]["log-odds"] + self.adm[statej-1],
                            v_b[0][seqi-1]["log-odds"] + abm,
                            np.inf]
                    argtemp = ["M","I","D","B","sussybaka"]
                    v_m[seqi][statej] = {"log-odds": np.min(temp) + self.emM[statej-1,self.symb_index[seqi-1]] - self.inemM[statej,self.symb_index[seqi-1]], "prev": (argtemp[np.argmin(temp)], seqi-1, statej-1)}
                    #print(v_m[seqi][statej])
                    #i
                    temp = [v_m[seqi-1][statej]["log-odds"] + self.ami[statej-1],
                            v_i[seqi-1][statej]["log-odds"] + self.aii[statej-1],
                            np.inf]
                    argtemp = ["M","I","sussybaka"]
                    v_i[seqi][statej] = {"log-odds": np.min(temp), "prev": (argtemp[np.argmin(temp)], seqi-1, statej)}

                    #d
                    temp = [v_m[seqi][statej-1]["log-odds"] + self.amd[statej-1],
                            v_d[seqi][statej-1]["log-odds"] + self.add[statej-1],
                            np.inf]
                    argtemp = ["M","D","sussybaka"]
                    v_d[seqi][statej] = {"log-odds": np.min(temp), "prev": (argtemp[np.argmin(temp)], seqi, statej-1)}
                #now done with the inner model, now we iterate in the outer model
                


                #e #this one must be different from the rest because we need to pick out from all possible states and just the one with the lowest score
                #ugly because I finally understood that I have to evaluate all match and delete states for a given residue here, could not manage to do np.min here so I did this
                v_mbest = np.inf
                v_dbest = np.inf
                
                for bro in range(1, self.leng+1):
                    if v_mbest > v_m[seqi][bro]["log-odds"]:
                        v_mstatej = bro
                        v_mbest = v_m[seqi][bro]["log-odds"]

                    if v_dbest > v_d[seqi][bro]["log-odds"]:
                        v_dstatej = bro
                        v_dbest = v_d[seqi][bro]["log-odds"]

                temp = [v_mbest + ame,
                        v_dbest + ade,
                        np.inf]
                argtemp = ["M","D","sussybaka"]
                argtemp1 = [v_mstatej, v_dstatej]
                v_e[0][seqi] = {"log-odds": np.min(temp), "prev": (argtemp[np.argmin(temp)], seqi, argtemp1[np.argmin(temp)])}

                #n
                temp = [v_n[0][seqi-1]["log-odds"] + ann,
                        np.inf]
                argtemp = ["N","sussybaka"]
                #print(temp)
                v_n[0][seqi] = {"log-odds": np.min(temp), "prev": (argtemp[np.argmin(temp)], seqi-1, statej)}

                #j just like when calculating v_n, no emisson because we normilize with the background and the background is the average emission (???????) problem, if we normilize with the nullemission then the emisson for insertion is still present but Durbin explicitly said that the insertion emission should be cancled by the normalization
                temp = [v_e[0][seqi]["log-odds"] + aej,
                        v_j[0][seqi-1]["log-odds"] + ajj,
                        np.inf]
                argtemp = ["E","J","sussybaka"]
                argtemp1 = [seqi, seqi-1]
                v_j[0][seqi] = {"log-odds": np.min(temp), "prev": (argtemp[np.argmin(temp)], argtemp1[np.argmin(temp)], statej)}

                #c
                temp = [v_e[0][seqi]["log-odds"] + aec,
                        v_c[0][seqi-1]["log-odds"] + acc,
                        np.inf]
                argtemp = ["E","C","sussybaka"]
                argtemp1 = [seqi, seqi-1]
                v_c[0][seqi] = {"log-odds": np.min(temp), "prev": (argtemp[np.argmin(temp)], argtemp1[np.argmin(temp)], statej)}

                #b
                temp = [v_n[0][seqi]["log-odds"] + anb,
                        v_j[0][seqi]["log-odds"] + ajb,
                        np.inf]
                argtemp = ["N","J","sussybaka"]
                v_b[0][seqi] = {"log-odds": np.min(temp), "prev": (argtemp[np.argmin(temp)], seqi, statej)}


            
            #to even be able to compare to hmmer, we need to define our score as the same way they do. Eddy uses log2 and an actuall ODDS against the null
            #we construct a sequence genererated by the null hypothesis
            nullscore = 0
            arr = (len(seq))/(len(seq)+1) #the transition prob for going from R to R in the null model
            that_part = 1-arr
            that_part = -np.log(that_part)
            arr = -np.log(arr)
            print(that_part)
            #for i in seq:
            #    nullscore = nullscore + arr
            nullscore = arr * len(seq) #the nullscores corresponding to the transitions
            #okey, turns out that the null model is already implemented, all probabilities found in the hmm file are actually odds ratios so I believe we dont have to do anything
            #turn it back to normal prob
            #nullscore = np.exp(-nullscore)
            print("null", nullscore)
            print(v_m[1][130])







            #done! the log-odds ratio (against the null because we normed all emissions with the null emisson?) should be contained in the last element in v_c, we should be able to backtrack if we want by just referensing to the prev key
            print("The sequence id is ", id_seq, "with len", len(seq))
            print("dess log odds är", (v_c[0][-1]["log-odds"] + act - (nullscore + that_part))/-np.log(2))
            #print((v_c))
            #print(v_e)
            #print(v_m[1][130])
            #print(v_b[0][0])
            #print(v_n)
            #print(abm)
            #print(v_m[1][123])






                            
                





                

            




                
            


            




            
test = Decoder("Testse.txt","globins4.hmm")
test.viterbi()