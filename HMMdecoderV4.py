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
        self.amm = self.tranM[:,0] 
        self.ami = self.tranM[:,1] 
        self.amd = self.tranM[:,2]
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

            v_m = [[{}]*(len(seq)+2) for _ in range(self.leng+2)]
            v_i = [[{}]*(len(seq)+2) for _ in range(self.leng+2)]
            v_d = [[{}]*(len(seq)+2) for _ in range(self.leng+2)]

            #time to initiate!

            v_m[0][0] = {"log-odds": 0, "prev": None} #START, which is MATCH0
            for col in range(1, (len(seq)+1)):
                v_m[0][col]={"log-odds": np.inf, "prev": None} #the thing is, the first column is a fake amino acid, the rest of the match states cant be aligned to it, ever
            for row in (range(1, self.leng+1)):
                v_m[row][0]={"log-odds": np.inf, "prev": None} #can never have START and an emission

            for row in (range(0, self.leng+1)): #initiating the inserts chains from start
                v_i[row][0]={"log-odds": np.inf, "prev": None} #cant have an insert in the fake amino acid
            for col in range(1, (len(seq)+1)):
                    temp = [v_m[0][col-1]["log-odds"] + self.ami[0], v_i[0][col-1]["log-odds"] + self.aii[0]]
                    temp_state = ["M", "I"]
                    v_i[0][col]={"log-odds": np.min(temp), "prev": (temp_state[np.argmin(temp)], 0, col-1)} 
                

            for col in range(0, (len(seq)+1)):# initiating the delete chain from start
                v_d[0][col]={"log-odds": np.inf, "prev": None}
            for row in (range(1, self.leng+1)):
                temp = [v_m[row-1][0]["log-odds"] + self.amd[row-1], v_d[row-1][0]["log-odds"] + self.add[row-1]]
                temp_state = ["M", "D"]
                v_d[row][0]={"log-odds": np.min(temp), "prev": (temp_state[np.argmin(temp)], row, 0)}
            #now fill in all matrices, columnwise
            for col in range(1, (len(seq)+1)):
                for row in (range(1, self.leng+1)):
                    

                    #v_d
                    temp = [v_m[row-1][col]["log-odds"] + self.amd[row], v_d[row-1][col]["log-odds"] + self.add[row]]
                    v_d[row][col] = {"log-odds": np.min(temp), "prev": (["M", "D"][np.argmin(temp)], row-1, col)}

                    #v_i
                    temp = [v_m[row][col-1]["log-odds"] + self.ami[row], v_i[row][col-1]["log-odds"] + self.aii[row]]
                    v_i[row][col] = {"log-odds": np.min(temp), "prev": (["M", "I"][np.argmin(temp)], row, col-1)}

                    #v_m
                    temp = [v_m[row-1][col-1]["log-odds"] + self.amm[row], v_d[row-1][col-1]["log-odds"] + self.adm[row], v_i[row-1][col-1]["log-odds"] + self.aim[row]]
                    v_m[row][col] = {"log-odds": np.min(temp) + self.emM[row-1,self.symb_index[col-1]] - self.inemM[row,self.symb_index[col-1]], "prev": (["M", "D", "I"][np.argmin(temp)], row-1, col-1)}

            #finally, what we all have been waiting for, what move will take us to the end?
            temp = [v_m[-2][-2]["log-odds"] + self.amm[-1], v_d[-2][-2]["log-odds"] + self.adm[-1], v_i[-2][-2]["log-odds"] + self.aim[-1]]
            v_m[-1][-1] = {"log-odds": np.min(temp), "prev": (["M", "D", "I"][np.argmin(temp)], -2, -2)} #durbin et al said there is no emission here, so it be

            #easy backtrace
            #where from where did we reach the end?
            #print(v_m[-1][-1]["prev"])
            state, row, col = v_m[-1][-1]["prev"]
            score = v_m[-1][-1]["log-odds"]
            score = score / np.log(2)
            #print(score)
            #print(v_d[-2][-2])
            state_list = []
            while row != None and col != None:
                state_list.append(state)

                if state == "M":
                    if v_m[row][col]["prev"] == None:
                        break
                    state, row, col = v_m[row][col]["prev"]
                elif state == "I":
                    if v_m[row][col]["prev"] == None:
                        break
                    state, row, col = v_i[row][col]["prev"]
                elif state == "D":
                    if v_m[row][col]["prev"] == None:
                        break
                    state, row, col = v_d[row][col]["prev"]

            #state_list.reverse()
            #print(state_list)

            #Lets be fancy and make an alignment
            #I did not really understand with what HMMER means with "inserts are unaligned" so I will just just do my best
            seq.reverse()
            align_seq = []
            emission_counter = 0
            #print(len(seq))
            #print(len(state_list))
            counter_uhm = 0
            for stuff1 in state_list:
                counter_uhm += 1
                if counter_uhm == len(state_list): #I have no clue why this but it just works
                    break
                if stuff1 == "D":
                    align_seq.append("-")
                elif stuff1 == "M":
                    align_seq.append(seq[emission_counter])
                    #print(align_seq)
                    emission_counter += 1
                elif stuff1 =="I":
                    align_seq.append(seq[emission_counter].lower())
                    emission_counter += 1
            align_seq.reverse()
            align_seq = "".join(align_seq)
            #print("here is the alignment!", align_seq)

            output_file_path = "viterbi alignment with score.txt"
            
            with open(output_file_path, 'a') as file:
                file.write(">" + id_seq + '\n' + align_seq + '\n' + str(score) + '\n')



        


        

        



                        

        #np.savetxt('please.txt', v_m, fmt='%9.6s', delimiter='\t')

            
            


test = Decoder("globins45.FA","globins4.hmm")
test.viterbi()
#print(test.sequences)


