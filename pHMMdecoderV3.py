import numpy as np
import pandas as pd

class Decoder:
    def __init__(self, sequence, profhmm):
        self.seq = list(sequence)
        self.symbols =  ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        self.symb_index = [self.symbols.index(j) for i, j in enumerate(sequence)] #easier usage later
        print(self.symb_index)
        #time the read from the hmm file and get emission and transition
        counter_compo = 0
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
        #keep in mind, inemM has one extra row, because of the state type INSERT has one extra state, 0. Same goes for the tranM, has one extra row (self.leng + 1), 

        self.states = []
        for i in range(1, self.leng+1):
            for j in ["M","D","I"]:
                self.states.append(j + str(i))
        self.states = ["I0"] + self.states
        self.states = ["D0"] + self.states
        self.states = ["START"] + self.states

    def viterbi(self):
        dp_matrix = np.full(shape=(self.leng * 3 + 4, len(self.seq) + 1), fill_value=np.inf) #filling out some extra states for the M(k+1) = END and also the I0, M0 = START and also the nonexistant D0

        backtrace = np.full(shape=(self.leng * 3 + 4, len(self.seq) + 1, 2), fill_value=[-2,-2]) #3D matrice, needed because both row and col indices are needed during the backtrace

        row_temp = 1
        #Initiate
        dp_matrix[0,0] = 0 #start is defined as having prob 1 (0 in natural log space)
        dp_matrix[4,0] = self.tranM[0,2]
        for row in range(7, dp_matrix.shape[0], 3): #initiating the delete chain from start (the probabilities here are independant from the others)
            dp_matrix[row,0] = dp_matrix[row-3,0] + self.tranM[row_temp,6] + 10
            backtrace[row,0] = [row-3,0]
            row_temp += 1
        dp_matrix[2,1] = self.tranM[0,1]
        for col in range(2, dp_matrix.shape[1]): #initiating the insert0 chain, independant from the others
            dp_matrix[2,col] = dp_matrix[2,col-1] + self.tranM[0,1]

        np.savetxt('please.txt', dp_matrix, fmt='%8.6s', delimiter='\t')
        dp_matrix[3,1] = self.tranM[0,0] + self.emM[0, self.symb_index[0]]

        row_temp = 1
        state_counter = 0
        for row in range(3, dp_matrix.shape[0]): #we start from row that corresponds to I0 state, skipping START and D0 states
            #problem, row indices in the dp matrix is a factor 3 (and plus 4) larger then the actuall real states, a line of code at the vary last increments a counter and resets when we reach self.leng
            for col in range(1,dp_matrix.shape[1]): #skipping the first row, just a gap used to access 0 probabilities for some transitions)
                if state_counter == 0: #we landed on V(ei)M
                    if col == 1 and row == 3:
                        continue #because thisone is alrady initiated
                    else:
                        temp = (dp_matrix[row-3,col-1] + self.tranM[row_temp,0], dp_matrix[row-2,col-1] + self.tranM[row_temp,5], dp_matrix[row-1,col-1] + self.tranM[row_temp,3]) #evaluating previous legal transitions
                        ind_temp = ([row-3,col-1], [row-2,col-1], [row-1,col-1])
                        dp_matrix[row,col] = np.min(temp) + self.emM[row_temp-1, self.symb_index[col-1]]
                        backtrace[row,col] = ind_temp[np.argmin(temp)]
                if state_counter == 1: #V(ei)D #something may be wrong here, deletions gets a really good score all the time
                        temp = (dp_matrix[row-4,col] + self.tranM[row_temp,2], dp_matrix[row-3,col] + self.tranM[row_temp,6]) #evaluating previous legal transitions
                        ind_temp = ([row-4,col], [row-3,col])
                        dp_matrix[row,col] = np.min(temp) + 10
                        backtrace[row,col] = ind_temp[np.argmin(temp)]
                if state_counter == 2:#V(ei)I
                    temp = (dp_matrix[row-2,col-1] + self.tranM[row_temp,1], dp_matrix[row,col-1] + self.tranM[row_temp, 4]) #evaluating previous legal transitions
                    ind_temp = ([row-2,col-1], [row,col-1])
                    dp_matrix[row,col] = np.min(temp) + self.inemM[row_temp,self.symb_index[col-1]]
                    backtrace[row,col] = ind_temp[np.argmin(temp)]



            state_counter = 0 if state_counter == 2 else state_counter + 1 
            row_temp = 1 if row_temp == ((self.leng)) else row_temp + 1

        #best_end = np.argmin(dp_matrix[-1, -1]) #find the best end state, only the column index
        #print(best_end)
        best_end = [backtrace[-1,-1]] #find what the END state points at, not sure if it is right but I always take the END state at the vary last column, that would be an end state that has lead to just the right amount of emission to even observe a sequence we have observed
        pointer_col = -3
        while pointer_col != -2: #we check for -2 because that is what we initiated the backtrace with, if we find it then we come to the end (the initiated probabilities)
            pointer_row = best_end[-1][0]
            pointer_col = best_end[-1][1]
            if pointer_col == -2:
                continue
            best_end.append(backtrace[pointer_row, pointer_col])
        #we must remove the last [-2,-2] because that is suppost to point at start
        best_end = best_end[:-1]
        print(best_end)
        print(backtrace.shape)
        best_path = []
        for i in best_end:
            best_path.append(self.states[i[0]])
        best_path.reverse()
        print(best_path)

            
        np.savetxt('please.txt', dp_matrix, fmt='%8.6s', delimiter='\t')

        np.savetxt('pleaseback.txt', backtrace[:, :, 0], fmt='%8.6s', delimiter='\t')



test = Decoder(list("PIVDTGSVAPLSAAEKTKIRSAWAPVYSTYETSGVDILVKFFTSTPAAQEFFPKFKGLTTADQLKKSADVRWHAERII"),"globins4.hmm")
test.viterbi()
