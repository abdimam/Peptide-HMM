import numpy as np
import pandas as pd




class Decoder:
    def __init__(self,seq,hmm):
        self.emission_matrix = np.empty((0, 20)) #for match state :)
        self.insertion_emission_matrix = np.empty((0, 20))
        self.transition_matrix = np.empty((0, 7))
        self.seq = seq
        lines_after_compo = 0  # counter so we can store the initial parameters in a reasonable way 
        is_compo_line = False
        is_compo_done = False
        counter_to_two = 0
        leng_counter = 0
        self.states = ["M","I","D"]
        
        with open(hmm, 'r') as hmm_file:
            
            for line in hmm_file:
                line = line.split() #what we do here is removing all blank spaces, each string surronded by whitespaces will be an element to the list'
                if line == ["//"]:
                    break 
                if line == []:
                    break 
                if "LENG" in line: #if we find it we can safely say that index 1 of this list is the number we are after
                    self.leng = str(line[1])
                    #print(self.leng)
                    #self.leng =  len(line[4:]) #we removed all white spaces so removing the first 4 characters, which should be "LENG" will give us the length of the consensus sequence in the model!
                    #print(self.leng)
                 
                if "HMM" in line: #we take out all the symbols
                    self.symbols = line[1:]
                    #print(self.symbols)
                if "COMPO" in line:
                    self.nullprob_emission = line[1:] #obs, background emission for match states!
                    #print(self.nullprob)
                    is_compo_line = True
                    continue #skip and go to next line, 
                if is_compo_line and lines_after_compo == 0:
                    self.initial_insertion_emission = line
                    lines_after_compo += 1 #increase this with 1
                    continue
                if is_compo_line and lines_after_compo == 1: #when this is achieved, we know we are on the line with the initial transition probabilities
                    self.initial_transition = line
                    is_compo_line = False #this is to break the conditions above because we are done with capturing the initiation probabilities in the model
                    is_compo_done = True
                    continue #next iteration of lines!
                #now we capture the "normal" transition, insertion-emission and emission probabilites
                #we know the length of the consensus length, the amount of lines after are that amount times 3
                if is_compo_done and counter_to_two == 0:
                    self.emission_matrix = np.vstack((self.emission_matrix,np.array(line[1:21])))
                    counter_to_two += 1
                    continue
                if is_compo_done and counter_to_two == 1:
                    self.insertion_emission_matrix = np.vstack((self.insertion_emission_matrix,np.array(line)))
                    counter_to_two += 1
                    continue
                if is_compo_done and counter_to_two == 2:
                    self.transition_matrix = np.vstack((self.transition_matrix,np.array(line)))#The line with transition probabilites are proceeded and followed with stuff we dont want, we do not include them by using correct splicers
                    counter_to_two = 0 #reset it back to 0 and we alternate between these 3 if statements
                    leng_counter += 1
                    if leng_counter == int(self.leng):
                        continue
        #make all elements into floats, replace all "*" with negative infinity"


        self.initial_transition[self.initial_transition.index("*")] = np.inf
        self.initial_transition = np.array(self.initial_transition, dtype=float)
        #print(self.initial_transition)

        if "*" in self.initial_insertion_emission:
            self.initial_insertion_emission[self.initial_insertion_emission.index("*")] = np.inf
        self.initial_insertion_emission = np.array(self.initial_insertion_emission, dtype=float)
        #print(self.initial_insertion_emission)

        if "*" in self.emission_matrix:
            self.emission_matrix[np.where(self.emission_matrix == "*")] = np.inf
        self.emission_matrix = self.emission_matrix.astype(float)
        
        if "*" in self.transition_matrix:
            self.transition_matrix[np.where(self.transition_matrix == "*")] = np.inf
        self.transition_matrix = self.transition_matrix.astype(float)

        if "*" in self.insertion_emission_matrix:
            self.insertion_emission_matrix[np.where(self.insertion_emission_matrix == "*")] = np.inf
        self.insertion_emission_matrix = self.insertion_emission_matrix.astype(float)

        self.leng = int(self.leng)

        #correcting that the elements in self_initial_transition contains the transitions from BEGIN > (any state) and INSERT0 to (any state)
        self.initial_transition_insert = self.initial_transition[3:7]
        self.initial_transition = self.initial_transition[0:3]
        

                    
      


    def viterbi(self): #rewritten
        prob_matrix = np.full((len(self.states),len(self.seq)), np.inf)
        backyardigans = np.full((len(self.states)+1,len(self.seq)), np.inf)
        begin_counter = 0
        seq_leng_counter = 0
        seq_leng_threshold = len(self.seq)

        #initiating

        prob_matrix[:,0] = self.initial_transition #these are handling the transitions from BEGIN
        prob_matrix[0,0] += self.emission_matrix[0,self.symbols.index(self.seq[0])]
        prob_matrix[1,0] += self.initial_insertion_emission[self.symbols.index(self.seq[0])]


        #now we need to take into consideration that D state is silent and thus emitts NOTHING but we still have to emitt a total of len(self.seq_symbols), I think the termination will be done by itself not sure
        temp1 = [np.inf,np.inf,np.inf]
        temp2 = [np.inf,np.inf,np.inf]
        temp3 = [np.inf,np.inf,np.inf]
        for i in range(1,len(self.seq)):
            #the inner loop will differ depending on if we succesfully leave the BEGIN state into an MATCH state
            
            if begin_counter == 0:
                for j in range(len(self.states)):
                    #V(t)M, this would be going from any state to MATCH
                    if j == 0:
                        temp1[0] = prob_matrix[0,i-1] + self.transition_matrix[i,0] 
                        temp1[1] = prob_matrix[1,i-1] + self.initial_transition_insert[0] #uses the special transition probability from INSERT0 to MATCH
                        temp1[2] = prob_matrix[2,i-1] + self.initial_transition_insert[2]

                        prob_matrix[j,i] = np.min(temp1) + self.emission_matrix[i,self.symbols.index(self.seq[i])]
                        backyardigans[j,i] = np.argmin(temp1)

                    if j == 1:
                        temp2[0] = prob_matrix[0,i-1] + self.transition_matrix[i,1] 
                        temp2[1] = prob_matrix[1,i-1] + self.initial_transition_insert[1] 
                        temp2[2] = prob_matrix[2,i-1] + np.inf #stockholm format says that I > D is very unlikely, I think Durbin agrees and so do I

                        if np.argmin(temp2) == 0:
                            prob_matrix[j,i] = np.min(temp2) + self.insertion_emission_matrix[i,self.symbols.index(self.seq[i])]
                           # begin_counter += 1
                        else:
                            prob_matrix[j,i] = np.min(temp2) + self.initial_insertion_emission[self.symbols.index(self.seq[i])] #dont have to worry about the case from D > I cuz it will never happen
                        backyardigans[j,i] = np.argmin(temp2) 
                    if j == 2: #maybe I should do it like durbin shows in page 71?????? not sure how to implement it
                        temp3[0] = prob_matrix[0,i-1] + self.transition_matrix[i,2] 
                        temp3[1] = prob_matrix[1,i-1] + np.inf #I>D never happens
                        temp3[2] = prob_matrix[2,i-1] + self.initial_transition_insert[3]

                        prob_matrix[j,i] = np.min(temp3)
                        backyardigans[j,i] = np.argmin(temp3)
                if np.argmin(prob_matrix[:,i]) == 0 or np.argmin(prob_matrix[:,i]) == 2:
                    begin_counter += 1 #should work because if these are the biggest then it means the most likely state 


                if np.argmin(temp1) or np.argmin(temp2) or np.argmin(temp3) == 0: #this would mean that we came from an match state
                    begin_counter += 1

            if begin_counter > 0:
                for j in range(len(self.states)):
                    #V(t)M, this would be going from any state to MATCH
                    if j == 0:
                        temp1[0] = prob_matrix[0,i-1] + self.transition_matrix[i,0] 
                        temp1[1] = prob_matrix[1,i-1] + self.transition_matrix[i,3] 
                        temp1[2] = prob_matrix[2,i-1] + self.transition_matrix[i,5]

                        prob_matrix[j,i] = np.min(temp1) + self.emission_matrix[i,self.symbols.index(self.seq[i])]
                        backyardigans[j,i] = np.argmin(temp1)

                    if j == 1:
                        temp2[0] = prob_matrix[0,i-1] + self.transition_matrix[i,1] 
                        temp2[1] = prob_matrix[1,i-1] + self.transition_matrix[i,4] 
                        temp2[2] = prob_matrix[2,i-1] + np.inf #stockholm format says that I > D is very unlikely, I think Durbin agrees and so do I

                        prob_matrix[j,i] = np.min(temp2) + self.insertion_emission_matrix[i,self.symbols.index(self.seq[i])]
                        backyardigans[j,i] = np.argmin(temp2) 
                    if j == 2: #maybe I should do it like durbin shows in page 71?????? not sure how to implement it
                        temp3[0] = prob_matrix[0,i-1] + self.transition_matrix[i,2] 
                        temp3[1] = prob_matrix[1,i-1] + np.inf #I>D never happens
                        temp3[2] = prob_matrix[2,i-1] + self.transition_matrix[1,6]

                        prob_matrix[j,i] = np.min(temp3)
                        backyardigans[j,i] = np.argmin(temp3)
       
        last_state = np.argmin(prob_matrix[:, -1])

       
        best_path = [self.states[last_state]]

        # Backtrace from the last column to the first
        for i in range(prob_matrix.shape[1] - 1, 0, -1):
            last_state = int(backyardigans[last_state, i])
            best_path.append(self.states[last_state])





        return prob_matrix, best_path




                


                    
                        







test = Decoder(list("VLSDAEWQLVLNIWAKVEADVAGHGQDILIRLFKGHPETLEKFDKFKHLKTEAEMKASEDLKKHGNTVLTALGGILKKKGHHEAELKPLAQSHATKHKIIKYLEFISDAIIHVLH"),"globins4.hmm")
lol, lol1 = test.viterbi()
print(lol.shape)
print(lol1)


#test = Decoder("VLSDAEWQLVLNIWAKVEADVAGHGQDILIRLFKGHPETLEKFDKFKHLKTEAEMKASEDLKKHGNTVLTALGGILKKKGHHEAELKPLAQSHATKHKIIKYLEFISDAIIHVLHSRHPGDFGADAQAAMNKALELFRKDIAAKYKELGFQG".split(),"globins4.hmm")
#lol = test.viterbi()
        

