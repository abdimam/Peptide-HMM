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
        

                    
      


    def viterbi(self): #rewritten
        prob_matrix = np.full((3, len(self.seq)), np.inf)
        backtrack_matrix = np.full((3, len(self.seq)), np.inf)
        prob_matrix[0,0] = self.initial_transition[0] + self.emission_matrix[0,np.where(self.seq[0])]
        prob_matrix[1,0] = self.initial_transition[1] + self.initial_insertion_emission[np.where(self.seq[0])]
        prob_matrix[2,0] = self.initial_transition[2]


        #doing what durbin says in chapter 5 for viterbi
        #might be wrong, but we need to use the initial insertion emission till we score a match state, also transitions such as D > I and vice versa are illegal, so those are inf
        begin_check=0
        for i in range(1,prob_matrix.shape[1]):
            if begin_check == 0:
              print(prob_matrix)
              #V(t)M
              prob_matrix[0,i] = np.min(prob_matrix[:,i-1]+np.vstack([self.transition_matrix[i,0],self.initial_transition[3],self.initial_transition[5]])) + self.emission_matrix[i,np.where(self.seq[i])] #the weird splicing in trans matrix is cuz I messed up earlier
              backtrack_matrix[0,i] = np.argmin(prob_matrix[:,i-1]+self.transition_matrix[i,[0,3,5]]) 

              #V(t)I

              prob_matrix[1, i] = np.min(prob_matrix[:, i - 1] + (np.vstack([self.transition_matrix[i, 1], self.initial_transition[4], np.inf]))) if np.argmin(prob_matrix[:, i - 1] + (np.vstack([self.transition_matrix[i, 1], self.initial_transition[4], np.inf]))) == 1 + self.emission_matrix[i, np.where(self.seq[i])] else self.initial_insertion_emission[np.where(self.seq[i])]
              backtrack_matrix[1,i] = np.argmin(prob_matrix[:,i-1]+(np.vstack([self.transition_matrix[i,1],self.initial_transition[4],np.inf])))

              #V(t)D
              prob_matrix[2,i] = np.min(prob_matrix[:,i-1]+(np.vstack([self.transition_matrix[i,2],np.inf,np.inf])))
              backtrack_matrix[2,i] = np.argmin(prob_matrix[:,i-1]+(np.vstack([self.transition_matrix[i,2],np.inf,np.inf])))
                
              if np.argmin(prob_matrix[:,i]) == 0:
                  begin_check += 1
            if begin_check != 0:
            #V(t)M
                prob_matrix[0,i] = np.min(prob_matrix[:,i-1]+self.transition_matrix[i,[0,3,5]]) + self.emission_matrix[i,np.where(self.seq[i])] #the weird splicing in trans matrix is cuz I messed up earlier
                backtrack_matrix[0,i] = np.argmin(prob_matrix[:,i-1]+self.transition_matrix[i,[0,3,5]]) 

                #V(t)I
                prob_matrix[1,i] = np.min(prob_matrix[:,i-1]+(np.vstack([self.transition_matrix[i,[1,4]],np.inf]))) + self.emission_matrix[i,np.where(self.seq[i])]
                backtrack_matrix[1,i] = np.argmin(prob_matrix[:,i-1]+(np.vstack([self.transition_matrix[i,[1,4]],np.inf])))

                #V(t)D
                prob_matrix[2,i] = np.min(prob_matrix[:,i-1]+(np.vstack([self.transition_matrix[i,2],np.inf,self.transition_matrix[i,6]])))
                backtrack_matrix[2,i] = np.argmin(prob_matrix[:,i-1]+(np.vstack([self.transition_matrix[i,2],np.inf,self.transition_matrix[i,6]])))
            return prob_matrix
        

test = Decoder("VLSDAEWQLVLNIWAKVEADVAGHGQDILIRLFKGHPETLEKFDKFKHLKTEAEMKASEDLKKHGNTVLTALGGILKKKGHHEAELKPLAQSHATKHKIIKYLEFISDAIIHVLHSRHPGDFGADAQAAMNKALELFRKDIAAKYKELGFQG".split(),"globins4.hmm")
lol = test.viterbi()
        

