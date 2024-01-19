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
                    print(np.array(line[1:21]).shape[0])

                    print(np.array(line[1:21]))
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

                    
            


    def viterbi(self):
        prob_matrix = np.zeros((self.emission_matrix.shape[0],len(self.seq)))
        state_matrix = np.zeros((self.emission_matrix.shape[0],len(self.seq)))
        state_list = ["M","I","D","M","I","M","D"]
        most_seq_states = [None] * len(self.seq)

        #initiate
        prob_matrix[:,0] = self.initial_insertion_emission[:self.symbols.index(self.seq[0])] * self.initial_transition

        for i in range(1, prob_matrix.shape[1]):
            for j in range(prob_matrix.shape[0]):
                temp = prob_matrix[i,j-1] * self.transition_matrix[j,:] #just relized that we need to check what the most probebiable state transition is
                best_state, = np.argmax(self.transition_matrix[j,:])
                prob_matrix[i,j] = np.max(temp)*self.emission_matrix[best_state,self.seq[i]]
                most_seq_states[i-1] = state_list[np.argmax(self.transition_matrix[j,:])]

        print(prob_matrix)





    #print(self.transition_matrix)
    #print(self.insertion_emission_matrix)
    #print(self.initial_insertion_emission)
    #print(self.initial_transition)
    #print(self.nullprob_emission)
                


                
test = Decoder(["A","A","A","A","A","A","A","A","A","A","A","A","S","G","H","A","G","H","S","K","L"],"globins4.hmm")
test.viterbi()







        

