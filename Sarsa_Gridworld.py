import numpy as np
import matplotlib.pyplot as plt
    
    
def directionPlot(weights):
    fig,ax=plt.subplots(1,1)
    data=np.empty((6,6))
    c=0#counter
    for x in xrange(6):
        for y in xrange(6):
            action=np.argmax(weights[:,c])
            data[x,y]=action
            
            if x==1 and y==1:
                continue
                #dont draw arrow on goal
            else:
                if action == 0: #west
                    ax.arrow(y+1,x+0.5,-0.7,0,head_width=0.2,color='w')
                elif action == 1: #north
                    ax.arrow(y+0.5,x,0,0.7,head_width=0.2,color='w')
                elif action == 2: #east
                    ax.arrow(y,x+0.5,0.7,0,head_width=0.2,color='w')
                elif action == 3: #south
                    ax.arrow(y+0.5,x+1,0,-0.7,head_width=0.2,color='w')
                elif action == 4: #NW
                    ax.arrow(y+1,x,-0.7,0.7,head_width=0.2,color='w')
                elif action == 5: #NE
                    ax.arrow(y,x,0.7,0.7,head_width=0.2,color='w')
                elif action == 6: #SW
                    ax.arrow(y+1,x+1,-0.7,-0.7,head_width=0.2,color='w')
                elif action == 7: #SE
                    ax.arrow(y,x+1,0.7,-0.7,head_width=0.2,color='w')
                
            c+=1
    data[1,1]=-1        
    #print data
    
    ax.pcolormesh(data) 
    ax.set_yticks(np.arange(0.5,5.5)) #set ticks midway through square
    ax.set_xticks(np.arange(0.5,5.5))
    ax.set_yticklabels(range(1,7))
    ax.set_xticklabels(range(1,7))
    #plt.colorbar()
    plt.draw()
    plt.show()
    
    

def homing_nn(n_trials,learning_rate,eps,gamma,decay,actions,Trace):
    n_steps = 50
    ## Definition of the environment
    N = 6                               #height of the gridworld ---> number of rows
    M = 6                               #length of the gridworld ---> number of columns
    N_states = N * M                    #total number of states
    states_matrix = np.eye(N_states)
    
    N_actions=actions #which move set to use
    #W N E S NW NE SW SE
    action_row_change = np.array([-1,0,+1,0,-1,+1,-1,+1])#x               #number of cell shifted in vertical as a function of the action
    action_col_change = np.array([0,+1,0,-1,+1,+1,-1,-1])#y               #number of cell shifted in horizontal as a function of the action
    
    End = np.array([1, 1])                                  #terminal state--->reward
    s_end = np.ravel_multi_index(End,dims=(N,M),order='F')  #terminal state. Conversion in single index

    ## Rewards
    R = 10                              #only when the robot reaches the charger, sited in End state

    ## Variables
    weights = np.random.rand(N_actions,N_states)
    #print weights
    eligibility = np.zeros((N_actions,N_states))
    #print eligibility
    learning_curve = np.zeros((1,n_trials))

    # Start trials
    for trial in range(n_trials):
        # Initialization
        Start = np.array([np.random.randint(N),np.random.randint(M)])   #random start
        s_start = np.ravel_multi_index(Start,dims=(N,M),order='F')      #conversion in single index
        state = Start                                                   #set current state
        s_index = s_start                                               #conversion in single index
        step = 0         

        # Start steps
        while s_index != s_end and step <= n_steps:
            step += 1
            learning_curve[0,trial] = step
            input_vector = states_matrix[:,s_index].reshape(N_states,1)         #convert the state into an input vector        
            #compute Qvalues. Qvalue=logsig(weights*input). Qvalue is 2x1, one value for each output neuron
            Q = 1 / ( 1 + np.exp( - weights.dot(input_vector)))    #Qvalue is 2x1 implementation of logsig
            #eps-greedy policy implementation
            greedy = (np.random.rand() > eps)               #1--->greedy action 0--->non-greedy action
            if eps>0:
                eps-=0.01
            if greedy:
                action = np.argmax(Q)                           #pick best action
            else:
                action = np.random.randint(N_actions)           #pick random action

            state_new = np.array([0,0])
            #move into a new state
            state_new[0] = state[0] + action_row_change[action]
            state_new[1] = state[1] + action_col_change[action]
           
            #put the robot back in grid if it goes out. 
            #robot is given a negative reward for going off the grid
            flag=False   
            if state_new[0] < 0:
                state_new[0] = 0
                flag=True
            if state_new[0] >= N:
                state_new[0] = N-1
                flag=True
            if state_new[1] < 0:
                state_new[1] = 0
                flag=True
            if state_new[1] >= M:
                state_new[1] = M-1
                flag=True

            s_index_new = np.ravel_multi_index(state_new,dims=(N,M),order='F')  #conversion in a single index
                                    
            #update Qvalues
            if step >1:
                dw = learning_rate * (r_old - Q_old + gamma * Q[action]) * output_old.dot(input_old.T)
                eligibility+=output_old.dot(input_old.T)
                #use elegibility trace or not
                if Trace ==True:
                    weights+=dw*eligibility
                    eligibility *= gamma*decay
                else:
                    weights+=dw

            #store variables for sarsa computation in the next step
            output = np.zeros((N_actions,1))
            output[action] = 1
           
            #update variables
            input_old = input_vector
            output_old = output
            Q_old = Q[action]
            if flag==False:
                r_old = 0
            else:
                r_old = -1

            state[0] = state_new[0]
            state[1] = state_new[1]
            s_index = s_index_new

            #check if state is terminal and update the weights consequently
            if s_index == s_end:
                dw = learning_rate * (R - Q_old) * output_old.dot(input_old.T)
                eligibility+=output_old.dot(input_old.T)
                if Trace ==True:
                    weights+=dw*eligibility
                    eligibility *= gamma*decay
                else:
                    weights+=dw
      
    return learning_curve,weights


if __name__ == '__main__':
    np.random.seed(1)
    ##Alter these variables as needed##
    alpha =1.0#learning rate
    epsilon =0.1
    gamma =0.8
    n_trials = 100
    decay = 0.9#lambda
    actions =8 #can be either 8 or 4 (standard or king's moves)
    Trace=True #False=SARSA, True=SARSA(lambda), Eligibility trace
    repetitions = 100   # number of episodes, should be integer, greater than 0; for statistical reasons

    totalRewards = np.zeros((repetitions,n_trials))  # reward matrix. each row contains rewards obtained in one episode
    
    fontSize = 18
    
    # Start iterations over episodes
    for j in range(repetitions):
        learning_curve,weight=homing_nn(n_trials,alpha,epsilon,gamma,decay,actions,Trace)
        totalRewards[j,:] = learning_curve
        
        if j==0:
            weights=weight
        else:
            weights=(weights+weight)/2
        if j==repetitions-1:
            directionPlot(weights)
    # Plot the average reward as a function of the number of trials --> the average has to be performed over the episodes
    plt.figure()
    means = np.mean(totalRewards, axis = 0)
    print np.mean(means)    
    errors = 2 * np.std(totalRewards, axis = 0) / np.sqrt(repetitions) # errorbars are equal to twice standard error i.e. std/sqrt(samples)
    plt.errorbar(np.arange(n_trials), means, errors, 0, elinewidth = 3,ecolor='cyan')
    plt.axis([0,100,0,50])
    plt.xlabel('Trial',fontsize = fontSize)
    plt.ylabel('Number of Steps',fontsize = fontSize)

    plt.show()
    
