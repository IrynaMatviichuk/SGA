#########################################################
#                                                       #
#       SIMPLE GENETIC ALGORITHM                        #
#                                                       #
#                 Mazur, Matviichuk                     #
#                                                       #
#########################################################
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#########################################################
# ALGORITHM PARAMETERS                                  #
#########################################################
N=50                 # Population size
Genome=4             # Genome length 
generation_max= 10  # Maximum of generations - iterations

#########################################################
# VARIABLES ALGORITHM                                   #
#########################################################
popSize = N+1
genomeLength  = Genome+1
# init best chromosome
the_best_chrom=0;         
# fitness
fitness = np.empty([popSize])
# probability
probability = np.empty([popSize])
net = np.empty([popSize])
# chromosomes
chromosome = np.empty([popSize, genomeLength],dtype=np.int)
nchromosome = np.empty([popSize, genomeLength],dtype=np.int)
child1 = np.empty([popSize, genomeLength],dtype=np.int)
child2 = np.empty([popSize, genomeLength],dtype=np.int)
# init and save best chromosome
best_chrom = np.empty([generation_max],dtype=np.int);
the_best_chrom=0;
generation=0;

#########################################################
# Define your problem in this section. For instance:    #
#                                                       #
# Let f(x)=abs(x-5/2+sin(x)) be a function that takes   #
# values in the range 0<=x<=15. Within this range f(x)  #
# has a maximum value at x=11 (binary is equal to 1011) #
#########################################################
def fitness_function(x):
    return np.fabs((x-5)/(2+np.sin(x)))

#########################################################
# POPULATION INITIALIZATION                             #
#########################################################
def Init_population():
    for i in range(1,popSize):
        for j in range(1,genomeLength):
            allele=np.random.random_integers(100)
            allele=allele/100
            if allele<=0.5:
                chromosome[i,j]=0
            else:
                chromosome[i,j]=1

#########################################################
# SHOW POPULATION                                       #
#########################################################
def Show_population():
    for i in range(1,popSize):
        for j in range(1,genomeLength):
            print(chromosome[i,j],end="")
        print()

#########################################################
# FITNESS EVALUATION                                    #
#########################################################
def Fitness_evaluation(generation):
    # fitness evaluation
    i=1; j=1; fitness_total=0; sum_sqr=0;
    fitness_average=0; variance=0;
    for i in range(1,popSize):
        fitness[i]=0

    for i in range(1,popSize):
       x=0;
       for j in range(1,genomeLength):
           # translate from binary to decimal value
           x=x+chromosome[i,j]*pow(2,genomeLength-j-1)
           # replaces the value of x in the function f(x)
           y = fitness_function(x)
           # the fitness value is calculated below:
           fitness[i]=y*100

       print("fitness = ",i," ",fitness[i])
       fitness_total=fitness_total+fitness[i]
    fitness_average=fitness_total/N
    i=1;
    while i<=N:
        sum_sqr=sum_sqr+pow(fitness[i]-fitness_average,2)
        i=i+1
    variance=sum_sqr/N
    if variance<=1.0e-4:
        variance=0.0
    # Best chromosome selection
    the_best_chrom = 0;
    fitness_max=fitness[1];
    for i in range(1,popSize):
        if fitness[i]>=fitness_max:
            fitness_max=fitness[i]
            the_best_chrom=i
    best_chrom[generation]=the_best_chrom
    # Statistical output
    f = open("output.txt", "a")
    f.write(str(generation)+" "+str(fitness_average)+"\n")
    f.write(" \n")
    f.close()
    print("Population size = ", popSize - 1)
    print("mean fitness = ",fitness_average)
    print("variance = ",variance," Std. deviation = ",math.sqrt(variance))
    print("fitness max = ",best_chrom[generation])
    print("fitness sum = ",fitness_total)

#########################################################
# TOURNAMENT SELECTION OPERATOR                         #
#########################################################
def select_p_tournament():
    u1=0; u2=0; parent=99;
    while (u1==0 and u2==0):
        u1=np.random.random_integers(popSize-1)
        u2=np.random.random_integers(popSize-1)
        if fitness[u1] <= fitness[u2]:
            parent = u1
        else:
            parent = u2
    return parent

#########################################################
# WHEEL PARENTS SELECTION OPERATOR                      #
#########################################################
def wheel_p_selection():
    fitness_total=0; parent=0;
    for i in range(1,popSize):
        fitness_total = fitness_total + fitness[i]
    for i in range(1,popSize):
        probability[i]=fitness[i]/fitness_total
    for i in range(1,popSize):
        net[0]=0
    for i in range(1,popSize):
        net[i]=net[i-1]+probability[i]
    u=np.random.uniform(0,1)
    bugs=1; k=1;
    while bugs<=popSize-1:
        for i in range(1,popSize):
            if u<net[i]:
                parent=i
                for j in range(1,genomeLength):
                    nchromosome[k,j]=chromosome[parent,j]
                k=k+1
                break
        u=np.random.uniform(0,1)
        bugs=bugs+1
    for i in range(1,popSize):
        for j in range(1,genomeLength):
            chromosome[i,j]=nchromosome[i,j]

#########################################################
# FLIP BIT MUTATION OPERATOR                            #
#########################################################
# pop_mutation_rate: mutation rate in the population
# mutation_rate: probability of a mutation of a bit 
def mutation(pop_mutation_rate, mutation_rate):
    for i in range(1,popSize):
        up=np.random.random_integers(100)
        up=up/100
        if up<=pop_mutation_rate:
            for j in range(1,genomeLength):
                um=np.random.random_integers(100)
                um=um/100
                if um<=mutation_rate:
                    if chromosome[i,j]==0:
                        nchromosome[i,j]=1
                    else:
                        nchromosome[i,j]=0
                else:
                    nchromosome[i,j]=chromosome[i,j]
        else:
            for j in range(1,genomeLength):
                nchromosome[i,j]=chromosome[i,j]
    for i in range(1,popSize):
        for j in range(1,genomeLength):
            chromosome[i,j]=nchromosome[i,j]
            
########################################################
# ONE-POINT CROSSOVER                                  #
########################################################
# crossover_rate: setup crossover rate
def mating(crossover_rate):
    j=0;
    crossover_point=0;
    parent1=select_p_tournament()
    parent2=select_p_tournament()
    if random.random()<=crossover_rate:
        crossover_point = np.random.random_integers(genomeLength-2)
    j=1; 
    while (j<=genomeLength-1):
        if j<=crossover_point:
            child1[parent1,j]=chromosome[parent1,j]
            child2[parent2,j]=chromosome[parent2,j]

        else:
            child1[parent1,j]=chromosome[parent2,j]
            child2[parent2,j]=chromosome[parent1,j]
        j=j+1

    j=1
    for j in range(1,genomeLength):
        chromosome[parent1,j]=child1[parent1,j]
        chromosome[parent2,j]=child2[parent2,j]

def crossover(crossover_rate):
    c=1;
    while (c<=N):
        mating(crossover_rate)
        c=c+1

#########################################################
# PERFORMANCE GRAPH                                     #
#########################################################
# Read the Docs in http://matplotlib.org/1.4.1/index.html
def plot_Output():
    data = np.loadtxt('output.txt')
    # plot the first column as x, and second column as y
    x=data[:,0]
    y=data[:,1]
    sns.lineplot(x=x, y=y)
    plt.show()

#########################################################
#                                                       #
# MAIN PROGRAM                                          #
#                                                       #
#########################################################
def Simple_GA():
    generation=0
    print(f"============== GENERATION: {generation} =========================== ")
    print()
    Init_population()
    Show_population()
    Fitness_evaluation(generation)
    while (generation<generation_max-1):
        print("The best of generation [",generation,"] ", best_chrom[generation])
        print()
        print(f"============== GENERATION: {generation+1} =========================== ")
        print()
        # Select individuals, update generation number and obtain fitness
        wheel_p_selection()
        generation=generation+1
        Fitness_evaluation(generation)
        # Apply genetic operators  
        crossover(0.75)
        mutation(0.01,0.01)

print("SIMPLE GENETIC ALGORITHM")
input("Press Enter to continue...")
Simple_GA()
plot_Output()
