# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread

class threadAndReturn(Thread):
	def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
		Thread.__init__(self, group, target, name, args, kwargs, daemon=daemon)

		self._return = None

	def run(self):
		if self._target is not None:
			self._return = self._target(*self._args, **self._kwargs)

	def join(self):
		Thread.join(self)
		return self._return

def initialize_random_number(seed=None):
	#seed=0
	print("seed=",seed)
	np.random.seed(seed)
	
def random_int(n):
	return np.longlong(np.random.randint(0,n))

def random_double(n):
	return np.double(np.random.sample(1)*n)

def swap_unsigned_char(n1,n2):
	n=n1
	n1=n2
	n2=n
	return np.ulonglong(n1),np.ulonglong(n2)

def swap_int(n1,n2):
	n=n1
	n1=n2
	n2=n
	return np.longlong(n1),np.longlong(n2)
	
def initialize_parameters(pop_size=np.longlong(200),crs_type=np.longlong(3),crossover_rate=np.double(0.8),mutation_rate=np.double(0.9),elite_flag=np.longlong(1)):
	#work=np.double(0.5)
	pop_size=pop_size#np.random.random_integers(0,1,(200,))
	crs_type=crs_type
	crossover_rate=crossover_rate
	mutation_rate=mutation_rate
	elite_flag=elite_flag
	if elite_flag != 1:
		elite_flag=0
	return pop_size,crs_type,crossover_rate,mutation_rate,elite_flag
	
def initialize_genes(genotype,pop_size,gene_size):
	i=j=np.longlong(0)
	for i in range(pop_size):
		for j in range(gene_size):
			if random_double(1.)<0.5:
				genotype[i,j]=0
			else:
				genotype[i,j]=1
	return genotype

def copy_new_to_old(fitness,new_fitness,genotype,new_genotype,pop_size,gene_size):
	genotype=new_genotype
	fitness=new_fitness
	#i=j=np.longlong(0)
	#for i in range(pop_size):
	#	#for j in range(gene_size):
	#	#	genotype[i,j]=new_genotype[i,j]
	#	fitness[i]=new_fitness[i]
	return fitness,new_fitness,genotype,new_genotype

def one_point_crossover(n1,n2,genotype,gene_size):
	crs_pnt=np.longlong(0)
	i=np.longlong(0)
	
	crs_pnt=random_int(gene_size)
	twr=np.full((gene_size-(crs_pnt+1),),None)
	if gene_size-(crs_pnt+1) != 0:
		a_=np.arange(crs_pnt+1,gene_size,1).min()
		for i in range(crs_pnt+1,gene_size,1):
			a=i-a_
			twr[a] = threadAndReturn(target=swap_unsigned_char, args=(genotype[n1,i],genotype[n2,i],))
			twr[a].start()
		for i in range(crs_pnt+1,gene_size,1):
			a=i-a_
			genotype[n1,i],genotype[n2,i]=twr[a].join()
		del twr
	#for i in range(crs_pnt+1,gene_size,1):
	#	genotype[n1][i],genotype[n2][i]=swap_unsigned_char(genotype[n1,i],genotype[n2,i])
	return genotype

def two_point_crossover(n1,n2,genotype,gene_size):
	i=crs_pnt1=crs_pnt2=np.longlong(0)
	crs_pnt1=random_int(gene_size)
	crs_pnt2=random_int(gene_size)
	while crs_pnt1==crs_pnt2:
		crs_pnt2=random_int(gene_size)
	if crs_pnt1 > crs_pnt2:
		crs_pnt1,crs_pnt2=swap_int(crs_pnt1,crs_pnt2)
	
	twr=np.full((crs_pnt2-(crs_pnt1+1),),None)
	if crs_pnt2-(crs_pnt1+1) != 0:
		a_=np.arange(crs_pnt1+1,crs_pnt2,1).min()
		for i in range(crs_pnt1+1,crs_pnt2,1):
			a=i-a_
			twr[a] = threadAndReturn(target=swap_unsigned_char, args=(genotype[n1,i],genotype[n2,i],))
			twr[a].start()
		for i in range(crs_pnt1+1,crs_pnt2,1):
			a=i-a_
			genotype[n1,i],genotype[n2,i]=twr[a].join()
		del twr
	#for i in range(crs_pnt1+1,crs_pnt2,1):
	#	genotype[n1,i],genotype[n2,i]=swap_unsigned_char(genotype[n1,i],genotype[n2,i])
	return genotype


def uniform_crossover(n1,n2,genotype,gene_size):
	rnd=np.full((gene_size,),0.)
	for i in range(gene_size):
		rnd[i]=random_double(1.)
	twr=np.full((gene_size,),None)
	for i in range(gene_size):
		if rnd[i]<0.5:
			twr[i] = threadAndReturn(target=swap_unsigned_char, args=(genotype[n1,i],genotype[n2,i],))
			twr[i].start()
	for i in range(gene_size):
		if rnd[i]<0.5:
			genotype[n1,i],genotype[n2,i]=twr[i].join()
	del twr
	#for i in range(gene_size):
	#	if random_double(1.)<0.5:
	#		genotype[n1,i],genotype[n2,i]=swap_unsigned_char(genotype[n1,i],genotype[n2,i])
	return genotype
	
def get_roulette_table(i,roulette_table,new_fitness,fitness,new_genotype,genotype,pop_size,gene_size):
	rand_real=random_double(1.)
	num=0
	for k in range(pop_size):
		if roulette_table[k] > rand_real:
			num=k
			break
	j=np.longlong(np.arange(gene_size))
	new_genotype[i,j]=genotype[num,j]
	new_fitness[i]=fitness[num]
	return new_genotype[i,j],new_fitness[i]

def selection_using_roulette_rule(genotype,new_genotype,new_fitness,fitness,pop_size,gene_size,MAX_POP_SIZE):
	i=j=num=np.longlong(0)
	sum_=rand_real=np.double(0.)
	roulette_table=np.full((MAX_POP_SIZE,),0.)
	
	sum_=0.
	sum_=fitness.sum()
	#for i in range(pop_size):
	#	sum_=sum_+fitness[i]
	i=np.longlong(np.arange(pop_size))
	roulette_table[i]=fitness[i]/sum_
	#for i in range(pop_size):
	#	roulette_table[i]=fitness[i]/sum_
	
	sum_=0.
	for i in range(pop_size):
		sum_=sum_+roulette_table[i]
		roulette_table[i]=sum_
	twr=np.full((pop_size,),None)
	for i in range(pop_size):
			twr[i] = threadAndReturn(target=get_roulette_table, args=(i,roulette_table,new_fitness,fitness,new_genotype,genotype,pop_size,gene_size,))
			twr[i].start()
	for i in range(pop_size):
			j=np.longlong(np.arange(gene_size))
			new_genotype[i,j],new_fitness[i]=twr[i].join()
	del twr
	#for i in range(pop_size):
	#	rand_real=random_double(1.)
	#	for num in range(pop_size):
	#		if roulette_table[num] > rand_real:
	#			break
	#	j=np.longlong(np.arange(gene_size))
	#	new_genotype[i,j]=genotype[num,j]
	#	#for j in range(gene_size):
	#	#	new_genotype[i,j]=genotype[num,j]
	#	new_fitness[i]=fitness[num]
	fitness,new_fitness,genotype,new_genotype=copy_new_to_old(fitness,new_fitness,genotype,new_genotype,pop_size,gene_size)
	
	return genotype,new_genotype,new_fitness,fitness


def execute_crossover(genotype,crs_type,crossover_rate,pop_size,gene_size,MAX_POP_SIZE):
	i=num1=num2=np.longlong(0)
	num_of_pair=i
	number=np.full((MAX_POP_SIZE,),0)
	u=np.arange(pop_size)
	number[u]=u
	for i in range(pop_size):
		num1=random_int(pop_size)
		num2=random_int(pop_size)
		number[num1],number[num2]=swap_int(number[num1],number[num2])
	
	num_of_pair=np.longlong(pop_size/2)
	i=np.longlong(np.arange(num_of_pair))
	num1=number[2*i]
	num2=number[2*i+1]
	if random_double(1.) <= crossover_rate:
		if crs_type==1:
			genotype=one_point_crossover(num1,num2,genotype,gene_size)
		elif crs_type==2:
			genotype=two_point_crossover(num1,num2,genotype,gene_size)
		else:
			genotype=uniform_crossover(num1,num2,genotype,gene_size)
	#for i in range(num_of_pair):
	#	num1=number[2*i]
	#	num2=number[2*i+1]
	#	if random_double(1.) <= crossover_rate:
	#		if crs_type==1:
	#			genotype=one_point_crossover(num1,num2,genotype,gene_size)
	#		elif crs_type==2:
	#			genotype=two_point_crossover(num1,num2,genotype,gene_size)
	#		else:
	#			genotype=uniform_crossover(num1,num2,genotype,gene_size)
	return genotype
	
def execute_mutation(genotype,mutation_rate,pop_size,gene_size):
	i=j=np.longlong(0)
	i=np.longlong(np.arange(pop_size))
	for j in range(gene_size):
		if random_double(1.) <= mutation_rate:
			genotype[i,j]=np.ulonglong(1-genotype[i,j])
	#for i in range(pop_size):
	#	for j in range(gene_size):
	#		if random_double(1.) <= mutation_rate:
	#			genotype[i,j]=np.ulonglong(1-genotype[i,j])
	return genotype

def find_and_set_best_individual(fitness,elite_genotype,genotype,elite_fitness,elite_number,pop_size,gene_size):
	i=np.longlong(0)
	best_fitness=np.double(0.)
	
	for i in range(pop_size):
		if fitness[i] > best_fitness:
			elite_number=i
			best_fitness=fitness[i]
	i=np.longlong(np.arange(gene_size))
	elite_genotype[i]=genotype[elite_number,i]
	#for i in range(gene_size):
	#	elite_genotype[i]=genotype[elite_number,i]
	elite_fitness=fitness[elite_number]
	return elite_genotype,elite_fitness,elite_number

def elitist_strategy(fitness,elite_genotype,genotype,elite_fitness,pop_size,gene_size):
	worst_number=i=np.longlong(0)
	worst_fitness=np.double(0.)
	
	worst_fitness=1.
	worst_number=0
	for i in range(pop_size):
		if fitness[i] < worst_fitness:
			worst_number=i
			worst_fitness=fitness[i]
	i=np.longlong(np.arange(gene_size))
	genotype[worst_number,i]=elite_genotype[i]
	#for i in range(gene_size):
	#	genotype[worst_number,i]=elite_genotype[i]
	fitness[worst_number]=elite_fitness
	return fitness,genotype


def set_optimizing_task(fparam,PNUM):
	i=np.longlong(0)
	gene_size=8*PNUM
	for i in range(PNUM):
		fparam[i]=random_int(200)
	return fparam,gene_size

def trans_from_genotype_to_parameters(genotype,number,p,PNUM):
	i=j=0
	for i in range(PNUM):
		p[i]=0
		for j in range(8):
			p[i]=p[i]*2+genotype[number,j+8*i]
	return p

def fitness_value(fparam,genotype,number,PNUM):
	i=0
	p=np.full((PNUM,),0)
	f=np.double(0.)
	
	p=trans_from_genotype_to_parameters(genotype,number,p,PNUM)
	f=1.
	for i in range(PNUM):
		f=f-np.abs(p[i]-fparam[i])/(120.*PNUM)
	if f<0.:
		f=0.
	elif f>1.:
		f=1.
	return f
	
def calculate_fitness(fitness,fparam,genotype,pop_size,PNUM):
	twr=np.full((pop_size,),None)
	for i in range(pop_size):
		twr[i] = threadAndReturn(target=fitness_value, args=(fparam,genotype,i,PNUM,))
		twr[i].start()
	for i in range(pop_size):
		fitness[i]=twr[i].join()
	del twr
	#i=0
	#for i in range(pop_size):
	#	fitness[i]=fitness_value(fparam,genotype,i,PNUM)
	return fitness

def display_best_individual(generation,elite_genotype,elite_fitness):
	print("No.",generation,"elite,",elite_genotype,"-->",elite_fitness)

def generation_iteration(genotype,new_genotype,fitness,new_fitness,crossover_rate,mutation_rate,elite_flag,crs_type,pop_size,gene_size,elite_genotype,elite_fitness,elite_number,fparam,MAX_POP_SIZE,MAX_GENE_SIZE,PNUM,SOLUTION_FITNESS,MAX_GENERATION):
	i=generation=0
	p=np.full((PNUM,),0)
	
	generation=0
	fitness=calculate_fitness(fitness,fparam,genotype,pop_size,PNUM)
	elite_genotype,elite_fitness,elite_number=find_and_set_best_individual(fitness,elite_genotype,genotype,elite_fitness,elite_number,pop_size,gene_size)
	while elite_fitness<SOLUTION_FITNESS and generation<MAX_GENERATION:
		generation+=1
		genotype,new_genotype,new_fitness,fitness=selection_using_roulette_rule(genotype,new_genotype,new_fitness,fitness,pop_size,gene_size,MAX_POP_SIZE)
		genotype=execute_crossover(genotype,crs_type,crossover_rate,pop_size,gene_size,MAX_POP_SIZE)
		genotype=execute_mutation(genotype,mutation_rate,pop_size,gene_size)
		fitness=calculate_fitness(fitness,fparam,genotype,pop_size,PNUM)
		if elite_flag==1:
			fitness,genotype=elitist_strategy(fitness,elite_genotype,genotype,elite_fitness,pop_size,gene_size)
		elite_genotype,elite_fitness,elite_number=find_and_set_best_individual(fitness,elite_genotype,genotype,elite_fitness,elite_number,pop_size,gene_size)
		display_best_individual(generation,elite_genotype,elite_fitness)
	p=trans_from_genotype_to_parameters(genotype,elite_number,p,PNUM)
	print("最終的な解:",p)
	print("真の最適解:",fparam)

def main():
	MAX_POP_SIZE=200
	MAX_GENE_SIZE=50
	
	genotype=np.full((MAX_POP_SIZE,MAX_GENE_SIZE),0.)
	new_genotype=np.full((MAX_POP_SIZE,MAX_GENE_SIZE),0.)
	
	fitness=np.full((MAX_POP_SIZE,),0.)
	new_fitness=np.full((MAX_POP_SIZE,),0.)
	
	elite_genotype=np.full((MAX_GENE_SIZE,),0.)
	elite_fitness=np.double(0.)
	elite_number=np.longlong(0)
	pop_size=np.longlong(0)
	gene_size=np.longlong(0)
	crossover_rate=np.double(0)
	mutation_rate=np.double(0)
	elite_flag=np.longlong(0)
	crs_type=np.longlong(0)
	
	PNUM=5
	fparam=np.full((PNUM,),0)
	SOLUTION_FITNESS=0.999
	MAX_GENERATION=5000
	
	initialize_random_number()
	fparam,gene_size=set_optimizing_task(fparam,PNUM)
	pop_size,crs_type,crossover_rate,mutation_rate,elite_flag=initialize_parameters(pop_size=np.longlong(200),crs_type=np.longlong(1),crossover_rate=np.double(0.8),mutation_rate=np.double(0.9),elite_flag=np.longlong(1))
	genotype=initialize_genes(genotype,pop_size,gene_size)
	print("genotype,new_genotype,fitness,new_fitness,crossover_rate,mutation_rate,elite_flag,crs_type,pop_size,gene_size,elite_genotype,elite_fitness,elite_number,fparam,MAX_POP_SIZE,MAX_GENE_SIZE,PNUM,SOLUTION_FITNESS,MAX_GENERATION",genotype,new_genotype,fitness,new_fitness,crossover_rate,mutation_rate,elite_flag,crs_type,pop_size,gene_size,elite_genotype,elite_fitness,elite_number,fparam,MAX_POP_SIZE,MAX_GENE_SIZE,PNUM,SOLUTION_FITNESS,MAX_GENERATION)
	generation_iteration(genotype,new_genotype,fitness,new_fitness,crossover_rate,mutation_rate,elite_flag,crs_type,pop_size,gene_size,elite_genotype,elite_fitness,elite_number,fparam,MAX_POP_SIZE,MAX_GENE_SIZE,PNUM,SOLUTION_FITNESS,MAX_GENERATION)

main()

