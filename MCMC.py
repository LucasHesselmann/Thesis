#! /usr/bin/env python3.4

import sys
import os
import numpy
import pylab
import random
import math
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing as mp


def multiDimDiagramm(string, numberOfSamples, directoryName, resultName):
  """
  Plot the convergence time for several dimensions.
  
  @param: string - 'RWM' or 'MALA' selects the algoritm type
  @param: numberOfSamples - The number of samples which  are generated
  @param: directoryName - Name of the directory for the analysis files generated for each dimension
  @param: resultName - Name of the diagramm showing the convergence time

  Example:  MCMC.multiDimDiagramm('RWM', 100000, 'Test01', 'ConvergenceTimeDiagramm')
  
  """
  result=[]
  acceptRate=[]
  autocor=[]
  colours=['r','g','b','k','y']
  k=0

  # Some preparation
  directory = '{0}_{1}_{2}'.format(directoryName, string, numberOfSamples)
  os.makedirs(directory) 
  
  pylab.figure()
  # Make diagramm for several dimensions
  dimensions=[5, 10, 20, 30]
  for dim in dimensions:
    result=makeDiagramm(string, dim, numberOfSamples, directoryName, resultName)
    acceptRate.append(result[0])
    autocor.append(result[1])
    pylab.plot(result[0], result[1], colours[k], label='Convergence time')
    pylab.ylabel('convergence time')
    pylab.xlabel('acceptance rate')
    pylab.grid(True)
    fileName1=os.path.join(directory, '{0}_{1}_Dim{2}.png'.format(string, resultName, dim))
    pylab.savefig(fileName1) 
    k+=1
  pylab.plot(acceptRate[0], autocor[0], colours[0], acceptRate[1], autocor[1], colours[1], acceptRate[2], autocor[2], colours[2], acceptRate[3], autocor[3], colours[3], label='Convergence time')
  pylab.ylabel('convergence time')
  pylab.xlabel('acceptance rate')
  #pylab.ylim([4.0, 25.0])
  pylab.grid(True)
  fileName1=os.path.join(directory, '{0}_{1}.png'.format(string, resultName))
  pylab.savefig(fileName1) 
  pylab.clf()
                          

 
def makeDiagramm(string, dimension, numberOfSamples, directoryName, resultName):
  """
  Make a diagramm to show behavior of the algo depending on the scaling of the variance.
  
  @param: string - 'RWM' or 'MALA' selects the algoritm type
  @param: dimension - Dimension of the Markov chain
  @param: numberOfSamples - The number of samples which  are generated
  @param: directoryName - Name of the directory for the analysis files generated for each dimension
  @param: resultName - Name of the diagramm showing the convergence time

  Example:  MCMC.makeDiagramm('RWM', 5, 100000, 'Test01', 'ConvergenceTimeDiagramm')
  
  """
  result=[]
  autocor=[]
  acceptRate=[]
  steps=[]
  init=[]
  variances=[]
  # Number of repititions of simulations for each variance, we take the median of the acceptance rates and autocorrelation times. 
  repitition = 1
  tmpVec = numpy.array([0.0 for i in range(repitition)])
  tmpVec2 = numpy.array([0.0 for i in range(repitition)])
  k=1
  dimensionIter=range(dimension)
  # Simulations with a higher value as autocorrelation time are neglected.
  threshold = 2500
  # Set order of convergence depending of algorithm type
  if string in ['MALA']:
    convOrder=1.0/3.0
  else:
    convOrder=1.0

  # Some preparation
  directory = '{0}_{1}_{2}_{3}'.format(directoryName, string, dimension, numberOfSamples)
  os.makedirs(directory) 
  # Initialise the corresponding algorithm with a burn-in of 100 steps (choose appropriate init value)
  algo = Algo(string, dimension, 100)
  # Initialize the init value
  init.append(random.gauss(2.5, 0.2))
  for dim in dimensionIter[1:]:
    init.append(random.gauss(-0.1, 0.2))
  # Generate the proposal variances and scale them according to the dimension
  if string in ['MALA']:
    if dimension == 1:
      helper=numpy.logspace(-3.4, 3.6, 5, True, 2.0) #MALA in 1D
    elif dimension == 2:
      helper=numpy.logspace(-3.8, 2.5, 5, True, 2.0) #MALA in 2D
    elif dimension == 5:
      helper=numpy.logspace(-2.0, 4.8, 5, True, 2.0) #MALA in 5D
    elif dimension == 10:
      helper=numpy.logspace(-3.8, 2.2, 8, True, 2.0) #MALA in 10D
    elif dimension == 15:
      helper=numpy.logspace(-2.0, 5.2, 5, True, 2.0) #MALA in 15D
    elif dimension == 20:
      helper=numpy.logspace(-1.8, 4.4, 5, True, 2.0) #MALA in 20D
    elif dimension == 30:
      helper=numpy.logspace(-1.8, 4.2, 5, True, 2.0) #MALA in 30D
    else:
      helper=numpy.logspace(-3.8, 2.2, 10, True, 2.0)
  elif string in ['RWM']:
    if dimension == 1:
      helper=numpy.logspace(-0.8, 15, 8, True, 2.0) #RWM in 1D
    elif dimension == 2:
      helper=numpy.logspace(-0.8, 8, 8, True, 2.0) #RWM in 2D
    elif dimension == 5:
      helper=numpy.logspace(-0.8, 5.2, 8, True, 2.0) #RWM in 5D
    elif dimension == 10:
      helper=numpy.logspace(-0.8, 5.0, 8, True, 2.0) #RWM in 10D
    elif dimension == 15:
      helper=numpy.logspace(-0.6, 4.8, 8, True, 2.0) #RWM in 15D
    elif dimension == 20:
      helper=numpy.logspace(-0.5, 4.7, 8, True, 2.0) #RWM in 20D
    elif dimension == 30:
      helper=numpy.logspace(-0.4, 4.6, 8, True, 2.0) #RWM in 30D
    elif dimension == 50:
      helper=numpy.logspace(-0.4, 4.6, 8, True, 2.0) #RWM in 50D
    else:
      helper=numpy.logspace(-1, 9, 11, True, 2.0)
  for var in helper:
    variances.append(var / dimension**(convOrder))
  # Simulate for each variance in variances.
  for var in variances:
    for i in range(repitition):
      result=algo.simulation(numberOfSamples, var, False, True, True, init)
      tmp = format(result[0], '.2f')
      if result[0]>0.95:
        tmpVec2[i]=tmp
        tmpVec[i]=threshold+1
        continue
      fileName=os.path.join(directory, '{0}_{1}-{2}_{3}.png'.format(string, k, i, tmp))
      # For analysis: mean = 0
      tmp2=algo.analyseData(result[1], result[2], result[3], fileName)  
      tmpVec[i] = (tmp2 / dimension**(convOrder))
      tmpVec2[i] = (tmp)
      print('-----------------------------------------------------------')
      print('Round {0}'.format(i))
      print('Acceptance rate: {0}'.format(tmp))
      print('Integrated autocorrelation: {0}'.format(tmp2 / dimension**(convOrder)))
      print('-----------------------------------------------------------')
    # Take the median of acceptance rate and convergence time
    tmp2=numpy.median(tmpVec2)
    if tmp2 > 0.95:
      continue
    tmp=numpy.median(tmpVec)
    if tmp < threshold:
      acceptRate.append(tmp2)
      autocor.append(tmp)
      print('-----------------------------------------------------------')
      print('The median of loop {0}'.format(k))
      print('Acceptance rate: {0}'.format(tmp2))
      print('Integrated autocorrelation: {0}'.format(tmp))
      print('-----------------------------------------------------------')
      k+=1
    if tmp2<0.05:
      break
  pylab.figure()
  pylab.plot(acceptRate, autocor, 'ro', label='Autocorrelation time')
  pylab.ylabel('scaled autocorrelation time')
  pylab.xlabel('acceptance rate')
  pylab.grid(True)
  fileName1=os.path.join(directory, '{0}_{1}_Dim{2}.png'.format(string, resultName, dimension))
  pylab.savefig(fileName1) 
  pylab.clf()
  returnValue=[]
  returnValue.append(acceptRate)
  returnValue.append(autocor)
  return returnValue
                          
  
class Algo:
  """
  This class generates a MCMC method (RWM or MALA) and generates a sample of random variables according to the target distribution.
  """
  
  def __init__(self, Algo, dimension, BurnIn=None):
    # Set type of algorithm: RWM and MALA
    if Algo in ['RWM', 'MALA']:
      self.setAlgoType( Algo )
    else:
      raise RuntimeError('Only "RWM" and "MALA" are supported')
    # Set BurnIn
    self.setBurnIn( BurnIn )
    # Set dimension
    self.setDimension( dimension )

      
  def simulation(self, numberOfSamples, variance, analyticGradient=False, analyseFlag=True, returnSamples=False, initialPosition=[]):
    """
    Main simulation.
    """
    dimension = int(self._dimension)
    initialPosition
    algoType = str(self._algoType)
    counter = 0
    warmUp = 0
    samples = [[] for i in range(dimension) ]
    acceptRate = 0.
    acceptCounter = 0
    # BurnIn flag
    flag = False
    #Acceptance flag
    acceptance = False
    # Print the sample mean and sample covariance
    printMean=False
    # Set method to calculate the gradient for MALA
    if analyticGradient:
      gradientMethod=self.evaluateGradientOfMultimodalGaussian
    else:
      gradientMethod=self.calculateGradient
    #  ---------- HERE YOU HAVE TO CHOOSE ----------
    # Set target distribution 
    #targetDistribution=self.evaluateGaussian
    targetDistribution=self.evaluateMultimodalGaussian
    # ------------------- END ----------------------
    # Temporary samples
    x = []
    # Proposals
    y = numpy.zeros(dimension+1)
    # Mean of the proposals
    mean = [0.0 for i in range(dimension)]
    # Flag for a simple analysis of only one dimension and the prefered dimension
    analyseDim=0
    simpleAnalysis=True
    # Some helpers
    tmp=[]
    sampleMean=[0.0 for i in range(dimension)]
    covarianceMatrix=[[0.0 for i in range(dimension)] for i in range(dimension)]
    if simpleAnalysis is False:
      covarianceMatrixSum=[[0.0 for i in range(dimension)] for i in range(dimension)]
      sampleMeanSum=[0.0 for i in range(dimension)]
    else:
      covarianceMatrixSum=0.0
      sampleMeanSum=0.0
    temp=0.0
    temp2=0.0
    temp3=0.0
       
    # Check dimensions and set initial sample
    if initialPosition and len(initialPosition) == dimension :
      #print('Start simulation with given initial value: {0}'.format(initialPosition))
      for dim in range(dimension):
        tmp = initialPosition[dim]
        x.append( tmp )
      # Last entry is for acceptance flag
      x.append(False)
    # If not initialize, set all entries to zero
    elif not initialPosition:
      #print('Start simulation with initial value zero')
      for dim in range(dimension):
        x.append(0.0)
      # Last entry is for acceptance flag
      x.append(False)
    else:
      raise RuntimeError('Dimension of initial value do not correspond to dimension of the MCMC method')
    
    # Repeat generating new samples
    print('Generate a sample of size: {0} and dimension: {1} with MCMC-type: {2}'.format(numberOfSamples, dimension, algoType))
    while counter < numberOfSamples+1:
      # Calculate the mean of your proposal
      if algoType in ['RWM']:
        for dim in range(dimension):
          mean[dim] = x[dim]
      elif algoType in ['MALA']:
        if analyticGradient is True:
          grad = gradientMethod(targetDistribution, x)
        else:
          grad = gradientMethod(targetDistribution, x )
        for dim in range(dimension):
          mean[dim] = x[dim] + 0.5*variance*grad[dim]
        
      # Generate the proposal
      for dim in range(dimension):
        y[dim]=  self.generateGaussian(mean[dim], variance)
          
          
      # Accept or reject
      tmp = self.acceptanceStep(targetDistribution, gradientMethod, y, x, mean, variance)
      for dim in range(dimension):
        x[dim] = tmp[dim]
      acceptance=tmp[dimension]
      
      # Count steps for the burn-in
      if flag is False:
        warmUp += 1
        #print(warmUp)
      if warmUp == self._burnIn:
        flag = True
      # Reaching the burn-in, we start the counter and sample
      if flag:
        counter += 1
        #print(counter, end='\r')
          
      # Calculate acceptance rate
      if acceptance and flag:
        acceptCounter += 1
      if flag:
        acceptRate = float(acceptCounter) / float(counter)

      # Sample only  after the burn-in
      if flag:
        for dim in range(dimension):
          samples[dim].append( x[dim] )

      percentage = format(100*counter/numberOfSamples, '.1f')
      print('Processing: {0}%'.format(percentage), end='\r')

      # Calculation of sample mean and sample variance of all steps and in addition the mean and covariance for each iteration to plot them at the end
      if analyseFlag is True and counter >= 1:
        if simpleAnalysis is False:
          for dim in range(dimension):
            # Add the new coordinate to the existent sum 
            sampleMeanSum[dim] += x[dim]
            # Divide by the number of added samples
            if counter > 1:
              sampleMean[dim] = sampleMeanSum[dim] / (counter-1)
            elif counter == 1:
              sampleMean[dim] = sampleMeanSum[dim]
        else:
          sampleMeanSum += x[analyseDim]
          if counter>1:
            sampleMean[analyseDim] = sampleMeanSum / (counter -1)
          else:
            sampleMean[analyseDim] = sampleMeanSum
        if simpleAnalysis is False:
          # Use symmetry of covariance matrix (upper triangular matrix)
          for dim1 in range(dimension):
            for dim2 in range(dim1, dimension):
              # sampled covariance matrix
              covarianceMatrixSum[dim1][dim2]+=( x[dim1]-sampleMean[dim1] )*( x[dim2]-sampleMean[dim2] ) 
              # Divide by (numberOfSamples-1) for an unbiased estimate.
              if counter > 1:
                covarianceMatrix[dim1][dim2] = covarianceMatrixSum[dim1][dim2] / (counter -1)
              elif counter == 1:
                covarianceMatrix[dim1][dim2] = covarianceMatrixSum[dim1][dim2]
        else:
          covarianceMatrixSum += (x[analyseDim]- 0.0)**2#sampleMean[analyseDim])**2
          if counter>1:
            covarianceMatrix[0][0] = (counter-1)**-1 * covarianceMatrixSum
          else:
            covarianceMatrix[0][0] = covarianceMatrixSum
            
    print('Acceptance rate: {0}'.format(acceptRate))
          
    if analyseFlag is True:
      #print('Sample variance: {0}'.format(covarianceMatrix))
      #print('Sample mean: {0}'.format(sampleMean))
      helperDim=analyseDim+1
      print('Sample mean of dimension {1}: {0}'.format(sampleMean[analyseDim], helperDim))
      print('Sample variance of dimension {1}: {0}'.format(covarianceMatrix[analyseDim][analyseDim], helperDim))


    if returnSamples:
      returnValue=[]
      returnValue.append(acceptRate)
      returnValue.append(samples)
      returnValue.append(sampleMean)
      returnValue.append(covarianceMatrix)
      return returnValue
    
  def analyseData(self, samples, mean, variance, printName):
    """
    Here we analyse only the first component (for the sake of simplicity)
    """
          
    print('Analysing sampled data...')

    helper=numpy.shape(samples)
    dimension=helper[0]
    numberOfSamples=helper[1]
    # For parallelization (number of processors)
    procs = 2
    results = numpy.array([0.0 for i in range(procs)])
    # Analyse the first component!
    dim=0
    # Maximal number of lag_k autocorrelations
    maxS=int((numberOfSamples-1)/3)
    # lag_k autocorrelation
    autocor = [0.0 for i in range(maxS)]
    autocor[0]=variance[dim][dim]
    # sample frequency
    m=1
    # modified sample variance = autocor[0] + 2 sum_{i}(autocor[i])
    msvar=0.0
    # SEM = sqrt( msvar/numberOfSamples )
    sem=0.0
    # ACT = m * msvar / autocor[0]
    act=0.0
    # ESS = m * numberOfSamples / act
    ess=0.0
          
    temp=0.0
    temp2=0.0

    flagSEM=True
    flagACT=True
    flagESS=True

    # Calculate lag_k for following k's
    evaluation = numpy.arange( maxS )
    evaluation = evaluation[1:]
    evaluation2 = numpy.arange(numberOfSamples)
    for lag in evaluation:

      evaluation2 = evaluation2[:-lag]
      # Do this expensive calculation parallel
      output = mp.Queue()
      morsel = numpy.array_split(evaluation2, procs)
      processes =  []
      for i in range(procs):
        processes.append( mp.Process(target = self.calculateACF, args = (samples[dim], mean[dim], lag, morsel[i], output, ) ) )
      for p in processes:
        p.start()
      for p in processes:
        p.join()
      results = [output.get() for p in processes]
      tmp = numpy.sum(results)
      #tmp=0.0
      #for lag2 in evaluation2:
      #  tmp += (samples[dim][lag2]-mean[dim])*(samples[dim][lag2+lag]-mean[dim])

      autocor[lag] = (numberOfSamples-lag)**-1 * tmp
      if (autocor[lag-1]+autocor[lag])<=0.00:
        maxS = lag
        break
      percentage = format(100*lag/maxS, '.2f')
      print('Processing: {0}%'.format(percentage), end='\r')

    # Calculate the modified sample variance
    evaluation = range( maxS-1 )
    evaluation = evaluation[1:]
    for lag in evaluation:
      msvar += 2*autocor[lag]
      # Calculate the autocovariance function by dividing by variance and multiplying a factor
      autocor[lag] = autocor[lag]/autocor[0]
    msvar += autocor[0]
    # Standard Error of the Mean
    sem = math.sqrt(msvar/numberOfSamples)
    # AutoCorrelation Time
    act = m*msvar/autocor[0]
    # Effective Sample Size
    ess = m*numberOfSamples/act
    # Normalizing autocor[0]
    autocor[0] = 1.0

    print('Modified sample variance: {0}'.format(msvar))   
    print('Standard Error of the Mean: {0}'.format(sem))   
    print('AutoCorrelation Time: {0}'.format(act))   
    print('Effective Sample Size: {0}'.format(ess))   

    #Print some results
    if True:
      lag=range(maxS-1)
      pylab.subplot(311)
      pylab.suptitle('Analysis of the MCMC simulation')
      pylab.plot(lag, autocor[:maxS-1], 'r-.', label='Autocorrelation')
      #pylab.acorr(autocor)
      #pylab.xlabel('lag')
      pylab.ylabel('ACF', fontsize=10)
      pylab.grid(True)
      iterations=range(numberOfSamples)
      pylab.subplot(312)
      pylab.plot(iterations, samples[dim], label='First dimension of samples')
      #pylab.xlabel('Iterations')
      pylab.ylabel('First dim of samples', fontsize=10)
      pylab.grid(True)
      pylab.subplot(313)
      num_bins=100
      n, bins, patches=pylab.hist(samples[dim], num_bins, normed=1, facecolor='green', alpha=0.5, label='Histogram of the first dimension')
      # add a 'best fit' line
      y = 0.5 * mlab.normpdf(bins, -3.0, 1) + 0.5 * mlab.normpdf(bins, 3.0, 1)
      #y = 0.5 * mlab.normpdf(bins, -1.0, 0.5) + 0.5 * mlab.normpdf(bins, 1.0, 0.8)
      plt.plot(bins, y, 'r--')
      pylab.xlabel('First dimension of samples', fontsize=10)
      pylab.ylabel('Relative frequency', fontsize=10)
      pylab.grid(True)
      pylab.savefig(printName)
      pylab.clf()
      newPrintName = printName.replace(".png", "ScatterPlot.png")
      #print(newPrintName)
      self.scatterPlot3D(samples, newPrintName)
      #pylab.show()

    return act

  def calculateACF(self,samples, mean, lag, array, output):
    """
    A helper function to calculate the autocorrelation coefficient parallel
    """
    tmp=0.0
    for lag2 in array:
      tmp += (samples[lag2]-mean)*(samples[lag2+lag]-mean)
    output.put(tmp)
      
  def setAlgoType(self, Algo):
    self._algoType = Algo
  
  def setBurnIn(self, BurnIn):
    if BurnIn is not None:
      self._burnIn = BurnIn
    else:
      self._burnIn = 10000
      
  def setDimension(self, dimension):
    self._dimension = dimension
    
  def evaluateMultimodalGaussian(self, position=[]):
    """
    Implement the target distribution without normalization constants. 
    Here we have the multimodal example of Roberts and Rosenthal (no product measure)
    """
    m = 3.0
    tmp = []
    for dim in range( self._dimension ):
      tmp.append( position[dim]**2 )
    tmp= tmp[1:]
    return math.exp( - 0.5 * ( (position[0]-m)**2 + math.fsum(tmp) )) + math.exp( - 0.5 * ( (position[0]+m)**2 + math.fsum(tmp) ))
 
  def evaluateGradientOfMultimodalGaussian(self, evaluateMultiModalGaussian, position=[]):
    """
    Calculates the analytical gradient
    """
    m=3.0
    tmp=[]
    grad=[]
    interval=range( self._dimension )

    for dim in interval:
      tmp.append( position[dim]**2 )
    tmp=tmp[1:]
    grad.append( -(position[0]-m)*math.exp( - 0.5 * ( (position[0]-m)**2 + math.fsum(tmp) )) -(position[0]+m)*math.exp( - 0.5 * ( (position[0]+m)**2 + math.fsum(tmp) ))/evaluateMultiModalGaussian(position) )
    interval=interval[1:]
    for dim in interval:
      grad.append( (-position[dim]*math.exp( - 0.5 * ( (position[0]-m)**2 + math.fsum(tmp) )) - position[dim]*math.exp( - 0.5 * ( (position[0]+m)**2 + math.fsum(tmp) )) )/evaluateMultiModalGaussian(position) ) 
    
    return grad

  def evaluateGaussian(self, position=[]):
    """
    Simple multi-dimensional Gaussian
    """
    mean1=-1.0
    mean2=1.0
    variance1=0.5
    variance2=0.8
    tmp=[]
    tmp2=[]
    #for dim in range(self._dimension):
    #  tmp.append( (position[dim]-mean1)**2 )
    #  tmp2.append( (position[dim]-mean2)**2 )
    #return variance1**(-0.5*self._dimension)*math.exp( -0.5*math.fsum(tmp)*variance1**-1 ) + variance2**(-0.5*self._dimension)*math.exp( -0.5*math.fsum(tmp2)*variance2**-1 ) 
    value=1.0
    for dim in range(self._dimension):
      value *= variance1**(-0.5) * math.exp(-0.5*(position[dim]-mean1)**2 *variance1**-2) + variance2**(-0.5) * math.exp(-0.5*(position[dim]-mean2)**2 *variance2**-2)
    return value
      
  def calculateGradient(self, evaluateTargetDistribution, position):
    """
    Calculate the gradient of the logarithm of your target distribution by finite differences
    """
    # Check dimension
    #if not position or len(position) != self._dimension+1:
    #  raise RuntimeError('In calculateGradient: Empty argument or wrong dimension')
    #else:
    if True:
      h = 1e-10
      grad = []
      shiftedpos1 = []
      shiftedpos2 = []
      for dim in range(self._dimension):
        shiftedpos1.append( position[dim] )
        shiftedpos2.append( position[dim] )
      for dim in range(self._dimension):
        shiftedpos1[dim] += h
        shiftedpos2[dim] -= h
        tmp1=evaluateTargetDistribution(shiftedpos1)
        # Check if value of the target distribution is not to small
        if tmp1 <= 0.0:
          tmp1=h
        tmp2=evaluateTargetDistribution(shiftedpos2)
        if tmp2 <= 0.0:
          tmp2=h
        grad.append(0.5 * h**-1 * ( math.log(tmp1)-math.log(tmp2) ))
        shiftedpos1[dim] -= h
        shiftedpos2[dim] += h
      return grad
        
  def generateGaussian(self, mean, variance):
    """
    Generates one-dimensional Gaussian random variable 
    """
    gauss=random.gauss(mean, math.sqrt(variance))
    return gauss
    
  def acceptanceStep(self, evaluateTargetDistribution, calculateGradient,  proposal=[], position=[], mean=[], variance=None):
    """
    Determines wether the proposal is accepted or rejected.
    """
    u = random.uniform(0,1)
    ratio = ( evaluateTargetDistribution(proposal) )/( evaluateTargetDistribution(position) )
    # Calculate the ratio of the transition kernels
    if self._algoType in ['MALA']:
      tmp2=0.0
      grad = calculateGradient( evaluateTargetDistribution, proposal )
      for dim in range(self._dimension):
        tmp = proposal[dim] + 0.5*variance*grad[dim]
        tmp2 += (mean[dim]-proposal[dim])**2 - (tmp-position[dim])**2
      ratio *= math.exp(-0.5*variance**-1 * tmp2)
    if u < ratio:
      proposal[self._dimension]=True
      return proposal
    else:
      position[self._dimension]=False
      return position

  def scatterPlot3D(self, samples, printName):
   """
   Plot samples in 3D as cloud.
   """
   vec = [ [ [],[],[] ] for i in range(5) ] 
   xs = []
   ys = []
   zs = []
   
   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')

   # Control number and dimension of samples
   helper=numpy.shape(samples)
   dimension=helper[0]
   numberOfSamples=helper[1]
   if numberOfSamples < 5000:
     return 0
     
   #print(samples)
   #print(numberOfSamples)
   #print(dimension)
   
   # Sort your samples
   for it in range(numberOfSamples):
     if (it < 1000):
       vec[0][0].append(samples[0][it])
       vec[0][1].append(samples[1][it])
       vec[0][2].append(samples[2][it])
     elif (it >= 1000 and it < 2000):
       vec[1][0].append(samples[0][it])
       vec[1][1].append(samples[1][it])
       vec[1][2].append(samples[2][it])
     elif (it >= 2000 and it < 3000):
       vec[2][0].append(samples[0][it])
       vec[2][1].append(samples[1][it])
       vec[2][2].append(samples[2][it])
     elif (it >= 3000 and it < 4000):
       vec[3][0].append(samples[0][it])
       vec[3][1].append(samples[1][it])
       vec[3][2].append(samples[2][it])
     elif (it >= 4000 and it < 5000):
       vec[4][0].append(samples[0][it])
       vec[4][1].append(samples[1][it])
       vec[4][2].append(samples[2][it])
     else:
       break
       
   #ax.scatter(samples[0], samples[1], samples[2], 'b', marker='.')
   ax.set_xlabel('X Label')
   ax.set_ylabel('Y Label')
   ax.set_zlabel('Z Label')
     
   #plt.show()
   # Define some colors
   #color = ['r', 'y', 'g', 'c', 'b']
  
   # Make scatterplots
   for c, sample in [('r', vec[0]), ('y', vec[1]), ('g', vec[2]), ('c', vec[3]), ('b', vec[4])]:
     xs = sample[0]
     ys = sample[1]
     zs = sample[2]
     ax.scatter(xs, ys, zs, c=c, marker='o')
     
   ax.set_xlabel('X ')
   ax.set_ylabel('Y ')
   ax.set_zlabel('Z ')
   pylab.savefig(printName, dpi=200)
   #plt.show()
   pylab.clf()


     
  
   
	
#makeDiagramm('MALA', 10, 1000, 'GEANY01', 'ACT')
#algo=Algo('RWM', 3)
#samples=algo.simulation(10000, 2.2, analyticGradient=False, analyseFlag=False, returnSamples=True)
#algo.scatterPlot3D(samples[1])