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
#from mpl_toolkits import mplot3d


def multiDimDiagramm(string, numberOfSamples, directoryName, resultName):
  """
  Plot the convergence time for several dimensions.
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
  # Make diagramm for dimension 5, 10, 15, 20, 30
  dimensions=[5, 10, 15, 20]
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
  pylab.plot(acceptRate[0], autocor[0], colours[0], acceptRate[1], autocor[1], colours[1], acceptRate[2], autocor[2], colours[2], acceptRate[3], autocor[3], colours[3],label='Convergence time')
  pylab.ylabel('convergence time')
  pylab.xlabel('acceptance rate')
  pylab.ylim([0.0, 550])
  pylab.grid(True)
  fileName1=os.path.join(directory, '{0}_{1}.png'.format(string, resultName))
  pylab.savefig(fileName1) 
  pylab.clf()
                          

 
def makeDiagramm(string, dimension, numberOfSamples, directoryName, resultName):
  """
  Make a diagramm to show behavior of the algo depending on the scaling of the variance.
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
 
  algo = Algo(string, dimension, 100)
  # Generate the proposal variances and scale them according to the dimension
  if string in ['MALA']:
    if dimension == 1:
      helper=numpy.logspace(0.05, 3.7, 10, True, 2.0) #MALA in 1D
    elif dimension == 2:
      helper=numpy.logspace(0.15, 4.4, 12, True, 2.0) #MALA in 2D
    elif dimension == 5:
      helper=numpy.logspace(0.2, 4.5, 6, True, 2.0) #MALA in 5D
    elif dimension == 10:
      helper=numpy.logspace(0.2, 4.5, 6, True, 2.0) #MALA in 10D
    elif dimension == 15:
      helper=numpy.logspace(0.3, 4.5, 6, True, 2.0) #MALA in 15D
    elif dimension == 20:
      helper=numpy.logspace(0.3, 4.5, 6, True, 2.0) #MALA in 20D
    else:
      helper=numpy.logspace(0.0, 5.0, 12, True, 2.0)
  elif string in ['RWM']:
    if dimension == 1:
      helper=numpy.logspace(-0.8, 15, 6, True, 2.0) #RWM in 1D
    elif dimension == 2:
      helper=numpy.logspace(-0.7, 8, 10, True, 2.0) #RWM in 2D
    elif dimension == 5:
      helper=numpy.logspace(-0.7, 5.4, 7, True, 2.0) #RWM in 5D
    elif dimension == 10:
      helper=numpy.logspace(-0.7, 5.1, 7, True, 2.0) #RWM in 10D
     #helper=numpy.logspace(1.4, 2.8, 2, True, 2.0) #RWM in 10D
    elif dimension == 15:
      helper=numpy.logspace(-0.6, 4.9, 7, True, 2.0) #RWM in 15D
    elif dimension == 20:
      helper=numpy.logspace(-0.5, 4.8, 6, True, 2.0) #RWM in 20D
    elif dimension == 30:
      helper=numpy.logspace(-0.4, 4.7, 6, True, 2.0) #RWM in 30D
    elif dimension == 50:
      helper=numpy.logspace(-0.4, 4.6, 15, True, 2.0) #RWM in 50D
    else:
      helper=numpy.logspace(-1, 9, 11, True, 2.0)
  for var in helper:
    variances.append(var/dimension)
  # Initialize the init value
  init.append(random.gauss(2.5, 0.2))
  for dim in dimensionIter[1:]:
    init.append(random.gauss(0.0, 0.2))
  # Simulate for each variance in variances.
  for var in variances:
    for i in range(repitition):
      result=algo.simulation(numberOfSamples, var, True, True, True, init)
      tmp = format(result[0], '.2f')
      if result[0]>0.80:
        tmpVec2[i]=tmp
        tmpVec[i]=threshold+1
        continue
      fileName=os.path.join(directory, '{0}_{1}-{2}_{3}.png'.format(string, k, i, tmp))
      tmp2=algo.analyseData(result[1], [0], result[3], fileName)  
      tmpVec[i] = (tmp2/dimension**(convOrder))
      tmpVec2[i] = (tmp)
      print('-----------------------------------------------------------')
      print('Round {0}'.format(i))
      print('Acceptance rate: {0}'.format(tmp))
      print('Integrated autocorrelation: {0}'.format(tmp2/dimension**(convOrder)))
      print('-----------------------------------------------------------')
    # Take the median of acceptance rate and convergence time
    tmp2=numpy.median(tmpVec2)
    if tmp2 > 0.80:
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
    if tmp2<0.01:
      break
  #pylab.figure()
  #pylab.plot(acceptRate, autocor, 'r', label='Convergence time')
  #pylab.ylabel('convergence time')
  #pylab.xlabel('acceptance rate')
  #pylab.grid(True)
  #fileName1=os.path.join(directory, '{0}_{1}_Dim{2}.png'.format(string, resultName, dimension))
  #pylab.savefig(fileName1) 
  #pylab.clf()
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
          grad = self.evaluateGradientOfMultimodalGaussian(self.evaluateMultimodalGaussian, x)
        else:
          grad = self.calculateGradient( self.evaluateMultimodalGaussian, x )
        for dim in range(dimension):
          mean[dim] = x[dim] + 0.5*variance*grad[dim]
        
      # Generate the proposal
      for dim in range(dimension):
        y[dim]=  self.generateGaussian(mean[dim], variance)
          
          
      # Accept or reject
      tmp = self.acceptanceStep(self.evaluateMultimodalGaussian, y, x)
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
    # Analyse the first component!
    dim=0
    # Maximal number of lag_k autocorrelations
    if int((numberOfSamples-1)/3) > 2000:
      #maxS=2000
      maxS=int((numberOfSamples-1)/3)
    else:
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
    evaluation = range( maxS )
    evaluation = evaluation[1:]
    evaluation2 = range(numberOfSamples)
    for lag in evaluation:
      tmp=0.0
      for lag2 in evaluation2[:-lag]:
        tmp += (samples[dim][lag2]-mean[dim])*(samples[dim][lag2+lag]-mean[dim])
      autocor[lag] = (numberOfSamples-lag)**-1 * tmp
      if (autocor[lag-1]+autocor[lag])<=0.03:
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

    #if maxS>100:
     # maxS=100

    #Print some results if possible
    if True:
      lag=range(maxS-1)
      pylab.subplot(311)
      pylab.suptitle('Analysis of the MCMC simulation')
      #pylab.bar(lag, autocor[:maxS], 0.001, label='Autocorrelation')
      pylab.plot(lag, autocor[:maxS-1], 'r-.', label='Autocorrelation')
      #pylab.ylim([-0.1, 1.1])
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
      plt.plot(bins, y, 'r--')
      pylab.xlabel('First dimension of samples', fontsize=10)
      pylab.ylabel('Relative frequency', fontsize=10)
      pylab.grid(True)
      pylab.savefig(printName)
      pylab.clf()
      #pylab.show()

    return act
      
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
    mean1=-.5
    mean2=1.0
    variance1=2.0
    variance2=1.0
    tmp=[]
    tmp2=[]
    for dim in range(self._dimension):
      tmp.append( (position[dim]-mean1)**2 )
      #tmp2.append( (position[dim]-mean2)**2 )
    return math.exp( -0.5*math.fsum(tmp)*variance1**-1 ) #+ math.exp( -0.5*math.fsum(tmp2)*variance2**-1 )
      
  def calculateGradient(self, evaluateTargetDistribution, position):
    """
    Calculate the gradient of the logarithm of your target distribution by finite differences
    """
    # Check dimension
    if not position or len(position) != self._dimension+1:
      raise RuntimeError('In calculateGradient: Empty argument or wrong dimension')
    else:
      h = 1e-8
      grad = []
      shiftedpos1 = []
      shiftedpos2 = []
      for dim in range(self._dimension):
        shiftedpos1.append( position[dim] )
        shiftedpos2.append( position[dim] )
      for dim in range(self._dimension):
        shiftedpos1[dim] += h
        shiftedpos2[dim] -= h
        grad.append(0.5 * h**-1 * ( math.log(evaluateTargetDistribution(shiftedpos1))-math.log(evaluateTargetDistribution(shiftedpos2)) ))
        shiftedpos1[dim] -= h
        shiftedpos2[dim] += h
      return grad
        
  def generateGaussian(self, mean, variance):
    """
    Generates one-dimensional Gaussian random variable 
    """
    gauss=random.gauss(mean, math.sqrt(variance))
    return gauss
    
  def acceptanceStep(self, evaluateTargetDistribution, proposal=[], position=[]):
    """
    Determines wether the proposal is accepted or rejected.
    """
    u = random.uniform(0,1)
    ratio = ( evaluateTargetDistribution(proposal) )/( evaluateTargetDistribution(position) )
    if u < ratio:
      proposal[self._dimension]=True
      return proposal
    else:
      position[self._dimension]=False
      return position

  def plotDistribution(self, samples, dimension):
    """
    Plot the results in 1D and 2D.
    """
    if dimension == 2:
      fig=pylab.figure()
      ax = Axes3D(fig)
      #ax = fig.add_subplot(111, projection='3d')
      x = samples[0]
      y = samples[1]
      hist, xedges, yedges = numpy.histogram2d(x, y, bins=40)
      elements = (len(xedges) - 1) * (len(yedges) - 1)
      xpos, ypos = numpy.meshgrid(xedges[:-1]+0.1, yedges[:-1]+0.1)
      xpos = xpos.flatten()
      ypos = ypos.flatten()
      zpos = numpy.zeros(elements)
      dx = 0.5 * numpy.ones_like(zpos)
      dy = dx.copy()
      dz = hist.flatten()
      ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
      pylab.show()
      #pylab.clf()
    if (dimension == 1):
      #pylab.figure()
      pylab.hist(samples[0], bins=300, normed=True)
      pylab.show()
      #pylab.clf()
