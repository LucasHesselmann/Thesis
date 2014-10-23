#! /usr/bin/env python3.4

import sys
import os
import numpy
import pylab
import random
import math
from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits import mplot3d


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
  k=1
  dimensionIter=range(dimension)
  threshold = 150
  
  # Some preparation
  directory = '{0}_{1}_{2}_{3}'.format(directoryName, string, dimension, numberOfSamples)
  os.makedirs(directory) 
 
  algo = Algo(string, dimension, 50)
  # Generate the proposal variances and scale them according to the dimension
  helper=numpy.logspace(-2, 5, 50, True, 2.0) #RWM in 5D
  #helper = numpy.logspace(-1, 4, 6, True, 2.0)
  for var in helper:
    variances.append(var/dimension)
  # Initialize the init value
  for dim in dimensionIter:
    init.append(random.gauss(0.0, 0.2))
  # Simulate for each variance in variances.
  for var in variances:
    result=algo.simulation(numberOfSamples, var, False, True, True, init)
    tmp = format(result[0], '.2f')
    if result[0]>0.79:
      continue
    fileName=os.path.join(directory, '{0}_{1}_{2}.png'.format(string, k, tmp))
    #print('Filename: ', fileName)
    tmp2=algo.analyseData(result[1],[0], result[3], fileName)  
    if tmp2/dimension < threshold:
      autocor.append(tmp2/dimension)
      acceptRate.append(tmp)
      k+=1
    print('-----------------------------------------------------------')
    print('Acceptance rate: {0}'.format(tmp))
    print('Integrated autocorrelation: {0}'.format(tmp2/dimension))
    print('-----------------------------------------------------------')
    #algo.plotDistribution(result[1], 2)
    if result[0]<0.02:
      break
  pylab.figure()
  pylab.plot(acceptRate, autocor, 'ro', label='Convergence time')
  pylab.ylabel('convergence time')
  pylab.xlabel('acceptance rate')
  pylab.grid(True)
  fileName1=os.path.join(directory, '{0}_{1}_Dim{2}.png'.format(string, resultName, dimension))
  pylab.savefig(fileName1) 
  pylab.clf()
                          
  
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
    sampleCounter = 0
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
    # Some helpers
    tmp=[]
    covarianceMatrixSum=[[0.0 for i in range(dimension)] for i in range(dimension)]
    covarianceMatrix=[[0.0 for i in range(dimension)] for i in range(dimension)]
    temp=0.0
    temp2=0.0
    temp3=0.0
    sampleMeanSum=[0.0 for i in range(dimension)]
    sampleMean=[0.0 for i in range(dimension)]
    
    # Check dimensions and set initial sample
    if initialPosition and len(initialPosition) == dimension :
      print('Start simulation with given initial value: {0}'.format(initialPosition))
      for dim in range(dimension):
        tmp = initialPosition[dim]
        x.append( tmp )
      # Last entry is for acceptance flag
      x.append(False)
    # If not initialize, set all entries to zero
    elif not initialPosition:
      print('Start simulation with initial value zero')
      for dim in range(dimension):
        x.append(0.0)
      # Last entry is for acceptance flag
      x.append(False)
    else:
      raise RuntimeError('Dimension of initial value do not correspond to dimension of the MCMC method')
    
    # Repeat generating new samples
    print('Generate a sample of size: {0} and dimension: {1} with MCMC-type: {2}'.format(numberOfSamples, dimension, algoType))
    while sampleCounter < numberOfSamples:
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
      for dim in range(dimension):
        tmp = self.acceptanceStep(self.evaluateMultimodalGaussian, y, x)
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
        #print(counter)
          
      # Calculate acceptance rate
      if acceptance:
        acceptCounter += 1
      acceptRate = float(acceptCounter) / float(warmUp+counter)

      # Sample only  after the burn-in
      if flag:
        sampleCounter += 1
        for dim in range(dimension):
          samples[dim].append( x[dim] )

      # Calculation of sample mean and sample variance of all steps and in addition the mean and covariance for each iteration to plot them at the end
      if analyseFlag is True and counter >= 1:
        for dim in range(dimension):
          # Add the new coordinate to the existent sum 
          sampleMeanSum[dim] += x[dim]
          # Divide by the number of added samples
          if counter > 1:
            sampleMean[dim] = sampleMeanSum[dim] / (counter-1)
          elif counter == 1:
            sampleMean[dim]=sampleMeanSum[dim]
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

    print('Acceptance rate: {0}'.format(acceptRate))
          
    if analyseFlag is True:
      #print('Sample variance: ', covarianceMatrix)
      #print('Sample mean: ', sampleMean)
      print('Sample mean of first dimension: {0}'.format(sampleMean[0]))
      print('Sample variance of first dimension: {0}'.format(covarianceMatrix[0][0]))


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
      maxS=2000
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
    for lag in evaluation:
      tmp=0.0
      for lag2 in evaluation[:-lag]:
        tmp += (samples[dim][lag2]-mean[dim])*(samples[dim][lag2+lag]-mean[dim])
      autocor[lag] = (numberOfSamples-lag)**-1 * tmp
      if (autocor[lag-1]+autocor[lag])<=0.0:
        maxS = lag
        break
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
    autocor[0] = 0

    print('Modified sample variance: {0}'.format(msvar))   
    print('Standard Error of the Mean: {0}'.format(sem))   
    print('AutoCorrelation Time: {0}'.format(act))   
    print('Effective Sample Size: {0}'.format(ess))   

    if maxS>50:
      maxS=50

    #Print some results if possible
    if True:
      lag=range(maxS)
      pylab.subplot(211)
      pylab.bar(lag, autocor[:maxS], 0.01, label='Autocorrelation')
      pylab.ylim([-0.1, 1.0])
      #pylab.acorr(autocor)
      pylab.xlabel('lag')
      pylab.ylabel('ACF')
      pylab.grid(True)
      iterations=range(numberOfSamples)
      pylab.subplot(212)
      pylab.plot(iterations, samples[dim], label='First dimension of samples')
      pylab.xlabel('Iterations')
      pylab.ylabel('First dim of samples')
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
    Implement the taget distribution without normalization constants. 
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
      grad.append( -(position[0])*math.exp( - 0.5 * ( (position[0]-m)**2 + math.fsum(tmp) )) -(position[0])*math.exp( - 0.5 * ( (position[0]+m)**2 + math.fsum(tmp) ))/evaluateMultiModalGaussian(position) ) 
    
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
