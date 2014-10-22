#! /usr/bin/env python3.4

import sys
import os
import numpy
import pylab
import random
import math
from mpl_toolkits.mplot3d import Axes3D

def makeDiagramm(string, dimension, numberOfSamples, directoryName, resultName):
  """
  Make a diagramm to show behavior of the algo depending on the scaling of the variance.
  """
  result=[]
  intAutocor=[]
  acceptRate=[]
  steps=[]
  init=[]
  variances=[]
  k=1
  dimensionIter=range(dimension)
  
  # Some preparation
  directory = '{0}_{1}_{2}_{3}'.format(directoryName, string, dimension, numberOfSamples)
  os.makedirs(directory) 
 
  algo = Algo(string, dimension)
  # Generate the proposal variances and scale them according to the dimension
  helper=numpy.linspace(0.01, 30, 50)
  for var in helper:
    variances.append(var/dimension)
  # Initialize the init value
  for dim in dimensionIter:
    init.append(random.gauss(-3.0, 1.0))
  # Simulate for each variance in variances.
  for var in variances:
    result=algo.simulation(numberOfSamples, var, False, True, True, init)
    tmp = format(result[0], '.2f')
    if result[0]>0.90:
      continue
    acceptRate.append(tmp)
    fileName=os.path.join(directory, '{0}_{1}.png'.format(k, tmp))
    #print('Filename: ', fileName)
    tmp2=algo.analyseData(result[1],-3, fileName)  
    intAutocor.append(tmp2)
    k+=1
    print('-----------------------------------------------------------')
    print('Integrated autocorrelation: ', tmp2)
    print('Acceptance rate: ', tmp)
    print('-----------------------------------------------------------')
    if result[0]<0.05:
      break
  pylab.figure()
  pylab.plot(acceptRate, intAutocor, 'ro', label='Convergence time')
  pylab.ylabel('convergence time')
  pylab.xlabel('acceptance rate')
  pylab.grid(True)
  fileName1=os.path.join(directory, '{0}.png'.format(resultName))
  pylab.savefig(fileName1)   
                          
  
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

      
  def simulation(self, numberOfSamples, variance, analyticGradient=False, analyseFlag=True, returnSamples=False, initialPosition=[], printName=None):
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
    mean = []
    # Some helpers
    tmp=[]
    covarianceMatrix=[[0.0 for i in range(dimension)] for i in range(dimension)]
    covarianceMatrix2=[[0.0 for i in range(dimension)] for i in range(dimension)]
    temp=0.0
    temp2=0.0
    temp3=0.0
    sampleMeanNorm=0.0
    sampleMeanNormVec=[]
    meanNorm=0.0
    meanNormVec=[]
    varianceVec=[]
    varianceVec2=[]
    sampleMeanSum=[0.0 for i in range(dimension)]
    sampleMean=[0.0 for i in range(dimension)]


    # Check dimensions and set initial sample
    if initialPosition and len(initialPosition) == dimension :
      print('Start simulation with given initial value: ', initialPosition)
      for dim in range(dimension):
        tmp = initialPosition[dim]
        x.append( tmp )
      # Last entry is for acceptance flag
      x.append(False)
    # If not initialize, set all entries to zero
    elif not initialPosition:
      print('Start simulation with initial value zero')
      for dim in range(dimension):
        x.append(0.)
      # Last entry is for acceptance flag
      x.append(False)
    else:
      raise RuntimeError('Dimension of initial value do not correspond to dimension of the MCMC method')
                
    # Repeat generating new samples
    print('Generate a sample of size:', numberOfSamples)
    while sampleCounter < numberOfSamples:
      # Calculate the mean of your proposal
      if algoType in ['RWM']:
        for dim in range(dimension):
          mean.append(x[dim])
      elif algoType in ['MALA']:
        if analyticGradient is True:
          grad = self.evaluateGradientOfMultimodalGaussian(self.evaluateMultimodalGaussian, x)
        else:
          grad = self.calculateGradient( self.evaluateGaussian, x )
        for dim in range(dimension):
          mean.append(x[dim] + 0.5*variance*grad[dim] )
            
      # Generate the proposal
      for dim in range(dimension):
        y[dim]=( self.generateGaussian(mean[dim], variance) )
 
      # Accept or reject
      for dim in range(dimension):
        tmp = self.acceptanceStep(self.evaluateGaussian, y, x)
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
        if counter == 1:
          for dim in range(dimension):
            # Take the sum of each coordinate for each dimension
            sampleMeanSum[dim]=x[dim]
            # In the first step: the coordinate is the mean
            sampleMean[dim]=sampleMeanSum[dim]
        elif counter > 1:
          for dim in range(dimension):
            # Add the new coordinate to the existent sum 
            sampleMeanSum[dim]= x[dim] + sampleMeanSum[dim]
            # Divide by the number of added samples
            sampleMean[dim]= sampleMeanSum[dim] /(counter-1)
        # Calculate the norm of the sample mean and the actual norm of each sample and calculate the covariance matrix
        sampleMeanNorm=0.0
        meanNorm=0.0
        # Use symmetry of covariance matrix
        for dim1 in range(dimension):
          for dim2 in range(dim1, dimension):
            # sampled covariance matrix
            covarianceMatrix[dim1][dim2]+=( x[dim1]-sampleMean[dim1] )*( x[dim2]-sampleMean[dim2] ) 
            # covariance matrix for each sample
            covarianceMatrix2[dim1][dim2]=( x[dim1]-sampleMean[dim1] )*( x[dim2]-sampleMean[dim2] ) 
          sampleMeanNorm+=sampleMean[dim1]**2
          meanNorm+=x[dim1]**2
        sampleMeanNormVec.append( math.sqrt( sampleMeanNorm ) )
        meanNormVec.append( math.sqrt( meanNorm ) )
        #Calculate Frobenius-norm of covariance matrix (use symmetry of covariance matrix)
        temp=0.0
        temp2=0.0
        for dim1 in range(dimension):
          for dim2 in range(dim1, dimension):
            if dim1==dim2:
              temp += math.fabs(covarianceMatrix[dim1][dim2])**2
              temp2 += math.fabs(covarianceMatrix2[dim1][dim2])**2
            elif dim2 > dim1:
              temp += 2*math.fabs(covarianceMatrix[dim1][dim2])**2
              temp2 += 2*math.fabs(covarianceMatrix2[dim1][dim2])**2
        if counter == 1:
          varianceVec.append( math.sqrt( temp ) )
          varianceVec2.append( math.sqrt( temp2 ) )
        elif counter > 1:
          varianceVec.append( (counter-1)**-1 *math.sqrt( temp ) )
          varianceVec2.append( math.sqrt( temp2 ) )

    print('The variance for the proposals: ', variance)
    #print('Acceptance rate: ', acceptRate)

    if analyseFlag is True:
      print('Sample mean (norm): ', sampleMeanNormVec[numberOfSamples-1])
      print('Sample mean of first dimension: ', sampleMean[0])
      #print('Sample covariance (norm): ', varianceVec[numberOfSamples-1])
    if printMean is True and analyseFlag is True:
      # Plot mean and covariance
      iterations=range(numberOfSamples)
      # A histogram is only possible in 1D and 2D
      if dimension == 2:
        fig=pylab.figure()
        ax = fig.add_subplot(111, projection='3d')
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
      pylab.figure()
      if (dimension == 1):
        pylab.hist(samples, bins=300, normed=True)
        pylab.show()
      pylab.subplot(211)
      pylab.plot(iterations, sampleMeanNormVec, label='Sample mean')
      pylab.grid(True)
      pylab.plot(iterations, meanNormVec, label='Mean')
      pylab.ylabel('Mean')
      pylab.grid(True)
      #pylab.legend()
      pylab.subplot(212)      
      pylab.plot(iterations, varianceVec , label='Sample variance')
      pylab.grid(True)
      pylab.plot(iterations, varianceVec2, label='Variance')
      pylab.grid(True)
      pylab.ylabel('Covariance')
      #pylab.legend()
      helper=format(acceptRate, '.2f')
      if printName is None:
        printName = algoType + '_' + str(dimension) + '_' + str(numberOfSamples) + '_' +  str(helper) + '.png'
      pylab.savefig(printName)

    if returnSamples:
      returnValue=[]
      returnValue.append(acceptRate)
      returnValue.append(samples)
      returnValue.append(sampleMean[0])
      return returnValue
    
  def analyseData(self, samples, mean, printName):

    print('Analysing sampled data...')

    helper=numpy.shape(samples)
    dimension=helper[0]
    numberOfSamples=helper[1]
    #print('dimension: ', dimension)
    #print('numberOfSamples: ', numberOfSamples)
    # lag_k autocorrelation
    autocor = [0.0 for i in range(int((numberOfSamples-1)/3))]
    autocor[0]=1.0
    intAutocor=[]
    intAutocor.append(0.0)
    temp=0.0
    temp2=0.0
    temp3=0.0
    c=8.0
    window=-1.
    M=-1
    MWindow=[]
    flag=True
    calceffectiveSize=True
    calcintAutocor=False

    evaluation = range( int((numberOfSamples-1)/4) )
    evaluation = evaluation[1:]
    # Calculate the effective sample size by the lag-1 autocorrelations.
    if calceffectiveSize is True:
      
    
    # Calculate the autocorrelation and integrated autocorrelation time for the first component (dim=0)
    # Only evaluate the autocorrelation function for lag < numberOfSamples / 4
    dim=0
    if calcintAutocor is True:
      temp2=0.0
      for sample in range(numberOfSamples):
        temp2+=(samples[dim][sample]-mean)**2
      for lag in evaluation:
        temp=0.0
        for lag2 in range(numberOfSamples-lag):
          temp+= (samples[dim][lag2]-mean)*(samples[dim][lag2+lag]-mean)
        autocor[lag]=temp/temp2
        # Calculate the integrated autocorrelation time
        temp3+=autocor[lag]
        if flag is True:
          intAutocor.append((0.5+temp3))
          if (float(lag/c) > intAutocor[lag]):
            window = intAutocor[lag]
            M=lag
            flag=False
            break
      if flag is True:
        M = evaluation[-1]
        window = intAutocor[-1]
      if M > 100:
        autocorcounter=100
      else:
        autocorcounter=M
      #estimator= ( -1 ) / ( math.log( math.fabs( (2*intAutocor[evaluation[-1]]-1)/(2*intAutocor[evaluation[-1]]+1)  ) ) )
      #print('The estimated convergence time: ', estimator)
      #print('The integrated autocorrelation: ', intAutocor[evaluation[-1]])
      print('The integrated autocorrelation with window algorithm: ', window, ' with a window: ', M)

    #Print some results if possible
    if calcintAutocor is True:
      lag=range(autocorcounter)
      pylab.subplot(311)
      pylab.plot(lag, autocor[:autocorcounter], label='Autocorrelation')
      pylab.xlabel('lag')
      pylab.ylabel('ACF')
      pylab.grid(True)
      #pylab.legend()
      MWindow=range(M)
      pylab.subplot(312)
      pylab.plot(MWindow, intAutocor[:M], label='Integrated autocorrelation')
      pylab.xlabel('Iterations')
      pylab.ylabel('IACF')
      pylab.grid(True)
      #pylab.legend()
      iterations=range(numberOfSamples)
      pylab.subplot(313)
      pylab.plot(iterations, samples[dim], label='First dimension of samples')
      pylab.xlabel('Iterations')
      pylab.ylabel('First dim of samples')
      pylab.grid(True)
      pylab.savefig(printName)
      #pylab.show()

    return window/dimension


      
  def setAlgoType(self, Algo):
    self._algoType = Algo
  
  def setBurnIn(self, BurnIn):
    if BurnIn is not None:
      self._burnIn = BurnIn
    else:
      self._burnIn = 1000
      
  def setDimension(self, dimension):
    self._dimension = dimension
    
  def evaluateMultimodalGaussian(self, position=[]):
    """
    Implement the taget distribution without normalization constants. 
    Here we have the multimodal example of Roberts and Rosenthal (no product measure)
    """
    m = 0.0
    tmp = []
    for dim in range( self._dimension ):
      tmp.append( position[dim]**2 )
    tmp= tmp[1:]
    return math.exp( - 0.5 * ( (position[0]-m)**2 + math.fsum(tmp) )) + math.exp( - 0.5 * ( (position[0]+m)**2 + math.fsum(tmp) ))
 
  def evaluateGradientOfMultimodalGaussian(self, evaluateMultiModalGaussian, position=[]):
    """
    Calculates the analytical gradient
    """
    m=0.0
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
    mean1=-3.0
    mean2=1.0
    variance1=2.0
    variance2=5.0
    tmp=[]
    tmp2=[]
    for dim in range(self._dimension):
      tmp.append( (position[dim]-mean1)**2 )
      #tmp2.append( (position[dim]-mean2)**2 )
    return math.exp( -0.5*math.fsum(tmp)*variance1**-1 )# + math.exp( -0.5*math.fsum(tmp2)*variance2**-1 )
      
  def calculateGradient(self, evaluateTargetDistribution, position):
    """
    Calculate the gradient of the logarithm of your target distribution by finite differences
    """
    # Check dimension
    if not position or len(position) != self._dimension+1:
      raise RuntimeError('In calculateGradient: Empty argument or wrong dimension')
    else:
      h = 1e-5
      grad = []
      shiftedpos1 = []
      shiftedpos2 = []
      shiftvector = []
      for dim in range(self._dimension):
        shiftedpos1.append( position[dim] )
        shiftedpos2.append( position[dim] )
      for dim in range(self._dimension):
        shiftedpos1[dim] += h
        shiftedpos2[dim] -= h
        grad.append(2.0* h**-1 * ( math.log(evaluateTargetDistribution(shiftedpos1))-math.log(evaluateTargetDistribution(shiftedpos2)) ))
        shiftedpos1[dim] -= h
        shiftedpos2[dim] += h
      #print(grad)
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
      
