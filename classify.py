import numpy as np

def makeSimMat(data, simfunction):
    '''A convenience function for creating the symmetric similarity matrix for a dataset 'data' whose
    first dimension is the number of data objects.  The function 'simfunction() must compute a floating point
    similarity value ranging from 0 to 1, where 1 must be returned for any pair of identical objects. 
    Returns a symmetric matrix with dimension (N,N), where N = the size of the dataset.
    For some applications (e.g. calculation of correlation or Euclidean distance), it may be far more efficient
    to compute the similarity matrix instead using custom code that exploits vectorized Numpy operations.
    '''
  
    N = data.shape[0]
    simMat = np.zeros((N,N), dtype='float32')
    for i in range(N):
        for j in range(i+1):
            s = simfunction(data[i],data[j])
            if s > 1 or s < 0:
                raise Exception('Similarity function must return value between 0 and 1 for all data pairs')
            simMat[i,j] = simMat[j,i] = s
            
    return simMat
        
    
class simClassMap():
    '''Class and associated methods for classifying the members of a data set 
    using an arbitrary similarity metric, as described by Petty (2022).  
    
    An instance of the object with first-pass classifications is created by
    passing the data set 'data' as a Numpy array, with the first dimension N 
    giving the number of members of the data set. 
    The data array is retained only for possible future reference and is not 
    utilized in the classification algorithm, which instead relies on the precomputed
    NxN  symmetric similarity matrix 'simmat' and the desired scalar similarity threshold
    'thresh'. '''
    

    
    def __init__(self, data, simmat, thresh, verbose=False, checkinputs=False):
        
        if verbose:
            print('Data shape: ',data.shape)
            print('Matrix shape: ',simmat.shape)
            print('Similarity threshold used: ',thresh)
            
        N = simmat.shape[0]
     
        if checkinputs:  # Set to false when memory is a concern
            if not np.allclose(simmat, simmat.T):
                raise Exception("Matrix must be symmetric")
            
        
            if data.shape[0] != N :
                raise Exception("Data length %d doesn't match matrix dimension %d" % (data.shape[0],N))
            
            if not np.allclose(np.diag(simmat), np.ones((N))):
                raise Exception("Matrix must have all ones on the diagonal")
                           
        self.thresh = thresh
        self.data   = data   # Not used below, but saved for possible future reference
                           
    # Initialize output lists
        classPrototypes = []
        classMembers = []
        classAssigned = np.zeros(N).astype('int16')
    
    # Track original indices of rows and columns
        indexMap = np.arange(N).astype('int')

    # Initialize variables used in iteration
        unclassified = N
        classno = 0
        mask = (simmat >= thresh).astype('int8')

    # begin classification - one pass per class found
        while unclassified > 0:
                           
            if verbose: print('Finding Class %d' % (classno+1) , flush=True)
                           
            cp = np.argmax(mask.sum(axis=1))      # Find prototype for new class
            classPrototypes.append(indexMap[cp])  # Save index number for prototype
            m,  = np.nonzero(mask[cp])            # Find members of the new class
            unclassified -= len(m)
                           
            if verbose: print("%d members, %d unclassified\n" % (len(m), unclassified))
                           
            members = indexMap[m]
            classMembers.append(members)          # Save list of members
            classno += 1
            classAssigned[members] = classno      # Update map of class assignments (numbered from 1)
            mask = np.delete(mask, m, axis=0)
            mask = np.delete(mask, m, axis=1)
            indexMap = np.delete(indexMap, m)
        
        classPrototypes = np.array(classPrototypes) # Convert list to numpy array
        
        self.simmat = simmat
        self.prototypes = classPrototypes
        self.members = classMembers
        self.map = classAssigned
        self.totalcount = len(self.map)
        
        classIndex = []
        for i in range(len(self.prototypes)):
            iclass = i + 1
            count = len(self.members[i])
            classIndex.append({'class': iclass, 'prototype': self.prototypes[i], 
                          'count' : count, 'members' : self.members[i], 'fraction' : count/self.totalcount})
        self.index = classIndex
        
    def reassign(self):
        '''This method reassigns all dataset members to the nearest prototype 
        determined in the first-pass classification.  It preserves the original 
        class numbering and ordering. It's possible, however, that classes will no longer 
        be listed in order of decreasing size. It may therefore be desirable to call the additional
        method .sort() to renumber and reorder classes according to size. '''
        
        submat = self.simmat[self.prototypes, :]
        self.map = np.argmax(submat,axis=0)+1  # Classes are numbered from one
        self.members = []
        for i in range(np.max(self.map)):  
            iclass = i+1
            self.members.append(np.where(self.map==iclass)[0])  
  
        
        self.index = []                        # Construct list of classes with all attributes as dicts
        for i in range(len(self.prototypes)):
            iclass = i + 1
            count = len(self.members[i])
            self.index.append({'class': iclass, 'prototype': self.prototypes[i], 
                 'count' : count, 'members' : self.members[i],'fraction' : count/self.totalcount})
           
    def sort(self):
        ''' This method sorts previously found classes by size and renumbers them accordingly.
        The .map attribute is updated to reflect the new numbering. '''
        
        self.index = sorted(self.index, key=lambda c: c['count'], reverse=True)   # classes sorted by size
        self.map = np.zeros(len(self.map)).astype('int')  # zero out and prepare to reconstruct assignments
        p = []
        self.members = []
        for i,c in enumerate(self.index):  # Loop over sorted classes
            p.append(self.prototypes[i])
            members = c['members']
            iclass = i + 1
            self.index[i]['class'] = iclass
            self.map[members] = iclass
            self.members.append(members)
            
        self.prototypes = np.array(p)
        
    def truncate(self, maxclasses):
        
        ''' This method discards all but the first 'maxclasses' classes and reassigns all
        data set members to the classes that were retained. It sorts the results so
        that classes are again listed and numbered in order of decreasing size. 
        This method should normally be called only on a previously sorted classification
        so that the largest classes are retained.'''

        self.prototypes = self.prototypes[0:maxclasses]
        self.reassign()
        self.sort()
        
    def print(self, maxlines=1000):
        
        ''' A convenience method for printing the current list of classes. If the number of 
        classes exceeds the keyword option maxlines, then only maxlines classes are printed, 
        including the first and last maxlines//2 classes.'''

        ncum = 0
        fcum = 0.
        skipflag = True
        nclasses = len(self.index)
        if nclasses > maxlines:
            skip1 = maxlines//2
            skip2 = nclasses-maxlines//2
        else:
            skip1 = skip2 = nclasses + 100
        print('Class Prototype Count  Fraction      Totals')
        for i,c in enumerate(self.index):
            ncum += c['count']
            fcum += c['fraction']
            if i < skip1 or i > skip2:
                print('%4d %8d %6d %8.3f   %6d %8.3f' % (c['class'],c['prototype'],
                        c['count'],c['fraction'], ncum, fcum))
            else:
                if skipflag:
                    print('...')
                    skipflag = False
                           
    def augment(self, datafull, simmat2):
          
        ''' This method allows a larger dataset to be assigned to previously determined classes. 
        'datafull' is Numpy array that is assumed to include the originally classified smaller data 
        set 'data' as a subset. The argument 'simmat2' is an external computed MxP similarity matrix,
        where M is the number of elements in 'datafull', and P is the number of classes (and associated
        prototypes) retained from classification of the original data set.
        
        The results do not overwrite the original class definitions and member lists but are added
        as new attributes with the suffix '2'   '''
            
        M = datafull.shape[0]
        if M != simmat2.shape[0]:
            raise Exception("First dimension %d of data array doesn't match first dimension %d of similarity matrix" 
                            % (M, simmat2.shape[0]))
            
        self.data2 = datafull
        self.simmat2 = simmat2
        self.prototypes2 = np.argmax(simmat2, axis=0)      # Find indices of original prototypes in new data set
        self.map2 = np.argmax(simmat2, axis=1) + 1         # Assign new data points to nearest class
        self.totalcount2 = M
        
        classMembers2 = []
        for iclass in range(1,np.max(self.map2)+1):        # Classes are numbered from one
            classMembers2.append(np.where(self.map2==iclass))
        self.members2 = classMembers2
        
        classIndex = []
        for i in range(len(self.prototypes2)):
            iclass = i + 1
            count = len(self.members2[i])
            classIndex.append({'class': iclass, 'prototype': self.prototypes2[i], 
                          'count' : count, 'members' : self.members2[i], 
                               'fraction' : count/self.totalcount2})
        self.index2 = classIndex
                           
        
        
        
    
    


        
    