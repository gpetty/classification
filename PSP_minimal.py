def PSP(simmat, thresh):
''' simmat:  a square, symmetric Numpy array representing the pairwise 
             similarities between data set members
    thresh:  the similarity threshold to use in the classification '''

    import numpy as np
        
    N = simmat.shape[0]
                           
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
        cp = np.argmax(mask.sum(axis=1))      # Find prototype for new class
        classPrototypes.append(indexMap[cp])  # Save index number for prototype
        m,  = np.nonzero(mask[cp])            # Find members of the new class
        unclassified -= len(m)
        members = indexMap[m]
        classMembers.append(members)          # Save list of members
        classno += 1
        classAssigned[members] = classno      # Update map of class assignments
        mask = np.delete(mask, m, axis=0)     # Eliminate already assigned members
        mask = np.delete(mask, m, axis=1)
        indexMap = np.delete(indexMap, m)
        
    return classAssigned, classPrototypes, classMembers


