class naivebayes:
    def fit(self, t_data, t_label):
        sample_rows, feature_col = t_data.shape
        self._classes = np.unique(t_label)
        numclasses = len(self._classes)
        
        self._mean = np.zeros((numclasses, feature_col), dtype = np.float64) #class size & features -> mean
        self._var = np.zeros((numclasses, feature_col), dtype = np.float64)
        self._prior = np.zeros((numclasses), dtype = np.float64) 
        
        for i in self._classes: #for each class we do
            data_i = t_data[i == t_label] #sample for i class as label
            self._mean[i : ] = data_i.mean(axis = 0) #for all columns
            self._var[i : ] = data_i.var(axis = 0) 
            self._prior[i] = data_i.shape[0] / float(sample_rows) # freq of how often i is occuring
    
    def predict(self, test_label):
        _predict = [self.predictsup(x) for x in test_label] #we call sup method for every sample here & loop it for each of it 
        return _predict
    
    def predictsup(self, single_test_label):
         #need to calculate posterior probability & class conditional. Choose one with highest Pr
            posterior = []
            for index, c in enumerate(self._classes): #c is class label
                prior = np.log(self._prior[index]) #have to apply log function for
                c_conditional = np.sum(np.log(self.probdensityfunc(index, single_test_label))) #have to apply gaussian function
                posterior_c = prior + c_conditional
                posterior.append(posterior_c)
                
            return self._classes[np.argmax(posterior)]
        
    def probdensityfunc(self, class_index, label):
            mean = self._mean[class_index] #conditional formula -> mean & var required
            var = self._var[class_index]
            num = np.exp(- (label - mean)**2 / (2 * var))
            denom = np.sqrt(2 * np.pi * var)
            return num / denom