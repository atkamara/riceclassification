# python general
import os,subprocess,time,multiprocessing,re
from tqdm.notebook import tqdm as tqdmnotebook
from tqdm import tqdm
from io import BytesIO

#collab
#from google.colab import drive

#images and plots
from PIL import Image

#array and frames
import numpy,pandas

#processing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

#machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

#deep learning
#from tensorflow import keras


#persisting tools
import joblib

import matplotlib.pyplot as plt

#graphs
#import networkx as nx
DATA_DIR                          = os.environ['RICEDATA']

to_dummies                        = lambda y,n : pandas.DataFrame(dict(map(lambda n0 : (n0,(y == n0) + 0),range(n))))
arrpath                           = lambda preview,arr,root=DATA_DIR,folder='fit_data',ext='.npy' : f'{root}/{folder}/{preview}/{arr}{ext}'
datafiles                         = lambda parent_directory,pattern,extension       : list(filter(re.compile(parent_directory+pattern).match,[os.path.join(r,f) for r, d, fs in os.walk(parent_directory) for f in fs  if f.endswith(extension) ]))
force_dir                         = lambda d : (os.makedirs(d,exist_ok=True),d)[1]
split_list_into_n_chunks          = lambda l,n :[l[i:i + int(len(l)/n)] for i in range(0, len(l), int(len(l)/n))]  
prodpath                          = lambda file,result,output=DATA_DIR,ext='.tsv' : force_dir(f'{output}/prod/{result}/')+f'{file}{ext}'



def custom_map(max_workers,f,L,process_dependant_funcs=[]) :
    if max_workers < 2:
        L_result                             = [*map(f,tqdm(L))]
    else:
        q                                    = multiprocessing.Manager().dict()
        def func_wrapper(l,q,id_)            :
            #this functions are process-dependant
            for h in process_dependant_funcs : h()
            res                              = [*map(f,tqdm(l))]
            q[f'{id_}']                      = res
            q[f'{id_}_not_looped']           = False
        processes                            = []
        for id_,l in enumerate(split_list_into_n_chunks(L,max_workers)):
            q[f'{id_}']                      = []
            q[f'{id_}_not_looped']           = True
            p                                = multiprocessing.Process(target=func_wrapper, args=(l,q,id_))
            p.start()
            processes.append(p)
        while any([
            q[k] 
            for k in q.keys() 
            if k.endswith('_not_looped')])  : 0
        for process in processes            : 
            process.kill()
            #process.close()
        L_result                            = sum([q[k] for k in q.keys() if not k.endswith('_not_looped')],[])
    return L_result



class MyRiceImgReader:
  '''

  Lists/Reads images per species

  valid species are : Arborio/Basmati/Ispala/Jasmine/Karacadag
  
  '''
  all_species = {'Karacadag': 0, 
                 'Jasmine': 1,
                 'Ipsala': 2,
                 'Arborio': 3,
                 'Basmati': 4}
  all_species_info = {
                  'Karacadag': {
                      'size':'medium',
                      'height':''}, 
                 'Jasmine': 1,
                 'Ipsala': 2,
                 'Arborio': {
                     'size':'',
                     'height':'short',
                     'origin':'italy',
                     'category':'aromatic'
                 },
                 'Basmati': {'size':'thin', # Ln/lr thin > 3, medium 2,wide 1, round<1 
                                 'height':'long(>6,61 mm)', #short <5.5 mm
                                 'origin':['india(65%)','pakistan'],
                                 'category':'aromatic',
                                 }}
  def __init__(self, 
               species  : str,
               zip      : str = 'archive',
               gs       : bool= False,
               max_imgs : int = None,
               readopt  : dict = {
                      'parent' : 'Rice_Image_Dataset',
                   'extension' : 'jpg'
               }
               )-> None:
        self.zip       = zip
        self.species   = species
        self.pattern   = lambda n=None,**kwargs :'/'.join((
                              'unzip -{opt} '+self.zip+' {parent}',
                              self.species,
                              '*\('+str(n or "*")+'\).{extension}'
                              ) 
                              ).format(**kwargs)
        self.readopt  = readopt
        self.gs       = gs
        self.index    = 0
        self.max_imgs = max_imgs
        self.pixels   = numpy.array(self[0]).shape
  def _PCA(self,arr,max_dim):
          im_shape      = arr.shape
          m = 1
          for i in arr.shape[1:]    : m*=i
          pca_shape     = arr.shape[0],m
          pca           = PCA(max_dim).fit(arr.reshape(*pca_shape)) 

          return pca,(im_shape,pca_shape)

  def _list_of_images(self): 
    self.all_files = [ 
                            line.split('  ')[-1]
                            for line in
                          subprocess.getoutput(
                          self.pattern(opt='l',**self.readopt)
                        ).split('\n') if line.endswith(
                            self.readopt['extension'])
        ]
    self.all_files.sort()
    return self.all_files

  def __getitem__(self,n)    : 
    """
    Return default Image obj

    or grayscaled Image obj

    """
    basmati_trick = lambda s : s if self.species != 'Basmati' \
                                     else (s.lower(),s)[n==0]
    jpgname       = basmati_trick(self.species)
    if n < len(self) :
        _file     = ('{parent}/'+f'{self.species}/{jpgname}\ \({n+1}\).jpg').format(**self.readopt)
        cmd       = ['unzip', '-p', self.zip,_file ] 
        imgio     = subprocess.run(cmd, stdout=subprocess.PIPE)
        imgio     = Image.open(BytesIO(imgio.stdout))
        if self.gs:
            imgio = numpy.array([row.mean(axis=1) for row in numpy.array(imgio)])
            imgio = Image.fromarray(imgio.astype('uint8') , 'L')
    else :
        raise IndexError
    return imgio
  def __len__(self)          : 
    size = self.max_imgs or len(self._list_of_images())
    return size
  def __iter__(self):
        return self

  def __next__(self): 
    if self.index < len(self): 
       arr = numpy.array(self[self.index])
       self.index += 1
       return arr
    else :
      raise StopIteration
  def write_fit_data(self,
                     output,
                     imparams,
                     finaltransf=lambda arr : arr.flatten()):
    os.makedirs(output,exist_ok=True)
    def to_hdf(arr,s0,s1):
      pandas.DataFrame(arr)\
        .T.pipe( lambda Df :Df if s0!=0 else Df.assign(y=MyRiceImgReader.all_species[self.species] )
           
           
      ).to_hdf(f'{output}/cols_{s0}_{s1}.h5',
                                    mode='a',
                                  format='table',
                                     key='Xy',
                                  append=True)  
    for arr in self.images_arrays(imparams) :
        arr   = finaltransf(arr)
        cols = [*range(0,arr.shape[0],2000)]+[arr.shape[0]]
        cols = zip(cols,cols[1:])
        for (s0,s1) in cols:
            to_hdf(arr[s0:s1],s0,s1)
        
  def Transformer(self,
                  
                  arr,
                  inverter_path=None,
                  enlarge_backgrd=False,
                 initsize=False,
                 max_dim=None,
                  remain_image=True,
                  pile_size=1):
            initshape = arr.shape

            if enlarge_backgrd:
              npixelsv      = arr.shape[0]*enlarge_backgrd
              npixelsh      = arr.shape[1]*enlarge_backgrd
                
              ndim          = (3,1)[arr.ndim==2]
              new_shape     = [npixelsv,npixelsh,ndim]
              im            = numpy.zeros(tuple(new_shape))
              for _ in range(pile_size):          
                randposv      = numpy.random.randint(0,new_shape[0]-2*arr.shape[0])
                randposh      = numpy.random.randint(0,new_shape[1]-2*arr.shape[1])  
                for i,dim in enumerate(arr.reshape( #250 times (250x3 or 250x1)
                        arr.shape[0],
                        arr.shape[1],
                        ndim)) :

                    im[i+randposv,randposh:randposh+arr.shape[1],:] = dim

            
              arr  = im
            if initsize and arr.ndim==3 and arr.shape[-1]==1 :
                arr      = numpy.array(Image.fromarray(arr.reshape(*arr.shape[:2]).astype('uint8'),mode='L').resize(initshape))
            if max_dim :
              self.pca,pca_shapes = self._PCA(arr,max_dim) 
              arr   = self.pca.transform(arr.reshape(*pca_shapes[1]))
              if remain_image:
                arr = self.pca.inverse_transform(arr).reshape(*pca_shapes[0])
              if inverter_path:
                os.makedirs((ivp:=f'{inverter_path}/{self.species}'),exist_ok=True)
                joblib.dump(self.pca.inverse_transform,f'{ivp}/{self.index}.joblib')
            return arr
  def images_arrays(self,
                    imparams):    
      for arr in tqdmnotebook(self,desc=f'{self.species}'):
          yield self.Transformer(arr,**imparams)
  def plot(self,size,figsize=(10,10)):
    fig,axes = plt.subplots(*size)
    fig.set_size_inches(*figsize)
    for i,ax in enumerate(axes.flatten()) :
      ax.imshow(self[i])
      ax.axes.axis('off')

class MyPCA:
  """
  Computes custom PCA projections
  
  """
  def __init__(self,
               M : numpy.array, #must be scaled
               real=True
               ) -> None :
    self.shape = M.shape
    cov        = (M.T.dot(M))/(self.shape[0]-1)
    self.eigv, \
     self.eigvc = numpy.linalg.eig(cov)
    if real:
      assert self.eigv.imag.sum() < 1e-6
      assert self.eigvc.imag.sum() < 1e-6
      self.eigv    = self.eigv.real
      self.eigvc   = self.eigvc.real
    self.c_xplain_var_ratio = self.eigv.cumsum()/self.eigv.sum()
    self.eigv      = sorted(enumerate(self.eigv),key=lambda x : x[1],reverse=True)

  def proj(self,X,k):
    picked = self.eigv[:k]
    pM     = self.eigvc[:,[x for x,_ in picked]]
    return X.dot(pM)

class MockReaderRandix(MyRiceImgReader):
  def __init__(self,
               species,
               max_imgs,
               gs):
    super().__init__(species,gs=gs)
    self.mx_im = max_imgs
    self.ix = 0
  @property
  def index(self):
    if self.ix < self.mx_im*3:
      self.ix += 1
      return numpy.random.randint(0,14999)
    else :
      self.max_imgs= self.mx_im
      return self.mx_im
  @index.setter
  def index(self,val):
      ...
class BatchStream(MyRiceImgReader):
  """
  
  Opens and transforms matrixes

  from shape n,k,max_dims to n,k,k,l
  
  """

  def __init__(self,
               X_filename,
               y_filename,
               max_dim,
               inverters_zip,
               parent,
               n_chunks=10,
               imshape=(250,250),
               as_im=True,
               gs=True):
    self.parent       = parent
    self.inverters_zip= inverters_zip
    self.imshape      = imshape
    self.as_im        = as_im
    self.nopen        = lambda f : numpy.load(open(f,'rb'),allow_pickle=True)
    self.X_f,\
          self.y_f    = X_filename,y_filename
    self.idx          = [
                          slice(s[0],s[-1]+1) 
                          for s in numpy.array_split(range(self.X.shape[0]),
                                                     n_chunks)
                          ]
    self.cidx         = 0
    self.n_chunks     = len(self.idx) 
    self.max_dim      = max_dim
    self.run          = 0
    
    cmd               = lambda spc: ['unzip','-p',self.inverters_zip,f'{self.parent}/inverse_transformers/{spc}.joblib']
    self.inv          ={ key : joblib.load(BytesIO(subprocess.run(cmd(spc),stdout=subprocess.PIPE).stdout))  
                        
                        for spc,key in MyRiceImgReader.all_species.items()
    }
  @property
  def X(self):
    return self.nopen(self.X_f)
  @property
  def y(self):
    return self.nopen(self.y_f)
  def __len__(self):
    return len(self.idx)
  def __iter__(self): 
    return self
  def __next__(self):
    if self.cidx < len(self) :
      cidx_l = self.idx[self.cidx]
      X,y    = (self.X[cidx_l],
              self.y[cidx_l])
      if self.as_im :
        X = self.to_img(X,y)
      self.cidx+=1
      return X,y
    raise StopIteration
  def to_img(self,X,y):
    X = X.reshape(len(X),self.imshape[0],self.max_dim) #from n,250*7 to n,250,7
    X = (x for x in X) #generator
    res = numpy.zeros((len(y),*self.imshape))
    
    for i,x in tqdmnotebook(enumerate(X),desc=f'{self.cidx}') :
      x      = self.inv[y[i]](x).reshape(*self.imshape)
      res[i] = x
      self.run+=i
    return res

class FitData(MyRiceImgReader):
    """
    - Dumps pca transformers 
    - Writes fit raw data
    - Loads X,y array
    - Dumps numpy array
    
    """
    def __init__(
        self,
        type_name,
        output=DATA_DIR,
        gs=True,
        preview={} 

        ):
        self.output        = output
        self.preview       = preview
        self.gs            = gs
        self.type_name     = type_name
        self.rpath         = f'{self.output}/inverters/{self.type_name}'
        self.rawdatapath   = f'{self.output}/raw_fit_data/{self.type_name}'
    def mean_images(self,n):
        sample_ims  = lambda spc : [*MockReaderRandix(spc,max_imgs=n,gs=self.gs).images_arrays(self.preview)]
        return {

            spc : numpy.mean(sample_ims(spc),axis=0).astype('uint8')

            for spc in self.all_species}
    def dump_transformers(self,n_comps,n_ims_sample=10,zip=False):
        self.ims    = self.mean_images(n_ims_sample)
        self.pcas   = {}
        for spc,im in self.ims.items():
          #paths  
          self.impath    = force_dir(f'{self.rpath}/mean_images/')+f'{spc}.jpeg'
          self.transpath = force_dir(f'{self.rpath}/transformers/')+f'{spc}.joblib'
          self.invpath   = force_dir(f'{self.rpath}/inverse_transformers/')+f'{spc}.joblib'
          #saving objects 
          Image.fromarray(im,mode='L').save(self.impath) 
          joblib.dump((pca:=PCA(n_comps).fit(im)).transform,self.transpath)
          joblib.dump(pca.inverse_transform,self.invpath)
          self.pcas[spc] = pca
          if zip:
            os.system(f'cd {self.output} && zip -r inverters.zip inverters && rm -rf inverters')
    def summarise(self):
      fig,axes = plt.subplots(1,5)
      fig.set_size_inches(10,10)
      for i,spc in enumerate(self.ims):
          axes[i].imshow(self.ims[spc])
          axes[i].axes.axis('off')
          axes[i].axes.set_title(spc)
      plt.show()
      return pandas.DataFrame({
          spc: pca.explained_variance_ratio_.cumsum()

          for spc,pca in self.pcas.items()
      }).rename('PC_{}'.format).tail(2)
    def write_rawdata(self,
                 max_imgs,
                 max_workers,
                 species=None):
        cmd               = lambda spc: ['unzip','-p',f'{self.output}/inverters.zip',f'inverters/{self.type_name}/transformers/{spc}.joblib']
        transfs           = { spc : joblib.load(BytesIO(subprocess.run(cmd(spc),stdout=subprocess.PIPE).stdout))  

                            for spc,key in self.all_species.items()
        }

        f = lambda spc : MyRiceImgReader(spc,
                                         gs = self.gs,
                                         max_imgs = max_imgs

                                    ).write_fit_data(
                                        f'{self.rawdatapath}/{spc}',
                                         imparams   = self.preview,
                                        finaltransf = lambda arr: transfs[spc](arr).flatten())
        custom_map(max_workers = max_workers,f = f,L = species or [*self.all_species])
    def map_rawdata_dir(self):
        _map = {}
        for spc in datafiles(self.rawdatapath,'.*/cols_\d+_\d+\.h5','h5') :
          (*_,spcname,_) = spc.split('/')
          _map[spcname]  = _map.get(spcname,[]) + [spc]
        return _map

    def save_fit_data(self,
                      test_rate):
        read_fit_data = lambda  rawdatadirs: pandas.concat( [

        pandas.concat([

                 pandas.read_hdf(f).reset_index(drop=True).pipe(

                     lambda D: D.assign(
                         file = f,
                         idx=D.index #last position

                    )
                 )
            for f in files
            ],
            axis=1

            ).dropna()
            
        for spc,files in [*rawdatadirs.items()]
          ],
        ignore_index=True
        ).sample(frac=1 #to shuffle data

                 ).pipe(lambda D : (D.drop('y',axis=1),D.y))
        self.rfitpath          = f'{self.output}/fit_data/{self.type_name}'
        os.makedirs(self.rfitpath,exist_ok=True)
        X,y                    = read_fit_data(self.map_rawdata_dir())
        X, X_test, y, y_test   = train_test_split(X, y, test_size=test_rate)
        Xix,X                  = X[['idx','file']],X.drop(['idx','file'],axis=1)
        X_testix,X_test        = X_test[['idx','file']],X_test.drop(['idx','file'],axis=1)
        numpy.save(open(f'{self.rfitpath}/X.npy','wb'),X.values)
        numpy.save(open(f'{self.rfitpath}/Xix.npy','wb'),Xix.values)
        numpy.save(open(f'{self.rfitpath}/y.npy','wb'),y.values)
        numpy.save(open(f'{self.rfitpath}/X_test.npy','wb'),X_test.values)
        numpy.save(open(f'{self.rfitpath}/X_testix.npy','wb'),X_testix.values)
        numpy.save(open(f'{self.rfitpath}/y_test.npy','wb'),y_test.values)
class Evaluator(MyRiceImgReader):
    def __init__(self,
                Zip   =f'{DATA_DIR}/inverters.zip',
                parent='inverters/gray'):

        cmd                   = lambda spc: ['unzip','-p',Zip,f'{parent}/transformers/{spc}.joblib']
        self.tranformers      = { key : joblib.load(BytesIO(subprocess.run(cmd(spc),stdout=subprocess.PIPE).stdout))  

                                for spc,key in self.all_species.items()
            }
        self.pca             = joblib.load(prodpath('fit_pca','projections',ext='.joblib'))

        #keras models
        Input   = keras.Input(shape=(100),name='Input')
        Dense0  = keras.layers.Dense(100, activation = 'relu',name='Hidden0')(Input)
        Drop0   = keras.layers.Dropout(rate=.1,name='Dropout0')(Dense0)
        Dense1  = keras.layers.Dense(50, activation = 'relu',name='Hidden1')(Drop0)
        Output  = keras.layers.Dense(5,   activation = 'softmax',name = 'Output')(Dense1)


        self.MLP    = keras.Model(Input, Output,name='MLPerceptron')

        Input   = keras.layers.Input(shape=(250,250,1),name='Input') 

        clayer0 = keras.layers.Convolution2D(25, kernel_size = 5, activation='relu',name='conv0' )(Input) 
        clayer1 = keras.layers.Convolution2D(50, kernel_size = 5, activation='relu',name='conv1' )(clayer0) 
        clayer2 = keras.layers.MaxPooling2D((5, 5),name='MaxPool0')(clayer1) 
        clayer3 = keras.layers.Flatten(name='Flatten')(clayer2) 


        Dense0  = keras.layers.Dense(75, activation = 'relu',name='Dense0')(clayer3)
        Drop0   = keras.layers.Dropout(rate=.2,name='Dropout0')(Dense0)
        Dense1  = keras.layers.Dense(50, activation = 'relu',name='Dense1')(Drop0)
        Output  = keras.layers.Dense(5,   activation = 'softmax',name = 'Output')(Dense1)


        self.Conv    = keras.Model(Input, Output,name='Convolution')


        keras.Model.load_weights(self.Conv,prodpath('Conv','fitted_models',ext=''))
        keras.Model.load_weights(self.MLP,prodpath('MLP','fitted_models',ext=''))
        self.evaluators      = {
        'RadmonForest'       : joblib.load(prodpath('Forest','fitted_models',ext='.joblib')),
        'LogisticRegression' : joblib.load(prodpath('Logistic','fitted_models',ext='.joblib')),
        'MLP'                : self.MLP,
        'Conv'               : self.Conv
        }
    def X_test(self,image):
        self.im = image
        self.im_array         = numpy.array([row.mean(axis=1) for row in numpy.array(self.im)])
        k                     = 100
        m,M                   = -1474.5641, 2537.4983
        transformed           = {
                            k : tr(self.im_array).flatten()
              for k,tr in self.tranformers.items()
        }
        transformed = numpy.array(list(transformed.values())).astype('float32')
        transformed -=m
        transformed/=(M-m)
        
        return self.pca.proj(transformed,k)
    def evaluate(self,image):
        X_test               = self.X_test(image)
        evaluations          = {
        'RadmonForest'       : pandas.DataFrame((self.evaluators['RadmonForest'].predict(X_test))).mode().values[0],
        'LogisticRegression' : to_dummies(self.evaluators['LogisticRegression'].predict(X_test),5).mode().values[0],
        'MLP'                : pandas.DataFrame((self.evaluators['MLP'].predict(X_test))).mode().values[0],
        'Conv'                : pandas.DataFrame((self.evaluators['Conv'].predict(X_test))).mode().values[0],
        }

        return pandas.DataFrame(evaluations).T.rename(columns=dict(zip(MyRiceImgReader.all_species.values(),
                                                         MyRiceImgReader.all_species.keys())))

def display_species(
    axes = (2,5),
    size = (10,5),
    
    
    spcs = ['Arborio',
            'Basmati',
            'Jasmine',
            'Karacadag',
            'Ipsala']
                   ) :
    images = {spc:[MyRiceImgReader(spc)[i] for i in range(2)] for spc in spcs}
    fig,axes = plt.subplots(*axes)
    fig.set_size_inches(*size)
    for i,spc in enumerate(images):
      for j,im in enumerate(images[spc]):
        axes[j][i].imshow(im)
        axes[j][i].axes.axis('off')
        if j == 0:
          axes[j][i].axes.set_title(spc)