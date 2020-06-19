import pickle
import datetime
import os

def log_input_output(func):
    def wrapper(*args,**kwargs):
        
        date = datetime.datetime.now()
        date = "_".join( ("-".join( (str(date.year),
                                   str(date.month).zfill(2),
                                   str(date.day).zfill(2)) ),
                         str(date.hour).zfill(2)+
                         str(date.minute).zfill(2)) )
        
        result = func(*args,**kwargs)
        
        list_args = [str(ii) for ii in args]
        list_kwargs = [(k, v) for k, v in kwargs.items()]
        commented_results = [list_args+list_kwargs, result]
        
        path = os.path.join('output_data',
                              "_".join((date,func.__name__))+".pickle")
        pickling_on = open(path,"wb")
        pickle.dump(commented_results,pickling_on)
        pickling_on.close()
        
        return result
    return wrapper