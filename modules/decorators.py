import pickle
import datetime

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
        
        pickling_on = open("_".join((date,func.__name__))+".pickle","wb")
        pickle.dump(commented_results,pickling_on)
        pickling_on.close()
        
        return result
    return wrapper