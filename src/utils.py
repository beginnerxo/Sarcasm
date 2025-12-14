import re 
import string 


def custom_preprocessor(text):
    
    """ This preprocesser seperates puntutation like "!" from teh input so we see 
        it as a distinct feature
    """
    
    if not isinstance(text,str):
        return " "
    
    
    text = text.lower() 
       
    text = re.sub(f"([{string.punctuation}])", r" \1 ", text) #adding a space after punctuation mark
    text = re.sub(r"\s+", " ", text).strip() # removing leading and trailing whitespaces
    
    return text