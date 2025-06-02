import sys
from src.logger import logging

def error_message_detail(error, detail:sys):
    """
    Returns a detailed error message.
    
    Args:
        error (Exception): The exception to get the details from.
        detail (sys): The sys module to access exc_info.
    
    Returns:
        str: A formatted string containing the error type, value, and traceback.
    """
    _, _, exc_tb = detail.exc_info()
    #_, _, exc_tb = detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = "Error occured in python script '{0}' at line number {1} with error message: {2}".format(
        file_name, line_number, str(error)
    )

    return error_message

class CustomException(Exception):
    """Custom exception class that captures detailed error information."""
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, detail=error_detail)

    def __str__(self):
        return self.error_message
    

"""
if __name__ == "__main__":
    try:
        a = 1/0  # This will raise a ZeroDivisionError
    except Exception as e:
        logging.info("Divide by zero error occurred")
        raise CustomException(e,sys)
"""

##    logging.info("Logging has been set up successfully.")
