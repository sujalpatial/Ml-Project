import sys
import traceback
import logging
def error_message_detail(error, error_detail: sys):
    """
    Returns a detailed error message including file name, line number, and error description.
    """
    _, _, exc_tb = error_detail.exc_info() #get trackback object
    file_name = exc_tb.tb_frame.f_code.co_filename # get filename
    error_message = f"Error occurred in Python script [{file_name}] at line [{exc_tb.tb_lineno}]: {str(error)}"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
