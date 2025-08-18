import sys

def error_msg(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    return f"Error occured in {filename} in the line {exc_tb.tb_lineno} and the error is {str(error)}"

class CustomException:
    def __init__(self, error, error_detail:sys):
        super().__init__(error)
        self.error_message = error_msg(error, error_detail)
    
    def __str__(self):
        return self.error_message